import numpy as np

import pandas as pd



import time



# path management

from pathlib import Path



# progress bars

from tqdm import tqdm



# plotting

import matplotlib.pyplot as plt

import seaborn as sn



# pytorch

import torch

from torch.utils.data import Dataset

from torch.utils.data import DataLoader



import torch.nn as nn

import torch.nn.functional as F



from torch.optim import SGD, Adam



# zipfile management

import zipfile
comp_data_path = Path('../input/rsna-str-pulmonary-embolism-detection')

prep_data_path = Path('../input/2020pe-preprocessed-train-data')
# set sizing

NSCANS = 20

NPX = 128



# set device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
allfiles = list((prep_data_path / f'proc_{NSCANS}_{NPX}_train').glob('*_data.npy'))

sample_file = allfiles[0]

sample_scans = np.load(str(sample_file), allow_pickle=True)



fig, ax = plt.subplots(5, 4, figsize=(20,20))

ax = ax.flatten()

for m in range(NSCANS):

    ax[m].imshow(sample_scans[m], cmap='Blues_r')
train = pd.read_csv(prep_data_path / 'train_proc.csv', index_col=0)

train.head()
class PE2020Dataset(Dataset):

    def __init__(self, scans, labels, datapath):

        self.scans = scans

        self.labels = labels

        self.datapath = Path(datapath)



    def __len__(self):

        return len(self.scans)



    def __getitem__(self, i):

        file = self.datapath / (self.scans[i] + '_data.npy')

        x = torch.tensor(np.load(str(file)), dtype=torch.float).to(device)

        y = torch.tensor(self.labels[i]).to(device)

        return x, y
class PE2020Net(nn.Module):

    def __init__(self, NOUT):

        super(PE2020Net, self).__init__()

        self.conv1 = nn.Conv2d(NSCANS, 32, 5)

        self.conv2 = nn.Conv2d(32, 64, 5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(64*29*29, 500)

        #self.fc2 = nn.Linear(5000, 5000)

        self.fc3 = nn.Linear(500, 3)



    def forward(self, x):

        x = x.view(-1, NSCANS, NPX, NPX)

        x = self.conv1(x)

        x = F.max_pool2d(x, 2)

        x = F.relu(x)

        x = self.conv2(x)

        x = self.conv2_drop(x)

        x = F.max_pool2d(x, 2)

        x = F.relu(x)

        x = x.view(-1, 64*29*29)

        x = self.fc1(x)

        x = F.dropout(x, training=self.training)

        x = F.relu(x)

        #x = self.fc2(x)

        #x = F.relu(x)

        x = self.fc3(x)

        x = F.log_softmax(x, dim=-1)

        return x
# separate scans from labels

scans = train['dcmpath']

all_labels = train.drop(labels='dcmpath', axis=1).astype(int)

labels = all_labels[['negative_exam_for_pe', 'indeterminate']].copy()

labels['positive_exam_for_pe'] = 1 - labels[['negative_exam_for_pe', 'indeterminate']].sum(axis=1)



# keep label names for later

label_names = labels.columns.tolist()

# get label index for all studies

labels = np.where(labels.values)[1]



# split into train and validation and create datasets

tmp = train.sample(len(train)).index.values

idx_train = tmp[:int(0.8*len(tmp))]

idx_valid = tmp[int(0.8*len(tmp)):]

trainset = PE2020Dataset(scans.loc[idx_train].values, labels[idx_train], prep_data_path / f'proc_{NSCANS}_{NPX}_train')

validset = PE2020Dataset(scans.loc[idx_valid].values, labels[idx_valid], prep_data_path / f'proc_{NSCANS}_{NPX}_train')



# instantiate DataLoaders

BATCHSIZE = 20

trainloader = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True)

validloader = DataLoader(validset, batch_size=BATCHSIZE, shuffle=True)
model = PE2020Net(3).to(device)

optimiser = Adam(model.parameters(), lr=0.005)

epochs = 20

accuracy = {}

accuracy['train'] = []

accuracy['valid'] = []



for epoch in range(epochs):

    

    # training

    correct = 0

    losses = torch.tensor([])

    for batch_data in tqdm(trainloader):

        X, y = batch_data

        # zero gradients

        model.zero_grad()

        # forward pass

        output = model(X)

        

        # count accurate predictions

        predicted = torch.max(output.detach(),1)[1]

        correct += (predicted == y).sum()

        

        # calculate loss and keep it for later

        loss = F.cross_entropy(output, y)

        losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)

        

        # backward pass

        loss.backward()

        # update parameters

        optimiser.step()

    

    # calculate mean loss and accuracy and print

    meanacc = float(correct) / (len(trainloader) * BATCHSIZE)

    meanloss = float(losses.mean())

    print('Epoch:', epoch, 'Loss:', meanloss, 'Accuracy:', meanacc)

    accuracy['train'].append(meanacc)

    # putting this in to keep the console clean

    time.sleep(0.5)

    

    

    

    # validation

    correct = 0

    for batch_data in tqdm(validloader):

        # forward pass

        X, y = batch_data

        with torch.no_grad():

            output = model(X)

        # get number of accurate predictions

        predicted = torch.max(output,1)[1]

        correct += (predicted == y).sum()

    # calculate mean accuracy and print

    meanacc = float(correct) / (len(validloader) * BATCHSIZE)

    print('Validation epoch:', epoch, 'Accuracy:', meanacc)

    accuracy['valid'].append(meanacc)

    # putting this in to keep the console clean

    time.sleep(0.5)
plt.plot(accuracy['train'],label="Training Accuracy")

plt.plot(accuracy['valid'],label="Validation Accuracy")

plt.xlabel('No. of Epochs')

plt.ylabel('Accuracy')

plt.legend(frameon=False)

plt.show()
torch.save(model, 'stg1_model.pt')

np.save('stg1_label_names', label_names)
validset = iter(validloader)

all_y = np.array([])

all_y_pred = np.array([])

for x, y in validset:

    with torch.no_grad():

        y_pred = model(x)

    all_y = np.append(all_y, y.tolist())

    all_y_pred = np.append(all_y_pred, torch.max(y_pred,1)[1].tolist())

    

df = pd.DataFrame(np.array([all_y, all_y_pred]).T, columns=['y','y_pred'])

confusion_matrix = pd.crosstab(df['y'], df['y_pred'], rownames=['y'], colnames=['y_pred'])



sn.heatmap(confusion_matrix, annot=True)

plt.show()
# choose only PE-positive samples

train_pe_positive = train[(train['negative_exam_for_pe'] == 0) &

                          (train['indeterminate'] == 0)]



# separate scans from labels

scans = train_pe_positive['dcmpath']

all_labels = train_pe_positive.drop(labels='dcmpath', axis=1).astype(int)

labels = all_labels[['acute_pe', 'chronic_pe', 'acute_and_chronic_pe']].copy()



# keep label names for later

label_names = labels.columns.tolist()

# get label index for all studies

labels['label'] = np.where(labels.values)[1]



# split into train and validation and create datasets

tmp = train_pe_positive.sample(len(train_pe_positive)).index.values

idx_train = tmp[:int(0.8*len(tmp))]

idx_valid = tmp[int(0.8*len(tmp)):]

trainset = PE2020Dataset(scans.loc[idx_train].values, labels.loc[idx_train, 'label'].values,

                         prep_data_path / f'proc_{NSCANS}_{NPX}_train')

validset = PE2020Dataset(scans.loc[idx_valid].values, labels.loc[idx_train, 'label'].values,

                         prep_data_path / f'proc_{NSCANS}_{NPX}_train')



# instantiate DataLoaders

BATCHSIZE = 20

trainloader = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True)

validloader = DataLoader(validset, batch_size=BATCHSIZE, shuffle=True)
model = PE2020Net(3).to(device)

optimiser = Adam(model.parameters(), lr=0.005)

epochs = 20

accuracy = {}

accuracy['train'] = []

accuracy['valid'] = []



for epoch in range(epochs):

    

    # training

    correct = 0

    losses = torch.tensor([])

    for batch_data in tqdm(trainloader):

        X, y = batch_data

        # zero gradients

        model.zero_grad()

        # forward pass

        output = model(X)

        

        # count accurate predictions

        predicted = torch.max(output.detach(),1)[1]

        correct += (predicted == y).sum()

        

        # calculate loss and keep it for later

        loss = F.cross_entropy(output, y)

        losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)

        

        # backward pass

        loss.backward()

        # update parameters

        optimiser.step()

    

    # calculate mean loss and accuracy and print

    meanacc = float(correct) / (len(trainloader) * BATCHSIZE)

    meanloss = float(losses.mean())

    print('Epoch:', epoch, 'Loss:', meanloss, 'Accuracy:', meanacc)

    accuracy['train'].append(meanacc)

    # putting this in to keep the console clean

    time.sleep(0.5)

    

    

    

    # validation

    correct = 0

    for batch_data in tqdm(validloader):

        # forward pass

        X, y = batch_data

        with torch.no_grad():

            output = model(X)

        # get number of accurate predictions

        predicted = torch.max(output,1)[1]

        correct += (predicted == y).sum()

    # calculate mean accuracy and print

    meanacc = float(correct) / (len(validloader) * BATCHSIZE)

    print('Validation epoch:', epoch, 'Accuracy:', meanacc)

    accuracy['valid'].append(meanacc)

    # putting this in to keep the console clean

    time.sleep(0.5)
plt.plot(accuracy['train'],label="Training Accuracy")

plt.plot(accuracy['valid'],label="Validation Accuracy")

plt.xlabel('No. of Epochs')

plt.ylabel('Accuracy')

plt.legend(frameon=False)

plt.show()
torch.save(model, 'stg2_model.pt')

np.save('stg2_label_names', label_names)
validset = iter(validloader)

all_y = np.array([])

all_y_pred = np.array([])

for x, y in validset:

    with torch.no_grad():

        y_pred = model(x)

    all_y = np.append(all_y, y.tolist())

    all_y_pred = np.append(all_y_pred, torch.max(y_pred,1)[1].tolist())

    

df = pd.DataFrame(np.array([all_y, all_y_pred]).T, columns=['y','y_pred'])

confusion_matrix = pd.crosstab(df['y'], df['y_pred'], rownames=['y'], colnames=['y_pred'])



sn.heatmap(confusion_matrix, annot=True)

plt.show()
label_names