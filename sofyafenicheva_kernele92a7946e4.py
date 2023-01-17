# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dir = '../input/iwildcam-2019-fgvc6/'

test_dir = '../input/iwildcam-2019-fgvc6/'
import os

import cv2

import math



from PIL import Image

# Помимо категории, есть еще полезная информация такие как время съемки и локация. 

# помимо изображений в сети буду использовать также 

train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'))

train_df.head()
test_df = pd.read_csv(os.path.join(test_dir, 'test.csv'))

test_df.head()
def get_time_df(df):

    try:

        df['date_time'] = pd.to_datetime(df['date_captured'], errors='coerce')

        df["month"] = df['date_time'].dt.month - 1

        df["hour"] = df['date_time'].dt.hour

    except Exception as ex:

        print("Exception:".format(ex))

    df.loc[np.isfinite(df['hour']) == False, ['month', 'hour']] = 0

    df['hour'] = df['hour'].astype(int)

    df['month'] = df['month'].astype(int)

    return df

train_df = get_time_df(train_df)

test_df = get_time_df(test_df)
from torchvision import models

import torch.nn.functional as F

import torch.optim as optim

import torch.nn as nn

import torch





class SimpleConv(nn.Module):

    def __init__(self, num_categories, len_dense, weighs):

        super(SimpleConv, self).__init__()

        self.model_conv = models.resnet152(pretrained=False)

        if weighs:

            self.model_conv.load_state_dict(torch.load(weighs))

        self.model_conv.fc = nn.Linear(self.model_conv.fc.in_features, num_categories)

        self.model_dense = nn.Linear(len_dense, num_categories)

        self.model = nn.Linear(2*num_categories, num_categories)

    

    def forward(self, x, y):

        x1 = self.model_conv(x)

        x2 = self.model_dense(y)

        x = F.relu(torch.cat((x1, x2), 1))

        return self.model(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

simple_conv = SimpleConv(23, 12+24, '../input/resnet152/resnet152.pth')

simple_conv = simple_conv.to(device)
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader



class SimpleDataset(Dataset):

    def __init__(self, folder, df, n_category, transform=None):

        self.transform = transform

        self.root_dir = folder

        self.df = df

        self.y = np.array(df.get('category_id', []))

#         self.y = np.eye(n_category)[category_ids]

        month = np.eye(12)[df.month.tolist()]

        hours = np.eye(24)[df.hour.tolist()]

        self.time = np.concatenate((month, hours), axis=1)

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name = os.path.join(self.root_dir, self.df.file_name[index])

        image = Image.open(img_name)

        if len(self.y):

            label = self.y[index]

        else:

            label = 0

        image = Image.open(img_name).convert('RGB')

        time = torch.from_numpy(self.time[index]).float()

        

        if self.transform:

            image = self.transform(image)

        return image, time, label, self.df.id[index]

from torchvision import transforms



data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((150, 150)),

        transforms.RandomResizedCrop((128, 128)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test': transforms.Compose([

        transforms.Resize((128, 128)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
train_ds = SimpleDataset(os.path.join(train_dir, 'train_images'), train_df, n_category=23, transform=data_transforms['train'])

test_ds = SimpleDataset(os.path.join(train_dir, 'test_images'), test_df, n_category=23, transform=data_transforms['test'])
batch_size = 256



data_size = len(train_ds)

validation_fraction = .2



indices = list(range(data_size))

data_size = len(indices)

val_split = int(np.floor((validation_fraction) * data_size))



np.random.seed(42)

np.random.shuffle(indices)



val_indices, train_indices = indices[:val_split], indices[val_split:]



train_sampler = SubsetRandomSampler(train_indices)

val_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 

                                           sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,

                                         sampler=val_sampler)

# Notice that we create test data loader in a different way. We don't have the labels

train_sampler = SubsetRandomSampler(train_indices)

# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512)
import copy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score





def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    

    loss_history = []

    train_history = []

    val_history = []

    best_acc = 0.0

    best_model_wts = copy.deepcopy(model.state_dict())

    

    for epoch in range(num_epochs):

        model.train() # Enter train mode

        

        loss_accum = 0

        correct_samples = 0

        total_samples = 0



        for i_step, (x1, x2, y, _) in enumerate(train_loader):

          

            x1_gpu = x1.to(device)

            x2_gpu = x2.to(device)

            y_gpu = y.to(device)

            

            prediction = model(x1_gpu, x2_gpu)    

            loss_value = loss(prediction, y_gpu)

            optimizer.zero_grad()

            loss_value.backward()

            optimizer.step()

            _, indices = torch.max(prediction, 1)

            correct_samples += torch.sum(indices == y_gpu)

            total_samples += y.shape[0]

            

            loss_accum += loss_value

#             print('{}/{}'.format(i_step, len(train_loader)))



        ave_loss = loss_accum / i_step

        train_accuracy = float(correct_samples) / total_samples

        val_f1, val_accuracy = compute_accuracy(model, val_loader)

        

        loss_history.append(float(ave_loss))

        train_history.append(train_accuracy)

        val_history.append(val_accuracy)

        

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f val F1: %f" % (ave_loss, train_accuracy, val_accuracy, val_f1))

        if val_f1 > best_acc:

            best_acc = val_f1

            best_model_wts = copy.deepcopy(model.state_dict())

            

    return loss_history, train_history, val_history, best_model_wts





def compute_accuracy(model, loader):

    model.eval() 

    correct = 0

    total = 0

    predictions = np.empty(shape=len(val_sampler)).astype(int)

    ground_truth = np.empty(shape=len(val_sampler)).astype(int)

    

    with torch.no_grad():

        for i,(x1, x2, y, _) in enumerate(loader):

            begin = i*batch_size

            x1_gpu = x1.to(device)

            x2_gpu = x2.to(device)

            y_gpu = y.to(device)

            

            outputs = model(x1_gpu, x2_gpu)

            _, predicted = torch.max(outputs.data, 1)

            total += y_gpu.size(0)

            correct += (predicted == y_gpu).sum().item()

#             print(predictions.shape, np.array(predicted.cpu()).shape)

#             print(begin, len(predictions), min(begin+batch_size, len(predictions)))

            predictions[begin : min(begin+batch_size, len(predictions))] = np.array(predicted.cpu())

            ground_truth[begin : min(begin+batch_size, len(ground_truth))] = np.array(y_gpu.cpu())

        val_f1 = f1_score(predictions, ground_truth, average='macro')

    return val_f1, correct / total
optimizer = optim.SGD(simple_conv.parameters(), lr=0.01, momentum=0.9)
loss = nn.CrossEntropyLoss()

result = train_model(simple_conv, train_loader, val_loader, loss, optimizer, num_epochs=2)

simple_conv.load_state_dict(result[3])
image_id = []

batch_size = test_loader.batch_size

predictions = np.zeros(shape=len(test_ds)).astype(int)

simple_conv.eval()

with torch.no_grad():

    for i,(x1,x2,_,id_img) in enumerate(test_loader):

#         print('{}/{}'.format(i+1, len(test_loader)))

        begin = i*batch_size

        x1_gpu = x1.to(device)

        x2_gpu = x2.to(device)

        outputs = simple_conv(x1_gpu, x2_gpu)

        _, predicted = torch.max(outputs.data, 1)

        predictions[begin : begin+len(predicted)] = np.array(predicted.cpu())

        image_id += id_img

submission_df = pd.read_csv(os.path.join('../input/iwildcam-2019-fgvc6','sample_submission.csv'),delimiter=',')

# submission_df['Predicted'] = predictions
import csv



with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerow(['Id', 'Predicted'])

    writer.writerows(zip(submission_df.Id.tolist(),predictions))