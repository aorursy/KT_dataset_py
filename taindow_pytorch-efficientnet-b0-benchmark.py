# Input



dir_csv = '../input/rsna-intracranial-hemorrhage-detection'

dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'

dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'



# Parameters



n_classes = 6

n_epochs = 5

batch_size = 128

# Installing useful libraries



!git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

!pip install --upgrade efficientnet-pytorch

    
# Libraries



from apex import amp

import os

import cv2

import glob

import pydicom

import numpy as np

import pandas as pd

from efficientnet_pytorch import EfficientNet

import torch

import torch.optim as optim

from albumentations import Compose, ShiftScaleRotate, Resize

from albumentations.pytorch import ToTensor

from torch.utils.data import Dataset

from tqdm import tqdm_notebook as tqdm

from matplotlib import pyplot as plt


# Functions



class IntracranialDataset(Dataset):



    def __init__(self, csv_file, path, labels, transform=None):

        

        self.path = path

        self.data = pd.read_csv(csv_file)

        self.transform = transform

        self.labels = labels



    def __len__(self):

        

        return len(self.data)



    def __getitem__(self, idx):

        

        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')

        img = cv2.imread(img_name)   

        

        if self.transform:       

            

            augmented = self.transform(image=img)

            img = augmented['image']   

            

        if self.labels:

            

            labels = torch.tensor(

                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])

            return {'image': img, 'labels': labels}    

        

        else:      

            

            return {'image': img}

    

    

# CSVs



train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))

test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))


# Split train out into row per image and save a sample



train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)

train = train[['Image', 'Diagnosis', 'Label']]

train.drop_duplicates(inplace=True)

train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()

train['Image'] = 'ID_' + train['Image']

train.head()
# Some files didn't contain legitimate images, so we need to remove them



png = glob.glob(os.path.join(dir_train_img, '*.png'))

png = [os.path.basename(png)[:-4] for png in png]

png = np.array(png)



train = train[train['Image'].isin(png)]

train.to_csv('train.csv', index=False)
# Also prepare the test data



test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)

test['Image'] = 'ID_' + test['Image']

test = test[['Image', 'Label']]

test.drop_duplicates(inplace=True)



test.to_csv('test.csv', index=False)
# Data loaders



transform_train = Compose([

    ShiftScaleRotate(),

    ToTensor()

])



transform_test= Compose([

    ToTensor()

])



train_dataset = IntracranialDataset(

    csv_file='train.csv', path=dir_train_img, transform=transform_train, labels=True)



test_dataset = IntracranialDataset(

    csv_file='test.csv', path=dir_test_img, transform=transform_test, labels=False)



data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# Plot train example



batch = next(iter(data_loader_train))

fig, axs = plt.subplots(1, 5, figsize=(15,5))



for i in np.arange(5):

    

    axs[i].imshow(np.transpose(batch['image'][i].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)

# Plot test example



batch = next(iter(data_loader_test))

fig, axs = plt.subplots(1, 5, figsize=(15,5))



for i in np.arange(5):

    

    axs[i].imshow(np.transpose(batch['image'][i].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)

# Model



device = torch.device("cuda:0")

model = EfficientNet.from_pretrained('efficientnet-b0') 

model._fc = torch.nn.Linear(1280, n_classes)



model.to(device)



criterion = torch.nn.BCEWithLogitsLoss()

plist = [{'params': model.parameters(), 'lr': 2e-5}]

optimizer = optim.Adam(plist, lr=2e-5)



model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# Train





for epoch in range(n_epochs):

    

    print('Epoch {}/{}'.format(epoch, n_epochs - 1))

    print('-' * 10)



    model.train()    

    tr_loss = 0

    

    tk0 = tqdm(data_loader_train, desc="Iteration")



    for step, batch in enumerate(tk0):



        inputs = batch["image"]

        labels = batch["labels"]



        inputs = inputs.to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.float)



        outputs = model(inputs)

        loss = criterion(outputs, labels)



        with amp.scale_loss(loss, optimizer) as scaled_loss:

            scaled_loss.backward()



        tr_loss += loss.item()



        optimizer.step()

        optimizer.zero_grad()



    epoch_loss = tr_loss / len(data_loader_train)

    print('Training Loss: {:.4f}'.format(epoch_loss))
# Inference



for param in model.parameters():

    param.requires_grad = False



model.eval()



test_pred = np.zeros((len(test_dataset) * n_classes, 1))



for i, x_batch in enumerate(tqdm(data_loader_test)):

    

    x_batch = x_batch["image"]

    x_batch = x_batch.to(device, dtype=torch.float)

    

    with torch.no_grad():

        

        pred = model(x_batch)

        

        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(

            pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))
# Submission



submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))

submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)

submission.columns = ['ID', 'Label']



submission.to_csv('submission.csv', index=False)

submission.head()
!rm -rf /kaggle/working/apex

!rm test.csv

!rm train.csv