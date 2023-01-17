import os

from tqdm import tqdm

import shutil 

import pydicom

from PIL import Image

import cv2



# def extract_dicom_images(rsna_dir, ):

#     '''

#     This function extracts jpg images from the dicom dataset(both train and test) and 

#     stores it in the ./data folder

#     Params:

#     rsna_dir: directory of the rsna dataset

#     '''

#     train_dir = os.path.join(rsna_dir, 'stage_2_train_images')

#     try:  

#         os.mkdir('data')  

#     except OSError as error:  

#         print('The directory already exists deleting the contents of the directiry')

#         shutil.rmtree('./data')

#         os.mkdir('data') 

#     outdir = './data/'

#     for file in tqdm(os.listdir(train_dir)):

#         file_path = os.path.join(train_dir, file)

#         ds = pydicom.read_file(file_path) # read dicom image

#         img = ds.pixel_array # get image array

#         img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)

#         img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface

#         img_mem.save(outdir + file.replace('.dcm','.png'))

        

        

        

    

    

        

    

    

    

# extract_dicom_images(rsna_dir = '../input/rsna-pneumonia-detection-challenge')

    
import torch

import pandas as pd

def get_labeller(df_dir):

    '''

    Returns a dictionary which maps patiendId to labels

    0: ND & 1:D and bounding boxes

    Params:

    df_dir: directory of the df with this info

    '''

    df = pd.read_csv(df_dir)

    df = df.set_index('patientId')

    return df.T.to_dict()



# get_labeller('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
import random

def get_all_files():

    labeller = get_labeller('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

    files = list(labeller.keys())

    print(f'Found {len(files)} files')

    random.shuffle(files)

    train_files = files[0:int(len(files) *0.80)]

    test_files = files[int(len(files) *0.80):]

    return train_files, test_files

# train, test = get_all_files()

# print(len(train), len(test))
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms

from PIL import Image

class DS(Dataset):

    def __init__(self, labeller, files, mode, base_path):

        '''

        Params:

        labeller: dict to map patID to target

        files: list of files to be trained on

        mode = train or test

        '''

        self.labeller = labeller

        self.files = files

        self.mode = mode

        self.trans_tr = transforms.Compose([

            transforms.Resize(256),

            transforms.ColorJitter(),

            transforms.RandomCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.Resize(128),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225]) 

        ])

        self.trans_test = transforms.Compose([

            transforms.Resize((128,128)),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225]) 

        ])

        self.trans = None

        if self.mode == 'train':

            self.trans = self.trans_tr

        else:

            self.trans = self.trans_test

        self.base_path = base_path

        

    def __len__(self):

        return  len(self.files)

    

    def __getitem__(self, idx):

        img_name = self.files[idx]

        path = os.path.join(self.base_path, img_name+'.png')

        img =  Image.open(path).convert('RGB')

        img = self.trans(img)

        img = img.numpy()

        return img.astype('float32'), self.labeller[img_name]['Target']

    

    

def get_dataloaders():

    labeller = get_labeller('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

    base_dir = '../input/rsna-pneumionia-detection/data/'

    files_train, files_test = get_all_files()

    

    train_ds = DS(labeller, files_train, 'train', base_dir)

    test_ds = DS(labeller, files_test, 'test', base_dir)

    dl_train = DataLoader(train_ds, batch_size = 32, shuffle=True, num_workers=4)

    dl_test = DataLoader(test_ds, batch_size = 32, shuffle=True, num_workers=4)

    return dl_train, dl_test

# a, b = get_dataloaders()  

a, b = get_dataloaders()  

import  matplotlib.pyplot as plt

import torchvision

import numpy as np

samples, labels = iter(b).next()

plt.figure(figsize=(16,24))

grid_imgs = torchvision.utils.make_grid(samples[:24])

np_grid_imgs = grid_imgs.numpy()

# in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.

plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
import torch.nn as nn

def get_model():

    device = 'cuda'

    model = torchvision.models.densenet121(pretrained=True)

#     for param in model.parameters():

#         param.requires_grad = False

        

    num_ftrs = model.classifier.in_features

    model.classifier = nn.Sequential(

        nn.Linear(num_ftrs, 256),

        nn.ReLU(),

#         nn.Dropout(0.4),

        nn.Linear(256, 2)

    )

    model = model.to(device)

    

    return model

# get_model()
import torch

def train(model, epochs, dataloader, test_dl):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

    loss_list = []

    acc_list = []

    val_loss_list = []

    acc_val_list = []

    model.train()

    device = 'cuda'

    for epoch in (range(epochs)):

        total_loss = 0

        num_batch = 0

        total_acc = 0

        print(f'STARTED EPOCH {epoch}')

        for samples, labels in tqdm(dataloader):

            samples, labels = samples.to(device), labels.to(device)

            labels = labels.long()

            optimizer.zero_grad()

            output = model(samples)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

#             scheduler.step()

            op = nn.Softmax(dim=1)(output)

            pred = torch.argmax(op, dim=1)

            correct = pred.eq(labels)

            acc = torch.mean(correct.float())

            total_acc += acc.item()

            num_batch += 1

            



        print(f'Loss is {total_loss / num_batch}')

        print(f'Accuracy is {total_acc / num_batch}')

        loss_val, acc_val = get_validation_stats(model, test_dl)

        acc_list.append(total_acc / num_batch)

        loss_list.append(total_loss / num_batch)

        val_loss_list.append(loss_val)

        acc_val_list.append(acc_val)

        if epoch%10 == 0:

            torch.save(model, f'model-v3-epoch{epoch}.pt')

        

    fig, axs = plt.subplots(4)

    fig.suptitle('Training Stats')

    axs[0].plot(loss_list)

    axs[1].plot(acc_list)

    axs[2].plot(val_loss_list)

    axs[3].plot(acc_val_list)

    

    

def get_validation_stats(model, test_dl):

    with torch.no_grad():

        model.eval()

        total_loss = 0

        device = 'cuda'

        criterion = nn.CrossEntropyLoss()

        total_loss = 0

        num_batch = 0

        total_acc = 0

        for samples, labels in tqdm(test_dl):

            samples, labels = samples.to(device), labels.to(device)

            labels = labels.long()

            output = model(samples)

            output = nn.Softmax(dim=1)(output)

            loss = criterion(output, labels)

            total_loss += loss.item()

#             scheduler.step()

            pred = torch.argmax(output, dim=1)

            correct = pred.eq(labels)

            acc = torch.mean(correct.float())

            total_acc += acc.item()

            num_batch += 1



        print(f'Val Loss is {total_loss / num_batch}')

        print(f'Val Accuracy is {total_acc / num_batch}')

        return (total_loss / num_batch), (total_acc / num_batch)



    

model = get_model()

train_dl, test_dl = get_dataloaders()  



train(model, 120, train_dl, test_dl)

get_validation_stats(model, test_dl)

torch.save(model, 'modelv3.pt')

    

    
# def plot(losses, accs, val_losses, val_accs):

#     plt.figure(figsize=(16, 9))

#     plt.plot(history.epoch, history.history['acc'])

#     plt.title('Model Accuracy')

#     plt.legend(['train'], loc='upper left')

#     plt.show()



#     plt.figure(figsize=(16, 9))

#     plt.plot(history.epoch, history.history['loss'])

#     plt.title('Model Loss')

#     plt.legend(['train'], loc='upper left')

#     plt.show()



#     plt.figure(figsize=(16, 9))

#     plt.plot(history.epoch, history.history['val_acc'])

#     plt.title('Model Validation Accuracy')

#     plt.legend(['train'], loc='upper left')

#     plt.show()



#     plt.figure(figsize=(16, 9))

#     plt.plot(history.epoch, history.history['val_loss'])

#     plt.title('Model Validation Loss')

#     plt.legend(['train'], loc='upper left')

#     plt.show()
samples, labels = iter(test_dl).next()

device = 'cuda'

samples = samples.to(device)

fig = plt.figure(figsize=(24, 16))

fig.tight_layout()

model.eval()

output = nn.Softmax(dim=1)(model(samples[:24]))

pred = torch.argmax(output, dim=1)

pred = [p.item() for p in pred]

real = [p.item() for p in labels]

ad = {0:'No', 1:'Yes'}

for num, sample in enumerate(samples[:24]):

    plt.subplot(4,6,num+1)

    plt.title(f'{ad[pred[num]]}*{ad[real[num]]}')

    plt.axis('off')

    sample = sample.cpu().numpy()

    plt.imshow(np.transpose(sample, (1,2,0)))
# def get_validation_stats(model, test_dl):

#     with torch.no_grad():

#         model.eval()

#         total_loss = 0

#         device = 'cuda'

#         criterion = nn.CrossEntropyLoss()

#         for samples, labels in tqdm(test_dl):

#             samples, labels = samples.to(device), labels.to(device)

#             labels = labels.long()

#             output = model(samples)

#             loss = criterion(output, labels)

#             total_loss += loss.item()

#         print(f'Loss is {total_loss / 32}')

#         pred = torch.argmax(output, dim=1)

#         correct = pred.eq(labels)

#         pos = torch.mean(1*(pred == 1))

#         acc = torch.mean(correct.float())

#         print(f'Accuracy is {acc}')

#         print(f'Fraction of positiive cases {pos}')



# get_validation_stats(model, test_dl)