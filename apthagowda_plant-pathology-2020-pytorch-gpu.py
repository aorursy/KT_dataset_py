!pip install knockknock #For sending message to telegram

!pip install efficientnet-pytorch
import os

import cv2

import math

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,precision_score,recall_score,ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold

from transformers import get_cosine_schedule_with_warmup

from albumentations import *

from albumentations.pytorch import ToTensorV2



import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from torch.optim import lr_scheduler

from efficientnet_pytorch import EfficientNet



from knockknock import telegram_sender 

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

token = user_secrets.get_secret("token")

chat_id = user_secrets.get_secret("chat_id")



import warnings  

warnings.filterwarnings('ignore')
SEED = 42

BATCH_SIZE = 16

SIZE = [420,420]

LR = 0.0008

WEIGHT_DECAY = 0

EPOCHS = 40

WARMUP = 15

STEP_SIZE = 5 

TTA = 4
def seed_everything(SEED):

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    

seed_everything(SEED)   

device = torch.device("cuda:0")
DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'

train_df = pd.read_csv(DIR_INPUT + '/train.csv')

test_df = pd.read_csv(DIR_INPUT + '/test.csv')

cols = list(train_df.columns[1:])
transform = {

    'train' : Compose([

        Resize(SIZE[0],SIZE[1],always_apply=True),

        HorizontalFlip(p=0.5),

        VerticalFlip(p=0.5),

        RandomRotate90(p=0.5),

        Normalize(p=1.0),

        ToTensorV2(p=1.0)

    ]),

    'valid': Compose([

        Resize(SIZE[0],SIZE[1],always_apply=True),

        Normalize(p=1.0),

        ToTensorV2(p=1.0)

    ]),

    'test_tta': Compose([

        Resize(SIZE[0],SIZE[1],always_apply=True),

        HorizontalFlip(p=0.5),

        VerticalFlip(p=0.5),

        RandomRotate90(p=0.5),

        Normalize(p=1.0),

        ToTensorV2(p=1.0)

    ])

}
class PLANT(Dataset):

    

    def __init__(self,df,transform=None,train=True):

        self.df = df

        self.transform = transform

        self.train = train

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        path = self.df.iloc[idx]['image_id']

        image = cv2.imread(DIR_INPUT+f"/images/{path}.jpg")

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.transform is not None:

            image = self.transform(image=image)['image']

        if self.train==True:

            label = np.argmax(self.df[cols].iloc[idx].values).reshape(1,1)

            return {'image': image,'label': label }

        if self.train==False:

            return {'image':image }
train,valid = train_test_split(train_df,test_size = 0.2,random_state = SEED)



dataset_train = PLANT(df=train, transform=transform['train'])

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)



dataset_valid = PLANT(df=valid, transform=transform['valid'])

dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=True)



dataset_test = PLANT(test_df,transform=transform['valid'],train=False)

dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)



dataset_test_tta = PLANT(test_df,transform=transform['test_tta'],train=False)

dataloader_test_tta = DataLoader(dataset_test_tta, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
def plot_transform(image_id,num_images=7):

    plt.figure(figsize=(30,10))

    plt.tight_layout()

    for i in range(1,num_images+1):

        plt.subplot(1,num_images+1,i)

        plt.axis('off')

        x = dataset_train.__getitem__(image_id)

        image = x['image'].numpy()

        image = np.transpose(image,[1,2,0])

        plt.imshow(image)

        

plot_transform(9)
def getmodel():

    model = EfficientNet.from_pretrained('efficientnet-b4')

    model._fc = nn.Sequential(

     nn.Linear(in_features=1792, out_features=4, bias=True))

    model = model.to(device)

    return model
new_model = getmodel()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(new_model.parameters(), lr=LR,weight_decay=WEIGHT_DECAY)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP, num_training_steps=EPOCHS)
results = pd.DataFrame(columns=['training_loss','training_accuracy','validation_loss','validation_accuracy','precision','recall','roc_auc_score'])

@telegram_sender(token=token, chat_id=int(chat_id))

def train(model, criterion, optimizer,scheduler, dataloader_train, dataloader_valid):

    global results

    for epoch in range(EPOCHS):

        print('Epoch {}/{}'.format(epoch,EPOCHS-1))

        since = time.time()

        model.train()

        training_accuracy  = []

        training_loss = []

        for bi, d in enumerate(tqdm(dataloader_train, total=int(len(dataloader_train)))):

            inputs = d["image"]

            labels = d["label"]

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs)

                labels = labels.squeeze()

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                outputs = torch.max(outputs,1)[1]

                outputs = outputs.cpu().detach().numpy()

                labels = labels.cpu().numpy()

                training_accuracy.append(accuracy_score(outputs,labels))

                training_loss.append(loss.item())

        print('Training accuracy: {:.4f} and Training Loss: {:.4f}'.format(np.mean(training_accuracy),np.mean(training_loss)))



        

                                     

        model.eval()

        validation_loss = []

        validation_labels = []

        validation_outputs = []

        with torch.no_grad():

            for bi,d in enumerate(tqdm(dataloader_valid,total=int(len(dataloader_valid)))):

                inputs = d["image"]

                labels = d["label"]

                inputs = inputs.to(device, dtype=torch.float)

                labels = labels.to(device, dtype=torch.long)

                outputs = model(inputs)

                labels = labels.squeeze()

                loss = criterion(outputs,labels)

                outputs_softmax = F.softmax(outputs).cpu().detach().numpy()

                labels_onehot = torch.eye(4)[labels].cpu().numpy()

                validation_labels.extend(labels_onehot)

                validation_outputs.extend(outputs_softmax)

                validation_loss.append(loss.item())

        precision = precision_score(np.argmax(validation_labels,axis=1),np.argmax(validation_outputs,axis=1),average='macro')

        recall = recall_score(np.argmax(validation_labels,axis=1),np.argmax(validation_outputs,axis=1),average='macro')

        accuracy = accuracy_score(np.argmax(validation_labels,axis=1),np.argmax(validation_outputs,axis=1))

        roc = roc_auc_score(validation_labels,validation_outputs,average='macro')

        print('Validation accuracy: {:.4f} and Validation Loss: {:.4f} and roc_auc_score: {:.4f}'.format(accuracy,\

                            np.mean(validation_loss),roc))

        res = pd.DataFrame([[np.mean(training_loss),np.mean(training_accuracy),np.mean(validation_loss),\

                             accuracy,precision,recall,roc]],columns=results.columns)

        results = pd.concat([results,res])

        scheduler.step()

    return results.iloc[-1]
train(new_model, criterion, optimizer,scheduler, dataloader_train, dataloader_valid)
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(15,15), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
results.reset_index(drop=True,inplace=True)

display_training_curves(results['training_loss'], results['validation_loss'], 'loss', 311)

display_training_curves(results['training_accuracy'], results['validation_accuracy'], 'accuracy', 312)

display_training_curves(1, results['roc_auc_score'], 'roc_auc_score', 313)
display_training_curves(1, results['precision'], 'precision', 211)

display_training_curves(1, results['recall'], 'recall', 212)
new_model.eval()

test_pred = np.zeros((len(test_df),4))

with torch.no_grad():

    for i, data in enumerate(tqdm(dataloader_test,total=int(len(dataloader_test)))):

        inputs = data['image']

        inputs = inputs.to(device, dtype=torch.float)

        predict = new_model(inputs)

        test_pred[i*len(predict):(i+1)*len(predict)] = predict.detach().cpu().squeeze().numpy()
submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')

submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = F.softmax(torch.from_numpy(test_pred),dim=1)

submission_df.to_csv('submission.csv', index=False)

pd.Series(np.argmax(submission_df[cols].values,axis=1)).value_counts()
new_model.eval()

test_pred = np.zeros((len(test_df),4))

for i in range(TTA):

    with torch.no_grad():

        for i, data in enumerate(tqdm(dataloader_test_tta,total=int(len(dataloader_test_tta)))):

            inputs = data['image']

            inputs = inputs.to(device, dtype=torch.float)

            predict = new_model(inputs)

            test_pred[i*len(predict):(i+1)*len(predict)] += predict.detach().cpu().squeeze().numpy()
submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')

submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = F.softmax(torch.from_numpy(test_pred/TTA),dim=1)

submission_df.to_csv('submission_tta.csv', index=False)

pd.Series(np.argmax(submission_df[cols].values,axis=1)).value_counts()