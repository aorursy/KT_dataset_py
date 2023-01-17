import os

import sys

import glob

import tqdm

import cv2

import pydicom 

import numpy as np

import pandas as pd

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')



train_dir = '../input/osic-pulmonary-fibrosis-progression/train'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test'
#Checking dimensions of all datasets



train.shape, test.shape, sub.shape
#Exploring Train data

train.head()
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,PowerTransformer

from sklearn.model_selection import train_test_split, cross_val_score,cross_validate, KFold,GroupKFold

from sklearn.metrics import make_scorer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
# getting base week for patient

# here min_week is the minimum number of week present for a patient

# baseline_week is the number of weeks counting from the min week for a patient

def get_baseline_week(data):

    df = data.copy()

    df['Weeks'] = df['Weeks'].astype(int)

    df['min_week'] = df.groupby('Patient')['Weeks'].transform('min')

    df['baseline_week'] = df['Weeks'] - df['min_week']

    return df
df_1 = get_baseline_week(train)

df_1[df_1['Patient']=='ID00026637202179561894768']
#getting FVC for base week and setting it as base_FVC for patient

def get_base_FVC(data):

    df = data.copy()

    base = df.loc[df['Weeks']==df['min_week']][['Patient','FVC']].copy()

    base.columns =['Patient','base_FVC']

    

    base['nb'] = 1

    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

    

    base = base[base['nb']==1]

    base.drop('nb',axis=1,inplace=True)

    df = df.merge(base,on='Patient',how='left')

    df.drop(['min_week'],axis=1)

    return df

    
train.drop_duplicates(keep=False,inplace=True,subset=['Patient','Weeks'])

train_data = get_baseline_week(train)

train_data = get_base_FVC(train_data)

train_data
#Exploring the content of the submission file

sub.head()
sub.shape
#Exploring the content of test file

test.head()
# Processing submission file

# Dropping FVC column and merging with Test data

sub.drop('FVC',axis=1,inplace=True)

sub[['Patient','Weeks']] = sub['Patient_Week'].str.split("_",expand=True)

sub = sub.merge(test.drop('Weeks',axis=1), on ='Patient', how='left')

sub.head()
sub['min_weeks'] = np.nan

sub = get_baseline_week(sub)

sub = get_base_FVC(sub)

sub.head()
# Split the train columns and train label

train_columns = ['baseline_week','base_FVC','Percent','Age','Sex','SmokingStatus']

train_label = ['FVC']



# Columns in submission file

sub_columns = ['Patient_Week','FVC','Confidence']



train = train_data[train_columns]

test = sub[train_columns]
train.head()
# applying standard scaling on numeric columns 0,1,2,3

# applying One-hot encoding on categorical columns 4,5 

# each transformation will be defined in a tuple

transformer = ColumnTransformer([('s',StandardScaler(),[0,1,2,3]),('o',OneHotEncoder(),[4,5])])



target = train_data[train_label].values

train = transformer.fit_transform(train)

test = transformer.transform(test)
# Transformed train data

train[0]
# Importing Pytorch libraries

import torch 

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

# Define class for our torch model

# This class will be inherited from nn.Module

class Model(nn.Module):

    

    def __init__(self,n):

        super(Model,self).__init__()

        

        #Define the metwork layers

        self.layer1 = nn.Linear(n,200)  # n inputs nodes, 200 output nodes

        self.layer2 = nn.Linear(200,100) # 200 input nodes, 100 output nodes

        

        self.out1 = nn.Linear(100,3) # 100 input nodes, 3 output nodes

        self.relu3 = nn.ReLU()

        self.out2 = nn.Linear(100,3) # 100 input nodes, 3 output nodes

        

    

    def forward(self,x):

        x1 = F.leaky_relu(self.layer1(x))

        x1 = F.leaky_relu(self.layer2(x1))

        

        o1 = self.out1(x1)

        o2 = F.relu(self.out2(x1))

        

        return o1+ torch.cumsum(o2,dim=1)
def run():

    

    # function to calculate metircs score

    def score(outputs,target):

        confidence = outputs[:,2] - outputs[:,0]

        clip = torch.clamp(confidence,min=70) # Condidence is clipped at min value of 70

        target = torch.reshape(target, outputs[:,1].shape)

        delta = torch.abs(outputs[:,1] - target)

        delta = torch.clamp(delta,max=1000) # delta is clipped at max value of 1000

        

        # calculating the metrics as provided in the challenge

        sqrt_2 = torch.sqrt(torch.tensor([2.])).to(device)

        metrics = (delta*sqrt_2/clip) + torch.log(clip*sqrt_2)

        return torch.mean(metrics)

        

    

    def qloss(outputs,target):

        qs = [0.25,0.5,0.75]

        qs = torch.tensor(qs,dtype=torch.float).to(device)

        e =  target - outputs

        e.to(device)

        v = torch.max(qs*e,(qs-1)*e)

        v = torch.sum(v,dim=1)

        return torch.mean(v)

    

    def loss_fn(outputs,target,l):

        return l*qloss(outputs,target) + (1-l) * score(outputs,target)

    

    def train_loop(train_loader,model,loss_fn, device,optimizer,lr_scheduler=None):

        model.train()

        losses = []

        metrics = []

        

        for i, (inputs,labels) in enumerate(train_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs)

                metric = score(outputs,labels)

                

                loss = loss_fn(outputs,labels,0.8)

                metrics.append(metric.cpu().detach().numpy())

                losses.append(loss.cpu().detach().numpy())

                

                loss.backward()

                optimizer.step()

                

                if lr_scheduler !=None:

                    lr_scheduler.step()

                    

        return losses,metrics

    

    def valid_loop(valid_loader,model,loss_fn, device):

        model.eval()

        losses = []

        metrics = []

        

        for i, (inputs, labels) in enumerate(valid_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            outputs = model(inputs)                 

            metric = score(outputs,labels)

            

            loss = loss_fn(outputs,labels,0.8)

            metrics.append(metric.cpu().detach().numpy())

            losses.append(loss.cpu().detach().numpy())

            

        return losses,metrics

    

    

    NFOLDS = 5

    kfold = KFold(NFOLDS,shuffle=True,random_state=42)

    

    #generate kfolds

    for k,(train_idx,valid_idx) in enumerate(kfold.split(train)):

        batch_size =64

        epochs = 50

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'{device} is used')

        

        #Split into train and validation

        X_train,X_valid,y_train,y_valid = train[train_idx,:], train[valid_idx,:], target[train_idx], target[valid_idx]

        n = X_train.shape[1] #number of inputs (records)

        model = Model(n)

        model.to(device)

        lr = 0.1

        optimizer = optim.Adam(model.parameters(),lr=lr)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        

        # create tensors for training

        train_tensor = torch.tensor(X_train, dtype = torch.float)

        y_train_tensor = torch.tensor(y_train,dtype=torch.float)

        

        train_ds = TensorDataset(train_tensor, y_train_tensor)

        train_dl = DataLoader(train_ds,batch_size=batch_size, num_workers=4, shuffle=True)

        

        # create tensors for validation

        valid_tensor = torch.tensor(X_valid,dtype=torch.float)

        y_valid_tensor = torch.tensor(y_valid,dtype=torch.float)

        

        valid_ds = TensorDataset(valid_tensor,y_valid_tensor)

        valid_dl = DataLoader(valid_ds,

                             batch_size = batch_size,

                             num_workers=4,

                             shuffle=False

                             )

        

        print(f"Fold {k}")

        

        for i in range(epochs):

            losses, metrics = train_loop(train_dl,model,loss_fn,device,optimizer,lr_scheduler)

            valid_losses,valid_metrics = valid_loop(valid_dl,model,loss_fn,device)

            

            if (i+1)%5==0:

                print(f"epoch:{i} Training | loss:{np.mean(losses)} score: {np.mean(metrics)}| \n Validation | loss:{np.mean(valid_losses)} score:{np.mean(valid_metrics)}|")

        

        # save the model for current fold

        torch.save(model.state_dict(),f'model{k}.bin')

run()
def inference():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nfold = 5

    all_prediction = np.zeros((test.shape[0],3))

    

    for i in range(nfold):

        n = train.shape[1]

        

        model = Model(n)

        model.load_state_dict(torch.load(f"model{i}.bin"))

        predictions = list()

        model.to(device)

        test_tensor = torch.tensor(test,dtype=torch.float)

        test_dl = DataLoader(test_tensor,

                        batch_size=64,

                        num_workers=2,

                        shuffle=False)

        

        with torch.no_grad():

            for i, inputs in enumerate(test_dl):

                inputs = inputs.to(device, dtype=torch.float)

                outputs= model(inputs) 

                predictions.extend(outputs.cpu().detach().numpy())

        

        all_prediction += np.array(predictions)/nfold

        

    return all_prediction

    
sub.head()
prediction = inference()

sub['Confidence'] = np.abs(prediction[:,2]-prediction[:,0])

sub['FVC'] = prediction[:,1]

submission = sub[sub_columns]
submission.head()
submission.shape
submission.to_csv('submission.csv',index=False)