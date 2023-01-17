# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns" , None)

import torch

import torch.nn as nn

import torch.nn.functional as F

import os

from collections import defaultdict

from sklearn.model_selection import train_test_split

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

torch.manual_seed(0)
BASE_PATH = "/kaggle/input/digit-recognizer/"

# BASE_PATH = "."

train = pd.read_csv(os.path.join(BASE_PATH , "train.csv") , low_memory=False)

test = pd.read_csv(os.path.join(BASE_PATH , "test.csv") , low_memory=False)

sub = pd.read_csv(os.path.join(BASE_PATH , "sample_submission.csv") , low_memory=False)
display(train.head())

display(test.tail())
TARGET = "label"
train , valid = train_test_split(train , test_size=0.2 , random_state = 0 ,stratify=train[TARGET])
train[TARGET].value_counts(normalize=True).sort_index()
valid[TARGET].value_counts(normalize=True).sort_index()
train.reset_index(drop=True , inplace=True)

valid.reset_index(drop=True , inplace=True)
train.shape , valid.shape
X_train , y_train = train.loc[: , "pixel0":"pixel783"] , train[TARGET]

X_valid , y_valid = valid.loc[: , "pixel0":"pixel783"] , valid[TARGET]
X_train = X_train / 255

X_valid = X_valid / 255

test = test / 255
X_train = X_train.values.reshape([-1,28*28*1])#28 by 28 picels with 1 color channel

X_valid = X_valid.values.reshape([-1,28*28*1])

X_test = test.values.reshape([-1,28*28*1])
X_train.shape , X_valid.shape , X_test.shape
def plot_sample_image(n):

    image = X_train[n].reshape([28,28])

    plt.imshow(image , cmap="Greys")

    plt.title(y_train[n] , fontsize=17)

    

plot_sample_image(n=930)
class MNISTDataset:

    def __init__(self,features,targets):

        self.features = features

        self.targets = targets

    

    def __len__(self):

        return len(self.features)

    

    def __getitem__(self , item):

        x = self.features[item] #For Features

        y = self.targets[item] #For Targets

        return x , y

    

class TestDataset:

    def __init__(self,features):

        self.features = features

        

    def __len__(self):

        return len(self.features)

    

    def __getitem__(self,item):

        x = self.features[item]#For Features

        return x , item
train_dataset = MNISTDataset(X_train , y_train)

valid_dataset = MNISTDataset(X_valid , y_valid)

test_dataset = TestDataset(X_test)
def create_dataloader(dataset,batch_size,num_workers):

    dataloader = torch.utils.data.DataLoader(

        dataset,

        batch_size=batch_size,

        num_workers=num_workers

    )

    return dataloader
train_loader = create_dataloader(train_dataset,batch_size=200,num_workers=4)

valid_loader = create_dataloader(valid_dataset,batch_size=200,num_workers=4)

test_loader = create_dataloader(test_dataset,batch_size=200,num_workers=4)
x , y = next(iter(train_loader))
# X_train[0].reshape([-1,28*28]).shape
in_features = X_train.shape[1]

num_targets = len(y_train.unique())
class Model(nn.Module):

    def __init__(self,in_features,num_targets):

        super(Model , self).__init__()

        

        #Linear Function 1: 784 --> 200

        self.fc1 = nn.Linear(in_features , 200)

        #Non-Linearity 1

        self.relu1 = nn.ReLU()

        

        #Linear Function 2: 200 --> 250

        self.fc2 = nn.Linear(200 , 250)

        #Non-Linearuty 2

        self.tanh2 = nn.Tanh()

        

        #Linear Function 3: 250 --> 200

        self.fc3 = nn.Linear(250,200)

        #Non-Linearity 3

        self.elu3 = nn.ELU()

        

        #Output Layer 200 --> 10

        self.output = nn.Linear(200 , num_targets)

        

    def forward(self,x):

        x = self.fc1(x)

        x = self.relu1(x)

        

        x = self.fc2(x)

        x = self.tanh2(x)

        

        x = self.fc3(x)

        x = self.elu3(x)

        

        x = self.output(x)

        return x
model = Model(in_features,num_targets)

model = model.to(device)

lr = 0.02

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

criterion = nn.CrossEntropyLoss().to(device)
def train_epoch(model,dataloader,loss_fn,optimizer):

    global inputs

    model.train()

    losses = []

    accuracy = 0

    model = model.float()

    

    for data in dataloader:

        optimizer.zero_grad()

        

        inputs , targets = data

        

        inputs = inputs.to(device)

        targets = targets.to(device)

        

        inputs = inputs.view(-1 ,28*28)

        

        outputs = model(inputs.float())

        

        loss = loss_fn(outputs , targets)

        losses.append(loss.item())

        _ , preds = torch.max(outputs , dim=1)

        acc = torch.sum(preds == targets)

        accuracy += acc.detach().cpu().numpy()

        

        loss.backward()

        optimizer.step()

        

    return np.mean(losses) , accuracy / len(train)

        

        

def eval_model(model,dataloader,loss_fn,optimizer):

    model.eval()

    losses = []

    accuracy = 0

    model = model.float()

    

    for data in dataloader:

        optimizer.zero_grad()

        inputs , targets = data

        inputs = inputs.to(device)

        targets = targets.to(device)

        inputs = inputs.view(-1,28*28)

        outputs = model(inputs.float())

        loss = loss_fn(outputs, targets)

        losses.append(loss.item())

        _ , preds = torch.max(outputs , dim=1)

        acc = torch.sum(preds == targets)

        accuracy += acc.detach().cpu().numpy()

        

        loss.backward()

        optimizer.step()

        

        

    return np.mean(losses) , accuracy / len(valid)
def Run_Test(model , dataloader):

    model.eval()

    predictions = []

    indeces = []

    model = model.float()

    

    with torch.no_grad():

        for data in dataloader:

            inputs , index = data

            inputs = inputs.to(device)

            index = index.to(device)

            inputs = inputs.view(-1,28*28)

            outputs = model(inputs.float())

            _ , preds = torch.max(outputs , dim=1)

            preds = preds.detach().cpu().numpy()

            predictions.extend(preds)

            indeces.extend(index.detach().cpu().numpy())

            

    return predictions , indeces
EPOCHS = 20
history = defaultdict(list)

for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}/{EPOCHS}")

    train_loss , train_acc = train_epoch(model , train_loader, criterion , optimizer)

    val_loss , val_acc = eval_model(model , valid_loader, criterion , optimizer)

    

    history["train_loss"].append(train_loss)

    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)

    history["val_acc"].append(val_acc)

    

    print(f"Train Loss {train_loss} Accuracy {train_acc}")

    print(f"Val Loss {val_loss} Accuracy {val_acc}")

    print()
plt.plot(history["train_loss"] , color="blue" , label="Train_Loss")

plt.plot(history["val_loss"] ,color="red" ,label="Val_Loss")

plt.xlabel("Epochs" , fontsize=17)

plt.ylabel("Loss" , fontsize=17)

plt.legend()

plt.show()
plt.plot(history["train_acc"] , color="blue" , label="Train_Accuracy")

plt.plot(history["val_acc"] ,color="red" ,label="Val_Accuracy")

plt.xlabel("Epochs" , fontsize=17)

plt.ylabel("Accuracy" , fontsize=17)

plt.legend()

plt.show()
predictions , indeces = Run_Test(model ,  test_loader)
sub["Label"] = predictions

sub["indeces"] = indeces
sub.set_index("indeces" , inplace=True)

sub.index.name = None
sub.to_csv("submission.csv" , index=False)