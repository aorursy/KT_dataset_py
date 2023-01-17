# Classifying Digits using Simple ML models

# Necessary Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score



# Exploring files in the Input Directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Step 1: Convert the data into usable format

# pandas is used to read the contents of csv into a dataframe

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



# Retrieve the features and label columns into a separate numpy arrays

features = train[train.columns[1:]].values

label = train.label.values



print(type(features))

print(type(label))



print(features.shape)

print(label.shape)

import warnings

warnings.simplefilter("ignore")

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)

acc = 0.0

global best_model



for jj, (train_index, val_index) in enumerate(kf.split(features)):

    print("Fitting fold", jj+1)

    train_features = features[train_index]

    train_target = label[train_index]

    

    val_features = features[val_index]

    val_target = label[val_index]

    

    model = LogisticRegression(C=20, solver='lbfgs', multi_class='multinomial')

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)

    fold_acc=accuracy_score(val_target, np.argmax(val_pred, axis=1))

    print("Fold accuracy:", accuracy_score(val_target, np.argmax(val_pred, axis=1)))

    #test_preds += model.predict_proba(test)/n_splits

    if(fold_acc>acc):

        acc = fold_acc

        best_model = model

    del train_features, train_target, val_features, val_target

    gc.collect()



    

    
print(acc)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(test.columns)

#Retrieve the features and label columns into a separate numpy arrays

test_features = test[test.columns[0:]].values

test_pred = model.predict_proba(test_features)

predict = np.argmax(test_pred, axis=1)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = predict

submission.to_csv('submission.csv', index=False)
from sklearn.linear_model import SGDClassifier

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)

acc = 0.0

global best_model



for jj, (train_index, val_index) in enumerate(kf.split(features)):

    print("Fitting fold", jj+1)

    train_features = features[train_index]

    train_target = label[train_index]

    

    val_features = features[val_index]

    val_target = label[val_index]

    

    # loss = 'hinge' represents linear regression

    # log loss implement logistic regression

    model = SGDClassifier(loss='log')

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)

    fold_acc=accuracy_score(val_target, np.argmax(val_pred, axis=1))

    print("Fold accuracy:", accuracy_score(val_target, np.argmax(val_pred, axis=1)))

    #test_preds += model.predict_proba(test)/n_splits

    if(fold_acc>acc):

        acc = fold_acc

        best_model = model

    del train_features, train_target, val_features, val_target

    gc.collect()

from sklearn.ensemble import RandomForestClassifier

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)

acc = 0.0

global best_model



for jj, (train_index, val_index) in enumerate(kf.split(features)):

    print("Fitting fold", jj+1)

    train_features = features[train_index]

    train_target = label[train_index]

    

    val_features = features[val_index]

    val_target = label[val_index]

    

    # max depth serves as an important hyperparameter

    # When the depth was set with a value of 2, accuracy was down by 60%

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)

    fold_acc=accuracy_score(val_target, np.argmax(val_pred, axis=1))

    print("Fold accuracy:", accuracy_score(val_target, np.argmax(val_pred, axis=1)))

    #test_preds += model.predict_proba(test)/n_splits

    if(fold_acc>acc):

        acc = fold_acc

        best_model = model

    del train_features, train_target, val_features, val_target

    gc.collect()

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(test.columns)

#Retrieve the features and label columns into a separate numpy arrays

test_features = test[test.columns[0:]].values

test_pred = model.predict_proba(test_features)

predict = np.argmax(test_pred, axis=1)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = predict

submission.to_csv('submission.csv', index=False)
import torch

import torch.nn.functional as F

from torch import nn

class Flatten(nn.Module):

    def __init__(self):

        super(Flatten, self).__init__()



    def forward(self, x):

        return x.view(x.size(0), -1)



class NN(torch.nn.Module):

    def __init__(self):

        super(NN,self).__init__()

        self.conv1 = torch.nn.Conv2d(1,6,3,padding=1)

        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(6,16,5,padding=0)

        self.pool2 = torch.nn.MaxPool2d(2)

        self.linear1 = torch.nn.Linear(400,120)

        self.linear2 = torch.nn.Linear(120,84)

        self.linear3 = torch.nn.Linear(84,10)

    def forward(self,x):

        c1= F.relu(self.conv1(x))

        s1 = self.pool1(c1)

        c2 = F.relu(self.conv2(s1))

        s2 = self.pool2(c2)

        f  = (Flatten()(s2))

        f1 = F.relu(self.linear1(f))

        f2 = F.relu(self.linear2(f1))

        f3 = self.linear3(f2)

        return f3
# Install Necessary Packages

!pip install torchsummary
import torch

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN().to(device)

summary(model,(1,28,28))
import pandas as pd

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



# Retrieve the features and label columns into a separate numpy arrays

features = train[train.columns[1:]].values

label = train.label.values



[rows, columns] = features.shape

print(type(features))

print(type(label))



print(features.shape)

print(label.shape)
features = features /255.0

features = features.reshape(rows,1,28,28)

print(features.shape)
from torch.utils.data import Dataset, DataLoader

class mnistDataset(Dataset):

    def __init__(self, images, labels):

        self.image =  torch.from_numpy(images)

        self.gt = torch.from_numpy(labels)



    def __len__(self):

        #print(self.image.shape)

        #print(self.gt.shape)

        return self.image.shape[0]



    def __getitem__(self,index):

        return self.image[index], self.gt[index]
dataset = mnistDataset(features, label)

print(features.shape)

print(label.shape)

trainLoader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)

import torch.optim

Criterion = torch.nn.CrossEntropyLoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
def train(model, Criterion, optimizer, trainLoader):

    model.train()

    epoch_list=[]

    loss_list=[]

    for epoch in range(100):

        running_loss = 0

        for data,target in trainLoader:

            x = data.to(device)

            x = x.type(torch.cuda.FloatTensor)

            y = target.to(device)

            #Compute model ouput

            pred = model(x)

            #print(pred)

            #print(y)

            #Compute loss

            loss = Criterion(pred,y)

            running_loss +=loss.item()

            #Optimizer to adjust weights

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('epoch:',epoch,'loss:',running_loss/len(trainLoader))

        epoch_list.append(epoch)

        loss_list.append(running_loss/len(trainLoader))

    return epoch_list,loss_list, model

epoch_list, loss_list, model = train(model, Criterion, optimizer, trainLoader)



from torch.utils.data import Dataset, DataLoader

class mnisttestDataset(Dataset):

    def __init__(self, images):

        self.image =  torch.from_numpy(images)



    def __len__(self):

        #print(self.image.shape)

        #print(self.gt.shape)

        return self.image.shape[0]



    def __getitem__(self,index):

        return self.image[index]
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#Retrieve the features and label columns into a separate numpy arrays

features = test[test.columns[0:]].values

print(features.shape)

features = features /255.0

features = features.reshape(28000,1,28,28)



dataset = mnisttestDataset(features)

print(features.shape)

print(label.shape)

testLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

import numpy as np

from torch.autograd import Variable

def validation(model, optimizer, testLoader, device, Criterion):

    model.eval()

    predictions = []

    with torch.no_grad():

        for vinput in testLoader:

            vinput = Variable(vinput)

            vinput = vinput.to(device)

            vinput = vinput.type(torch.cuda.FloatTensor)

            optimizer.zero_grad()

            vpredict = model(vinput)

            predict = F.softmax(vpredict, dim=1)

            predict = predict.to('cpu')

            predict = predict.numpy()

            #print(predict)

            predictions.append(np.argmax(predict, axis=1))

            #print(predictions)

    return predictions
predict = validation(model, optimizer, testLoader, device, Criterion)

predict_labels = []



for i in range(28000):

    predict_labels.append(predict[i][0])







submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = predict_labels

submission.to_csv('submission.csv', index=False)
import torchvision.models as models

import torch.nn as nn

import torch

#Instantiating ResNet model with pretrained weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained = True)

num_features = model.fc.in_features

#Modifying the fully connected layer to reduce number of classes from 1000 to 2

model.fc = nn.Linear(num_features, 10)

model = model.to(device)
import pandas as pd

import numpy as np

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



# Retrieve the features and label columns into a separate numpy arrays

features = train[train.columns[1:]].values

label = train.label.values



[rows, columns] = features.shape

features = np.tile(features[:,:,None],[1,1,3])

print(type(features))

print(type(label))



print(features.shape)

print(label.shape)
features = features /255.0

features = features.transpose((0,2,1))

features = features.reshape(rows,3,28,28)

print(features.shape)
from torch.utils.data import Dataset, DataLoader

class mnistDataset(Dataset):

    def __init__(self, images, labels):

        self.image =  torch.from_numpy(images)

        self.gt = torch.from_numpy(labels)



    def __len__(self):

        #print(self.image.shape)

        #print(self.gt.shape)

        return self.image.shape[0]



    def __getitem__(self,index):

        return self.image[index], self.gt[index]
dataset = mnistDataset(features, label)

print(features.shape)

print(label.shape)

trainLoader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)
params_to_update = model.parameters()

optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

Criterion = nn.CrossEntropyLoss(reduction='mean')
def train(model, Criterion, optimizer, trainLoader):

    model.train()

    epoch_list=[]

    loss_list=[]

    for epoch in range(10):

        running_loss = 0

        for data,target in trainLoader:

            x = data.to(device)

            x = x.type(torch.cuda.FloatTensor)

            y = target.to(device)

            #Compute model ouput

            pred = model(x)

            #print(pred)

            #print(y)

            #Compute loss

            loss = Criterion(pred,y)

            running_loss +=loss.item()

            #Optimizer to adjust weights

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('epoch:',epoch,'loss:',running_loss/len(trainLoader))

        epoch_list.append(epoch)

        loss_list.append(running_loss/len(trainLoader))

    return epoch_list,loss_list, model
epoch_list, loss_list, model = train(model, Criterion, optimizer, trainLoader)
from torch.utils.data import Dataset, DataLoader

class mnisttestDataset(Dataset):

    def __init__(self, images):

        self.image =  torch.from_numpy(images)



    def __len__(self):

        #print(self.image.shape)

        #print(self.gt.shape)

        return self.image.shape[0]



    def __getitem__(self,index):

        return self.image[index]
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#Retrieve the features and label columns into a separate numpy arrays

features = test[test.columns[0:]].values

print(features.shape)

features = features /255.0

features = np.tile(features[:,:,None],[1,1,3])

features = features.reshape(28000,3,28,28)

dataset = mnisttestDataset(features)

print(features.shape)

print(label.shape)

testLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

import numpy as np

from torch.autograd import Variable

import torch.nn.functional as F

def validation(model, optimizer, testLoader, device, Criterion):

    model.eval()

    predictions = []

    with torch.no_grad():

        for vinput in testLoader:

            vinput = Variable(vinput)

            vinput = vinput.to(device)

            vinput = vinput.type(torch.cuda.FloatTensor)

            optimizer.zero_grad()

            vpredict = model(vinput)

            predict = F.softmax(vpredict, dim=1)

            predict = predict.to('cpu')

            predict = predict.numpy()

            #print(predict)

            predictions.append(np.argmax(predict, axis=1))

            #print(predictions)

    return predictions
predict = validation(model, optimizer, testLoader, device, Criterion)

predict_labels = []



for i in range(28000):

    predict_labels.append(predict[i][0])



submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission['Label'] = predict_labels

submission.to_csv('submission.csv', index=False)