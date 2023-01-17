import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import time

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

print(os.listdir("../input"))

print(torch.__version__)

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")

df_train.head()
df_train.shape
df_train.info()
#Use only a small subset of the Test set.

df_test = pd.read_csv("../input/test.csv", nrows=10000)

df_test.head()
df_test.shape
sns.countplot(x='event', data=df_train)
print(df_train['experiment'].unique())

print(df_test['experiment'].unique())
dic = {'CA': 0, 'DA': 1, 'SS': 2, 'LOFT': 3}

df_train["experiment"] = df_train["experiment"].apply(lambda x: dic[x])

df_test["experiment"] = df_test["experiment"].apply(lambda x: dic[x])
dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

df_train["event"] = df_train["event"].apply(lambda x: dic[x])
X = df_train.drop(columns=['event'], axis = 1)

Y = df_train['event']
X.shape
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
def to_categorical(y, num_classes):

    """1-hot encodes a tensor"""

    return np.eye(num_classes, dtype='uint8')[y]
y_train = to_categorical(y_train, 4)

y_valid = to_categorical(y_valid, 4)
y_train.shape
featuresTrain = torch.from_numpy(X_train).float()

targetsTrain = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long
featuresValid = torch.from_numpy(X_valid).float()

targetsValid = torch.from_numpy(y_valid).type(torch.LongTensor) # data type is long
X_test = df_test.drop(['id'], axis = 1)

X_test.shape
X_test_scaled = scaler.transform(X_test)
featuresTest = torch.from_numpy(X_test_scaled).float()
batch_size = 512

num_epochs = 100

# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

valid = torch.utils.data.TensorDataset(featuresValid,targetsValid)



train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)

valid_loader = DataLoader(valid, batch_size = batch_size, shuffle = True)
class NeuralNetwork(nn.Module):

    def __init__(self, n_features, n_neurons, dropouts):

        super().__init__()

        self.layer1 = nn.Linear(in_features=n_features, out_features=n_neurons[0])

        self.dropout1 = nn.Dropout(dropouts[0])

        self.layer2 = nn.Linear(in_features=n_neurons[0], out_features=n_neurons[1])

        self.dropout2 = nn.Dropout(dropouts[1])

        self.out_layer = nn.Linear(in_features=n_neurons[1], out_features=4)

     

    def forward(self, X):

        out = F.relu(self.layer1(X))

        out = self.dropout1(out)

        out = F.relu(self.layer2(out))

        out = self.dropout2(out)

        out = self.out_layer(out)

        return F.log_softmax(out, dim=1)

        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_features = 27

n_neurons = [100, 50]

dropouts = [0.3, 0.2]

model = NeuralNetwork(n_features, n_neurons, dropouts)

model = model.to(device)

# Cross Entropy Loss 

error = nn.NLLLoss()

# Adam Optimizer

learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# train_losses = []

# train_acc = []

# valid_losses = []

# valid_acc = []

num_epochs = 5

for e in range(num_epochs):

    start_time = time.time()

    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    model.train()

    for i, (features, labels) in enumerate(train_loader):

        #print(labels)

        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = error(outputs, torch.max(labels, 1)[1])

        #loss = error(outputs, labels)

        loss.backward()

        optimizer.step()

        # update training loss

        train_loss += loss.item()*features.size(0)

        

        #TODO: Calculate the accuracy on the train test

        #acc = torch.eq(outputs.round(), labels).float().mean() # accuracy

        #_, predicted = torch.max(outputs.data, 1)

        #_, actual = torch.max(labels, 1)

        #total = len(labels)

        #correct = (predicted == actual).sum()

        #train_accuracy = 100 * correct / total

        #train_losses.append(train_loss.item())

        #train_acc.append(train_accuracy.item())

        

    model.eval()

    for i, (features, labels) in enumerate(valid_loader):

        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = error(outputs, torch.max(labels, 1)[1])

        #loss = error(outputs, labels)

        valid_loss += loss.item()*features.size(0)

        

        #_, predicted = torch.max(outputs.data, 1)

        #_, actual = torch.max(labels, 1)

        #total = len(labels)

        #correct = (predicted == actual).sum()

        #accuracy = 100 * correct / total

        #valid_losses.append(loss.item())

        #valid_acc.append(accuracy.item())

    

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.dataset)

    valid_loss = valid_loss/len(valid_loader.dataset)

    elapsed_time = time.time() - start_time

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime: {:.2f}'.format(

        e+1, train_loss, valid_loss, elapsed_time))    

    

#     if e % 1 == 0:

#         print("[{}/{}], Train Loss: {} Train Acc: {}, Validation Loss : {}, Validation Acc: {} ".format(e+1,

#         num_epochs, np.round(train_loss.item(), 3), np.round(train_accuracy.item(), 3), 

#         np.round(loss.item(), 3), np.round(accuracy.item(), 3)))
test = torch.utils.data.TensorDataset(featuresTest)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

test_preds = np.zeros((len(test), 4))



for i, (x_batch,) in enumerate(test_loader):

    x_batch = x_batch.to(device)

    y_pred = model(x_batch).detach()

    test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred.cpu().numpy()
test_preds.shape
test_preds[100:110, :]
submission = pd.DataFrame({'id':df_test['id'],'A':test_preds[:, 0], 'B':test_preds[:, 1],

                          'C':test_preds[:, 2], 'D':test_preds[:, 3]})

submission.head()