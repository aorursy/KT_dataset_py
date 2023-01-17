# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This is just data pre processing

# main code starts from importing torch libraries 10-15 cell below

# only neural network model is implemented here



df = pd.read_csv('/kaggle/input/indian-patient-si359/indian_liver_patient_train.csv')

df.head()

df['Gender'].head()
lis = []

print(df.count())

for gen in df['Gender']:

    if gen == 'Male':

        lis.append(1)

    else: 

        lis.append(2)

        

print(len(lis))
df['sex'] = lis

df.head()
df = df.drop('Gender', axis=1)
type(df['sex'])

print(df.Age.count())
for i in range(df.Age.count()):

    df.Age[i] = df.Age[i]/100

    
df.head()
df.columns

convert_dict = {'Age': float}
df = df.astype(convert_dict)
for i in range(df.Age.count()):

    df.Age[i] = df.Age[i]/100

    
df.head()
lis = []

print(df.count())

for gen in df['Gender']:

    if gen == 'Male':

        lis.append(1)

    else: 

        lis.append(2)

        

print(len(lis))

df['sex'] = lis
df = df.drop('Gender', axis=1)

df.head()
dfl = df.Category

dfid = df.ID

df = df.drop('Category', axis=1)

df = df.drop('ID', axis=1)

df['sex'].head()
type(list(df.columns))

headings = list(df.columns)

print(headings)



for heading in headings:

    if heading=='sex':

        continue

    dft = df[heading]

    df[heading] = (dft - dft.mean())/dft.std()
pro_df = df
##Main code starts here

import torch

import torch.utils.data

import torch.nn as nn

import torchvision.datasets as dsets

import torchvision.transforms as transforms

from torch.autograd import Variable



epochs = 10
def get_matrix(path, train):

    df = pd.read_csv(path)

    df = df.drop([169, 201, 248], axis=0)

    df2 = df

    lis = []

    for gen in df['Gender']:

        if gen == 'Male':

            lis.append(1)

        else: 

            lis.append(2)



    print("count", len(lis))

    df['sex'] = lis

    dfid = df['ID']

    df = df.drop('Gender', axis=1)

    df = df.drop('ID', axis=1)

    dfl = df['Category']



#     if train:

#         for i in range(df['Category'].count()):

#             if df['Category'][i]==2:

#                 df['Category'][i]=0

#         dfl = df['Category']

#         df = df.drop('Category', axis=1)

    df = df.drop('Category', axis=1)

    headings = list(df.columns)

    print("headings", headings)



    for heading in headings:

        if heading=='sex':

            continue

        dft = df[heading]

        df[heading] = (dft - dft.mean())/dft.std()

    

    data = df.as_matrix()

    

    if train == False:

        return data

    

    label = dfl.as_matrix()

    label = label-1

#     print(data.shape, label.shape)

    return data, label

    

    
data, label = get_matrix('/kaggle/input/indian-patient-si359/indian_liver_patient_train.csv', True)

label1 = label[0:397]

data2 = data[397:456, :]

print(data.shape, label1.shape, data2.shape)

print(label)
class Dataload(torch.utils.data.Dataset):

    def __init__(self, train = True, val = False):

        self.train = train

        self.val = val

        if self.train:

            self.data, self.label = get_matrix('/kaggle/input/indian-patient-si359/indian_liver_patient_train.csv', self.train)

            self.data = self.data[0:397, :]

            self.label = self.label[0:397]

        elif self.val:

            self.data, self.label = get_matrix('/kaggle/input/indian-patient-si359/indian_liver_patient_train.csv', True)

            self.data = self.data[397:456, :]

            self.label = self.label[397:456]

        else:

            self.data = get_matrix('/kaggle/input/indian-patient-si359/indian_liver_patient_test.csv', self.train)

            

    def __getitem__(self, index):

        if self.train or self.val:

            return torch.tensor(self.data[index].astype(float)), torch.tensor(self.label[index])

        else:

            return torch.tensor(self.data[index].astype(float))

        

    def __len__(self):

        return self.data.shape[0]
train_data = Dataload(train = True)

val_data = Dataload(train = False, val = True)

# test_data  = Dataload(train = False)
train_dataloader = torch.utils.data.DataLoader(dataset = train_data, batch_size = 397, shuffle = True, num_workers = 0)



val_dataloader = torch.utils.data.DataLoader(dataset = val_data, batch_size = 59, shuffle = False, num_workers = 0)



# test_dataloader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 124, shuffle = False)
class Neural_Network(nn.Module):

    def __init__(self,):

        super().__init__()

        self.hidden1 = nn.Linear(10, 5)

#         self.hidden2 = nn.Linear(8, 4)

        self.out = nn.Linear(5, 1)

        

    def forward(self, x):

        x = self.hidden1(x)

        x = torch.nn.functional.relu(x)

#         x = self.hidden2(x)

#         x = torch.nn.functional.relu(x)

        x = self.out(x)

#         x = torch.sigmoid(x)

        return x
model = Neural_Network()

model = model.float()

model

params = model.parameters()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params, lr = 0.01, weight_decay = 0.1, momentum = 0.9)
torch.autograd.set_detect_anomaly(True)

for epoch in range(300):

    total=0

    correct=0

    for i, (data_point, label) in enumerate(train_dataloader):

#         data_point = Variable(data_point).cuda()

#         label = Variable(label).cuda()

#         label[label==2] = 0

#         label = label -1

        optimizer.zero_grad()

        outputs = model(data_point.float())

        

#         outputs[outputs != outputs] = 0.00001

#         print(outputs != outputs)

#         c = outputs != outputs

#         for j in range(400):

#             if c[j]==1:

#                 print(j)

#         print(outputs.squeeze().size())

        if epoch==100 and i==1:

            print("outputs", torch.sigmoid(outputs.squeeze()), "label", label)

        loss = loss_fn(outputs.squeeze(), label.float())

        loss.backward()

        optimizer.step()

#         print(loss)

        ## Computing Accuracy

        predict = (torch.sigmoid(outputs) >= 0.1).long().squeeze()

#         print(predict.size(), label.size())

        total += label.size(0)

#         print(type(correct), type((predict == label).sum().item()), (predict == label).sum().item())

        correct += (predict == label).sum().item()

        if (i) % 10 == 0:

            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, epochs, i+1, len(train_dataloader)//397, loss.item()))

    

    print("Training Accuracy ", (correct/total)*100)

    print("correct", correct, "total", total)

    correct=0

    total=0

            

    for i, (data_point, label) in enumerate(val_dataloader):

#         data_point = Variable(data_point).cuda()

#         label = Variable(label).cuda()

#         label = label -1

        outputs = model(data_point.float())

#         label[label==2] = 0

        if epoch==3:

            print("outputs", torch.sigmoid(outputs.squeeze()), "label", label)

        loss = loss_fn(outputs.squeeze(), label.float())

        ## Computing Accuracy

        predict = (torch.sigmoid(outputs) >= 0.1).long().squeeze()

        total += label.size(0)

        correct += (predict == label).sum().item()

        print ('Validation Loss', 'Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, epochs, i+1, len(val_dataloader)//59, loss.item()))

    

    print("Validation  Accuracy ", (correct/total)*100)

    correct=0

    total=0
a = torch.Tensor([float('NaN'), 1, float('NaN'), 2, 3])

print(a)

a[a != a] = 0

print(a)
df = pd.read_csv('/kaggle/input/indian-patient-si359/indian_liver_patient_train.csv')

df

df.drop([1, 10, 11], axis=0).head(15)