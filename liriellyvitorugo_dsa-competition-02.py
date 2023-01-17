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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import torch

from torch import nn, optim

import torch.utils.data

import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler



%matplotlib inline
from sklearn import model_selection

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression, ARDRegression, OrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit

from sklearn.linear_model import SGDRegressor, TheilSenRegressor, RANSACRegressor, PassiveAggressiveRegressor, HuberRegressor

from sklearn.linear_model import RidgeCV, SGDClassifier, PassiveAggressiveClassifier, Perceptron, BayesianRidge, MultiTaskElasticNetCV

from sklearn.linear_model import MultiTaskElasticNet, ElasticNetCV, MultiTaskLassoCV, LassoLars, Lasso

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVR, SVC

from sklearn.naive_bayes import MultinomialNB

from collections import Counter

from scipy import interp

from sklearn import grid_search
treino = pd.read_csv('dataset_treino.csv')

teste = pd.read_csv("dataset_teste.csv")
treino.info()
sns.countplot(x="ENERGY STAR Score", data=treino)
sns.distplot(treino['ENERGY STAR Score'])
sns.heatmap(treino.corr(), annot=True)
def tratamento1(x):

    x = str(x)

    aux = x.split(";")

    if len(aux[len(aux)-1]) <= 10:

        return aux[len(aux)-2].strip(" ").replace("\u200b", "").replace("nan", "0000000000")

    else:

        return aux[len(aux)-1].strip(" ").replace("\u200b", "").replace("nan", "0000000000")



treino['BBL'] = treino["BBL - 10 digits"].apply(lambda x: tratamento1(x))

teste['BBL'] = teste["BBL - 10 digits"].apply(lambda x: tratamento1(x))
treino['Borough'] = treino['BBL'].apply(lambda x: int(x[:1]))

teste['Borough'] = teste['BBL'].apply(lambda x: int(x[:1]))
treino['TaxBlock'] = treino['BBL'].apply(lambda x: int(x[1:6]))

teste['TaxBlock'] = teste['BBL'].apply(lambda x: int(x[1:6]))
treino['TaxLotNumber'] = treino['BBL'].apply(lambda x: int(x[6:]))

teste['TaxLotNumber'] = teste['BBL'].apply(lambda x: int(x[6:]))
treino['Water Required?'] = treino['Water Required?'].map({'No': 0, 'Yes': 1})

teste['Water Required?'] = teste['Water Required?'].map({'No': 0, 'Yes': 1})
y = treino['ENERGY STAR Score']

X = treino.drop(['Order', 'Property Name', 'Postal Code', 'Parent Property Id', 'Parent Property Name', 'BBL - 10 digits', 

                 'NYC Borough, Block and Lot (BBL) self-reported', 'NYC Building Identification Number (BIN)', 

                 'Address 1 (self-reported)', 'Address 2', 'Street Number', 'Street Name', 

                 'Primary Property Type - Self Selected', 'List of All Property Use Types at Property',

                 'Largest Property Use Type', 'Largest Property Use Type - Gross Floor Area (ft²)',

                 '2nd Largest Property Use Type', '2nd Largest Property Use - Gross Floor Area (ft²)', 

                 '3rd Largest Property Use Type', '3rd Largest Property Use Type - Gross Floor Area (ft²)',

                 'Metered Areas (Energy)', 'Metered Areas  (Water)', 'Release Date', 'DOF Benchmarking Submission Status',

                 'Latitude', 'Longitude', 'NTA', 'BBL', 'ENERGY STAR Score', 'Fuel Oil #1 Use (kBtu)', 

                 'Fuel Oil #2 Use (kBtu)', 'Fuel Oil #4 Use (kBtu)', 'Fuel Oil #5 & 6 Use (kBtu)', 

                 'Diesel #2 Use (kBtu)', 'District Steam Use (kBtu)', 'Property Id'], axis=1)
X[X.columns[22:]].head(2)
X2 = teste.drop(['OrderId', 'Property Name', 'Postal Code', 'Parent Property Id', 'Parent Property Name', 'BBL - 10 digits', 

                 'NYC Borough, Block and Lot (BBL) self-reported', 'NYC Building Identification Number (BIN)', 

                 'Address 1 (self-reported)', 'Address 2', 'Street Number', 'Street Name', 

                 'Primary Property Type - Self Selected', 'List of All Property Use Types at Property',

                 'Largest Property Use Type', 'Largest Property Use Type - Gross Floor Area (ft²)',

                 '2nd Largest Property Use Type', '2nd Largest Property Use - Gross Floor Area (ft²)', 

                 '3rd Largest Property Use Type', '3rd Largest Property Use Type - Gross Floor Area (ft²)',

                 'Metered Areas (Energy)', 'Metered Areas  (Water)', 'Release Date', 'DOF Benchmarking Submission Status',

                 'Latitude', 'Longitude', 'NTA', 'BBL', 'Fuel Oil #1 Use (kBtu)', 

                 'Fuel Oil #2 Use (kBtu)', 'Fuel Oil #4 Use (kBtu)', 'Fuel Oil #5 & 6 Use (kBtu)', 

                 'Diesel #2 Use (kBtu)', 'District Steam Use (kBtu)', 'Property Id'], axis=1)
X.head(10)
X.replace('Not Available', np.nan, inplace=True)

X2.replace('Not Available', np.nan, inplace=True)
columns = ['Weather Normalized Site EUI (kBtu/ft²)', 'Weather Normalized Site Electricity Intensity (kWh/ft²)',

           'Weather Normalized Site Natural Gas Intensity (therms/ft²)', 'Weather Normalized Source EUI (kBtu/ft²)', 

           'Natural Gas Use (kBtu)', 'Weather Normalized Site Natural Gas Use (therms)', 

           'Electricity Use - Grid Purchase (kBtu)',

           'Weather Normalized Site Electricity (kWh)', 'Total GHG Emissions (Metric Tons CO2e)',

           'Direct GHG Emissions (Metric Tons CO2e)', 'Indirect GHG Emissions (Metric Tons CO2e)', 

           'Water Use (All Water Sources) (kgal)', 'Water Intensity (All Water Sources) (gal/ft²)']
for i in columns:

    X[i] = X[i].astype(float)

    X2[i] = X2[i].astype(float)
X2.info()
X.info()
X = X.apply(lambda x: x.fillna(x.mean()),axis=0) 

X2 = X2.apply(lambda x: x.fillna(x.mean()),axis=0)

X2
validation_size = 0.25

seed = 7

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

X2 = scaler.transform(X2)



# how many samples per batch to load

batch_size = 64



#y_train = pd.get_dummies(y_train).values

#y_test = pd.get_dummies(y_test).values



y_train = y_train.values

y_test = y_test.values







train_target = torch.tensor(y_train.astype(np.float32))

train = torch.tensor(X_train.astype(np.float32)) 

train_tensor = torch.utils.data.TensorDataset(train, train_target) 

trainloader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)



test_target = torch.tensor(y_test.astype(np.float32))

test = torch.tensor(X_test.astype(np.float32)) 

test_tensor = torch.utils.data.TensorDataset(test, test_target) 

testloader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = batch_size, shuffle = False)



valid_target = torch.tensor(np.zeros(len(X2)))

valid = torch.tensor(X2.astype(np.float32)) 

valid_tensor = torch.utils.data.TensorDataset(valid, valid_target) 

validloader = torch.utils.data.DataLoader(dataset = valid_tensor, batch_size = 1, shuffle = False)
def init_weights(m):

    if type(m) == nn.Linear:

        torch.nn.init.kaiming_normal_(m.weight)



class Network(nn.Module):

    def __init__(self):

        super().__init__()

        

        # Inputs to hidden layer linear transformation

        self.h1 = nn.Linear(27, 128)

        # Hidden layer linear transformation

        self.bn1 = nn.BatchNorm1d(128)

        self.h2 = nn.Linear(128, 64)

        self.bn2 = nn.BatchNorm1d(64)

        # Hidden layer linear transformation

        self.h3 = nn.Linear(64, 32)

        self.bn3 = nn.BatchNorm1d(32)

        # Hidden layer linear transformation

        self.h4 = nn.Linear(32, 16)

        self.bn4 = nn.BatchNorm1d(16)

        # Hidden layer linear transformation

        self.h5 = nn.Linear(16, 8)

        self.bn5 = nn.BatchNorm1d(8)

        # Output layer, 1 classes - one for each category

        self.output = nn.Linear(8, 1)

        #Dropout p=0.3

        self.dropout = nn.Dropout(0.3)

        

    def forward(self, x):

        # Pass the input tensor through each of our operations

        x = F.relu(self.bn1(self.h1(x)))

        x = self.dropout(x)

        x = F.relu(self.bn2(self.h2(x)))

        x = self.dropout(x)

        x = F.relu(self.bn3(self.h3(x)))

        x = self.dropout(x)

        x = F.relu(self.bn4(self.h4(x)))

        x = self.dropout(x)

        x = F.relu(self.bn5(self.h5(x)))

        x = self.output(x)

        

        return x
model = Network()

model.apply(init_weights)

criterion = nn.L1Loss()



# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.ASGD(model.parameters(), lr=0.001)
epochs = 300



valid_loss_min = np.Inf # track change in validation loss



for epoch in range(1, epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for data, target in trainloader:

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        target = np.reshape(target, (*target.shape, 1))

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    

    predictions = []

    model.eval()

    for data, target in testloader:

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        target = np.reshape(target, (*target.shape, 1))

        loss = criterion(output, target)

        # update average validation loss 

        valid_loss += loss.item()*data.size(0)

        predictions.append(int(output.data.numpy()[0]))

    

    # calculate average losses

    train_loss = train_loss/len(trainloader.dataset)

    valid_loss = valid_loss/len(testloader.dataset)

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model_cifar.pt')

        valid_loss_min = valid_loss
y_train = y

X_train = X



scaler.fit(X_train)

X_train = scaler.transform(X_train)



train_target = torch.tensor(y_train.values.astype(np.float32))

train = torch.tensor(X_train.astype(np.float32)) 

train_tensor = torch.utils.data.TensorDataset(train, train_target) 

trainloader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
epochs = 200



train_loss_min = np.Inf



for epoch in range(1, epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for data, target in trainloader:

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        target = np.reshape(target, (*target.shape, 1))

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*data.size(0)

     

    # calculate average losses

    train_loss = train_loss/len(trainloader.dataset)

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(

        epoch, train_loss))

    

    # save model if validation loss has decreased

    if train_loss <= train_loss_min:

        print('train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        train_loss_min,

        train_loss))

        torch.save(model.state_dict(), 'model_cifar.pt')

        train_loss_min = train_loss
model.load_state_dict(torch.load('model_cifar.pt'))
validation_size = 0.25

seed = 7

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
rf = RandomForestRegressor(bootstrap=True, max_depth=90, max_features=3, min_samples_leaf=3, 

                              min_impurity_split=8, n_estimators=1000) #SGDRegressor(penalty='l2',alpha=0.0001, l1_ratio=0.5, shuffle=False)



rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print('RF Tunning')

print('MAE:', mean_absolute_error(y_test, pred_rf))

print('R^2:', r2_score(y_test, pred_rf))
predictions = []



model.eval()

# iterate over test data

for data, target in validloader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    pred = output

    predictions.append(int(pred.data.numpy()[0]))





rf.fit(X, y)

pred = rf.predict(X2)
final_pred = [int((x1 + x2) // 2) for x1, x2 in zip(list(pred), predictions)]



df_out = pd.DataFrame({'Property Id': teste['Property Id'].values, 'score': final_pred}, columns=['Property Id', 'score'])



df_out.to_csv("sampleSubmission.csv", sep=',', index=False)

print("Done!")