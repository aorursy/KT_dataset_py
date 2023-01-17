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
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader, Dataset, ConcatDataset



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



import time
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# To process train as well as test dataset simultaneously

combine = [train_df, test_df]



for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name', 'PassengerId'], axis=1)



# Every time when the columns of train and test dataset changed

# we should redefine combine

combine = [train_df, test_df]



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



guess_ages = np.zeros((2,3))

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_mean = guess_df.mean()

            age_std = guess_df.std()

            age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]



for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



freq_port = train_df.Embarked.dropna().mode()[0]



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)



print (train_df.shape, test_df.shape)

################Finish Raw Data Process##################
testAttNum = 7

trainAttNum = 8



# Split data into features(pixels) and labels(numbers from 0 to 1)

targets_numpy = train_df.Survived.values

features_numpy = train_df.loc[:, train_df.columns != "Survived"].values



# Train test split. Size of train data is 80% and size of test data is 20%. 

features_train, features_test, targets_train, targets_test = train_test_split( 

                                                    features_numpy,

                                                    targets_numpy,

                                                    test_size = 0.2,

                                                    random_state = 42) 



# Create feature and targets tensor for train set. 

#featuresTrain = torch.from_numpy(features_train).type(torch.FloatTensor)

#targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) 

featuresTrain = torch.from_numpy(features_numpy).type(torch.FloatTensor)

targetsTrain = torch.from_numpy(targets_numpy).type(torch.LongTensor) 



# Create feature and targets tensor for test set.

featuresTest = torch.from_numpy(features_test).type(torch.FloatTensor)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) 



batch_size = 50

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)

train_loader = torch.utils.data.DataLoader(train, 

                                            batch_size = batch_size, 

                                            shuffle = False)

###################Finish Data Preparation##########################
class CNNModel(nn.Module):

    ### TODO: choose an architecture, and complete the class

    def __init__(self):

        super(CNNModel, self).__init__()

        ## Define layers of a CNN

        # convolutional layer 1

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), padding=1)

        # convolutional layer 2

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 2), padding=1)

        # convolutional layer 3

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 2), padding=1)

        # max pooling layer

        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

        # linear layer

        self.fc1 = nn.Linear(64*7, 1000)

        self.fc2 = nn.Linear(1000, 500)

        self.fc3 = nn.Linear(500, 2)

        self.dropout = nn.Dropout(0.25)#0.5

        

        self.bn1 = nn.BatchNorm2d(16)

        self.bn2 = nn.BatchNorm2d(32)

        self.bn3 = nn.BatchNorm2d(64)

    

    def forward(self, x):

        ## Define forward behavior

        x = self.bn1(self.pool(F.relu(self.conv1(x))))

        x = self.bn2(self.pool(F.relu(self.conv2(x))))

        x = self.bn3(self.pool(F.relu(self.conv3(x))))

        #flatten

        x = x.view(x.size(0), -1)

        #x = self.dropout(x)

        x = F.relu(self.fc1(x))

        #x = self.dropout(x)

        x = F.relu(self.fc2(x))

        #x = self.dropout(x)

        x = self.fc3(x)

        return x
model = CNNModel()

error = nn.CrossEntropyLoss()

learning_rate = 0.013

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# CNN model training

count = 0

loss_list = []

iteration_list = []

accuracy_list = []

num_epochs = 100

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        model.train()

        train = images.reshape(images.shape[0],1,1, 7)

        # Clear gradients

        optimizer.zero_grad()

        # Forward propagation

        outputs = model(train)       

        # Calculate softmax and ross entropy loss

        loss = error(outputs, labels)

        # Calculating gradients

        loss.backward()

        # Update parameters

        optimizer.step()

    

    # Iterate through test dataset

    model.eval()

    test = featuresTest.reshape(featuresTest.shape[0],1,1, 7)

    # Forward propagation

    outputs = model(test)

    # Get predictions from the maximum value

    _, predicted = torch.max(outputs.data, 1)

    # Total number of labels

    total = len(targetsTest)

    correct = (predicted == targetsTest).sum()

    accuracy = 100 * correct / float(total)



    # store loss and iteration

    loss_list.append(loss.item()*100)

    iteration_list.append(epoch)

    accuracy_list.append(accuracy)

           

    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(epoch, loss.item(), accuracy))



###############Start predictation#####################

model.eval()

test = torch.from_numpy(test_df.values).type(torch.FloatTensor)

outest = model(test.reshape(-1, 1, 1, 7))



_, predicted = torch.max(outest.data, 1)

idx = np.arange(start= 892, stop= 892 +test.shape[0])

df2 = pd.DataFrame({'Passenger': idx, 

                    'Survived': predicted})

df2.to_csv('./jrxie.csv', index= False)

print ("Finish!!")
fig, ax = plt.subplots()

plt.plot(loss_list, label='Loss', alpha=0.5)

plt.plot(accuracy_list, label='Accuracy', alpha=0.5)

plt.title("Training Losses")

plt.legend()