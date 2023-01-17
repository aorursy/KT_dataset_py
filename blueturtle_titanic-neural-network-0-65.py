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
#Read the train data and remove some columns

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

y_train = train_data.Survived

train_data = train_data.drop(columns = ["Ticket", "Cabin", "Survived"])



from sklearn.preprocessing import MinMaxScaler



Features = ["Fare"]

scalar = MinMaxScaler()

scaled = scalar.fit_transform(train_data[Features])

train_data.Fare = scaled



#Read the test data and remove some columns

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data = test_data.drop(columns = ["Ticket", "Cabin"])



scaled = scalar.fit_transform(test_data[Features])

test_data.Fare = scaled
train_data['Title'] = train_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



#cleanup rare title names

stat_min = 10

title_names = (train_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



train_data['Title'] = train_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

test_data['Title'] = test_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



#cleanup rare title names

stat_min = 10

title_names = (test_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



test_data['Title'] = test_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
#Encode the data to convert text into numbers that the model can use.

#Encode Title, Sex, and Embarked

X_encode_train = pd.get_dummies(train_data, columns = ["Title", "Sex", "Embarked"])

X_encode_test = pd.get_dummies(test_data, columns = ["Title", "Sex", "Embarked"])



#Drop the name column now that we have extracted Title

X_encode_test = X_encode_test.drop(columns = ["Name"])

X_encode_train = X_encode_train.drop(columns = ["Name"])
#Fill in missing Ages by using the average age for that particular Title group.

#On the assumption that all Masters will be children, all Miss will be young females, Mrs will be older females and Mr will be adult males.



X_encode_train['Age'] = train_data.groupby('Title').Age.transform(lambda x: x.fillna(x.mean()))

X_encode_test['Age'] = test_data.groupby('Title').Age.transform(lambda x: x.fillna(x.mean()))



#Although there is only 1 missing value for Fare, it was good coding practice

X_encode_train['Fare'] = train_data.groupby('Pclass').Fare.transform(lambda x: x.fillna(x.mean()))

X_encode_test['Fare'] = test_data.groupby('Pclass').Fare.transform(lambda x: x.fillna(x.mean()))

#Inherited code from the RandomClassifier model to customise for the NN if possible.



from scipy.stats import uniform, truncnorm, randint

from sklearn.model_selection import RandomizedSearchCV



#Find optimal paramters

model_params = {

    # randomly sample numbers from 4 to 204 estimators

    'n_estimators': randint(4,200),

    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1

    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),

    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)

    'min_samples_split': uniform(0.01, 0.199)

}



#Create RandomizedSearch

#clf = RandomizedSearchCV(model, model_params, n_iter=100, cv=5, random_state=0)

#clf.fit(X_train, y)

#results = pd.DataFrame(clf.cv_results_)

#print(clf.best_estimator_)

#print(clf.best_params_)

#print(results)

#Create the Neural Network

import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.fc1 = nn.Linear(16, 16)

        self.fc2 = nn.Linear(16, 2) #Output nodes must be equal to the number of options that y can take...

        #...in this case they can either survive == 1, or not survive == 0.

        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x)

    

net = Net()

print(net)
import tensorflow as tf

import torch



#As a test of the model we will input one instance of the training data to check we get an output

#Pytorch NN only take Tensors as inputs. Therefore we must use the torch module to convert X into a Tensor.

X = torch.tensor(X_encode_train.loc[2].values) 



net = net.float()

z = net(X.float())

z
#Create the optimiser

import torch.optim as optim



optimizer = optim.Adam(net.parameters(), lr= 0.01)



EPOCHS = 15



for epoch in range(EPOCHS):

    net.zero_grad()

    X = torch.tensor(X_encode_train.values)

    y = torch.tensor(y_train.values)

    output = net(X.float())

    loss = F.nll_loss(output, y)

    loss.backward()

    optimizer.step()
# Applying model on the test data

from torch.autograd import Variable



with torch.no_grad():

    test_result = net(torch.tensor(X_encode_test.values).float())

    

values, labels = torch.max(test_result, 1)

predictions = labels.data.numpy()

#Save the predictions to a csv



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")