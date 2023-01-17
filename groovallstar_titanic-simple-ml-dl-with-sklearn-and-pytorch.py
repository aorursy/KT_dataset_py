import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

combined_data = pd.concat([train_data, test_data])
combined_data.head()
print('# of data points: {}'.format(len(combined_data)))

print('# of features: {}'.format(len(combined_data.columns)-1))

print('Unique data types: {}'.format(combined_data.dtypes.unique()))
# Since we had python objects let's see the categorical features

combined_data.select_dtypes(include=['O']).columns.tolist()
# Let's see if any features contain null/NaN values

print(combined_data.isnull().any())
combined_data.info()
print(train_data.Age.isnull().sum(), '/', len(train_data))

print(test_data.Age.isnull().sum(), '/', len(test_data))
# We have 38% of survival rate

# Passengers are 29.6 years old on average

# Average fare was 32$

train_data.describe()
train_data.describe(include=['O'])
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)

train_data['Fare'].fillna(train_data['Fare'].value_counts().index[0], inplace=True)



test_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

test_data['Embarked'].fillna(train_data['Embarked'].value_counts(), inplace=True)

test_data['Fare'].fillna(train_data['Fare'].value_counts().index[0], inplace=True)
train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean()
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

test_data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
train_data = pd.get_dummies(train_data, drop_first=True)

test_data = pd.get_dummies(test_data, drop_first=True)
train_data.head()
test_data.head()
train_data.iloc[:, 2:].columns.tolist()

test_data.columns.tolist()
features = train_data.iloc[:, 2:].columns.tolist()

target = train_data.loc[:, 'Survived'].name
from scipy.stats import pearsonr

correlations = {}



for feature in features:

    data_temp = train_data[[feature, target]]

    print(data_temp)

    x1 = data_temp[feature].values

    x2 = data_temp[target].values

    key = feature + ' vs ' + target

    print(key)

    correlations[key] = pearsonr(x1, x2)[0]
correlations = pd.DataFrame(correlations, index=['Value']).T
correlations.loc[correlations['Value'].abs().sort_values(ascending=False).index]
train_data.plot.scatter(x='Fare', y='Pclass')
train_data.plot.scatter(x='Fare', y='Survived')
print(len(train_data[train_data['Fare'] > 400]), '/', len(train_data))

print(len(train_data[(train_data['Fare'] > 80) & (train_data['Fare'] < 200)]), 

     '/', len(train_data))

print(len(train_data[(train_data['Fare'] > 0) & (train_data['Fare'] < 80)]),

     '/', len(train_data))
train_data['Age'].plot.hist(stacked=True)
train_data['Age'].plot.hist(by=train_data['Survived'])
import torch

import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(8, 512)

        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.2)

        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        return x



model = Net()

print(model)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
X_train = train_data.iloc[:, 2:].values

y_train = train_data.loc[:, 'Survived'].values
from torch.autograd import Variable



batch_size = 64

n_epochs = 10000

batch_no = len(X_train) // batch_size

train_loss = 0

train_loss_min = np.Inf



for epoch in range(n_epochs):

    for i in range(batch_no):

        start = i * batch_size

        end = start + batch_size

        x_var = Variable(torch.FloatTensor(X_train[start:end]))

        y_var = Variable(torch.tensor(y_train[start:end]))

        

        optimizer.zero_grad()

        output = model(x_var)

        loss = criterion(output, y_var)

        loss.backward()

        optimizer.step()

        

        values, labels = torch.max(output, 1)

        num_right = np.sum(labels.data.numpy() == y_train[start:end])

        train_loss += loss.item() * batch_size

        

    train_loss = train_loss / len(X_train)

    if train_loss <= train_loss_min:

        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".\

             format(train_loss_min, train_loss))

        torch.save(model.state_dict(), 'model.pt')

        train_loss_min = train_loss

        

    if epoch % 100 == 0:

        print('epoch: {}\t Train Loss: {} \tTrain Accuracy: {}'.\

              format(epoch+1, train_loss, num_right / len(y_train[start:end])))

        
model.load_state_dict(torch.load('model.pt'))
X_test = test_data.iloc[:, 1:].values

X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=False)

with torch.no_grad():

    test_result = model(X_test_var)

values, labels = torch.max(test_result, 1)

survived = labels.data.numpy()
survived
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'], 'Survived':survived})
submission.head()
submission.to_csv('DNN_Submission', index=False)