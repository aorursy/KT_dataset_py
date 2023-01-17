# Same old same old

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Fancy library for graphs

import seaborn as sns

sns.set_style('darkgrid')



# Now this is random

import random



# Readymade Machine Learning imports

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Doctor Strange feels

import time



# Let's see what the csv has in store for us

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
data = [train, test]

for dataset in data:

    dataset.drop(['Name', 'PassengerId', 'Ticket'], axis=1,inplace=True)

train.head()
train.describe(include=['O'])
train.info()
test.info()
for dataset in data:

    dataset.drop(['Cabin'], inplace=True, axis=1)

    

train.head()
grid = sns.FacetGrid(train, row='Pclass', col='Sex')

grid.map(plt.hist, 'Age', bins=25)

grid.add_legend()
for dataset in data:

    dataset['Sex'] = dataset['Sex'].apply(lambda x:1 if x == 'female' else 0)

sns.swarmplot(x='Survived', y='Age', hue='Sex', data=train)
# Estimating values for missing entries in column 'Age'

"""

We're going to calculate median Age for every combination of values Sex and Pclass, and replace the missing values with the median

"""

for dataset in data:

    for i in range(0, 2):          # Iterating over 'Sex' 0 or 1

        for j in range(0, 3):      # Iterating over 'Pclass' 1, 2 or 3

            guess_df = dataset.loc[(dataset['Sex'] == i) & (dataset['Pclass']==j+1)]['Age'].dropna()

            

            age_guess = guess_df.median()

            

            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == i) & (dataset['Pclass']==j+1), ['Age']] = age_guess

            

    dataset['Age'] = dataset['Age'].astype(int)

                        

train.head()

                           

train['AgeBand'] = pd.cut(train['Age'], 5)

train[['AgeBand', 'Survived']].groupby(['AgeBand']).mean()
for dataset in data:

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age']) > 16 & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age']) > 32 & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age']) > 48 & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[(dataset['Age']) > 64 & (dataset['Age'] <= 80), 'Age'] = 4



train.head()
# We don't need the AgeBand Feature now, so it's gotta go!

train.drop(['AgeBand'], axis=1, inplace=True)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)



train['Fareband'] = pd.qcut(train['Fare'], 4)

train[['Fareband', 'Survived']].groupby(['Fareband']).mean()
for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train.drop(['Fareband'], axis=1, inplace=True)

train.head(10)
for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(train['Embarked'].dropna().mode()[0])
train[['Embarked', 'Survived']].groupby('Embarked').mean()
for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

train.head()
X_train = train.drop('Survived', axis=1)

Y_train = train['Survived']

X_test = test

X_train.shape, Y_train.shape, X_test.shape

accuracies = pd.DataFrame()
sgd = SGDClassifier()



tic = time.time()

sgd.fit(X_train, Y_train)

tok = time.time()



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

sgd_time = round(tok-tic, 4)



algorithms = ['Stochastic Gradient Decent']

model_accuracy = [acc_sgd]

exec_time = [sgd_time]

Y_pred = sgd.predict(X_test)
knn = KNeighborsClassifier(n_neighbors = 5)



tik = time.time()

knn.fit(X_train, Y_train)

tok = time.time()



Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



algorithms.append('KNN Classifier')

model_accuracy.append(acc_knn)

exec_time.append(round(tok-tic, 4))
svc = SVC(kernel='poly', degree=8)



tic = time.time()

svc.fit(X_train, Y_train)

toc = time.time()



Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

algorithms.append('Support Vector Machines')

model_accuracy.append(acc_svc)

exec_time.append(round(toc-tic, 4))
random_forest_classifier = RandomForestClassifier(n_estimators=100)



tic = time.time()

random_forest_classifier.fit(X_train, Y_train)

toc = time.time()



Y_pred = random_forest_classifier.predict(X_test)

random_forest_classifier = round(random_forest_classifier.score(X_train, Y_train) * 100, 2)

algorithms.append('Random Forest classifier')

model_accuracy.append(random_forest_classifier)

exec_time.append(round(toc-tic, 4))
test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived" : Y_pred

})

submission.to_csv("submission.csv", index=False)
# Bringing out the big guns

import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

# To get reproducible results

torch.manual_seed(2)
# Converting the values to a Pytorch Floar tensor

def to_tensor(data):

    return [torch.FloatTensor(point) for point in data]



# Dataset class for fetching data from the dataset

class TitanicDataset(Dataset):

    def __init__(self, df, X_col, Y_col):

        self.features = df[X_col].values

        self.targets = df[Y_col].values.reshape(-1, 1)

        

    def __len__(self):

        return len(self.targets)

    

    def __getitem__(self, idx):

        return to_tensor([self.features[idx], self.targets[idx]])

    



# Splitting our training data into 90 % training and 10 % validation data

split = int(0.9*int(len(train)))



# X_cols are features for our NN. The first column is 'Survived'(index 0), which is our target/prediction Variable. So we neglect that from our features.

X_col = list(train.columns[1:])

Y_col = 'Survived'



train_data = train[:split].reset_index(drop=True)

valid_data = train[split:].reset_index(drop=True)



print(f"train DF{train.shape}\nAfter Splitting\ntrain DF {train_data.shape}\nvalid_data{valid_data.shape}")





#a, b = train_data[0]

print(train_data)



train_set = TitanicDataset(train_data, X_col, Y_col)

valid_set = TitanicDataset(valid_data, X_col, Y_col)



train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Sequential(

                    nn.Linear(7, 128),

                    # nn.Dropout(0.5),

                    nn.ReLU(),

                    nn.BatchNorm1d(128))

        self.fc2 = nn.Sequential(

                    nn.Linear(128, 256),

                    #nn.Dropout(0.5),

                    nn.ReLU(),

                    nn.BatchNorm1d(256))

        self.fc3 = nn.Sequential(

                    nn.Linear(256, 512),

                    #nn.Dropout(0.5),

                    nn.ReLU(),

                    nn.BatchNorm1d(512))

        self.fc4 = nn.Sequential(

                    nn.Linear(512, 256),

                    #nn.Dropout(0.5),

                    nn.ReLU(),

                    nn.BatchNorm1d(256))

        self.fc5 = nn.Sequential(

                    nn.Linear(256, 1),

                    nn.Sigmoid())

        

        

    def forward(self, inputs):

        x = self.fc1(inputs)

        

        x = self.fc2(x)

        

        x = self.fc3(x)

        

        x = self.fc4(x)

        

        x = self.fc5(x)

        

        return x

        

# set use_gpu as 1 if you want to use gpu

use_gpu = 1



device = torch.device("cuda" if use_gpu else "cpu")





model = Net().to(device)

print(model)

        
def calc_accuracy(Y_pred, Y_actual):

    Y_pred = torch.round(Y_pred)

    #print(f"Rounded values: {Y_pred[:5]}\nActual Values: {Y_actual[:5]}")

    correct_results_sum = (Y_pred == Y_actual).sum().float()

    acc = correct_results_sum/Y_pred.shape[0]

    return acc
num_epochs = 300





criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())

loss = []

accuracy = []

valid_loss = []

valid_accuracy = []

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

tic = time.time()

for i in range(1, num_epochs+1):

    epoch_loss = 0

    epoch_acc = 0 

    for current_batch in train_loader:

        batch_X, batch_Y = current_batch

        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        #batch_Y = batch_Y.long().squeeze()

        outputs = model.forward(batch_X)

        # print(f"Batch_Y shape: {batch_Y.shape}\noutput Shape:{outputs.shape}")

        batch_loss = criterion(outputs, batch_Y)

        #print(f"Loss shape:{batch_loss}")

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        scheduler.step()

        # print(f"Batch_Y shape: {batch_Y[:5]}\noutput Shape:{outputs[:5]}")

        batch_acc = calc_accuracy(outputs.squeeze(), batch_Y.squeeze())

        """        

        print(f"epoch loss: {epoch_loss:.4f}\tbatch_loss: {batch_loss:.4f}")

        print(f"epoch acc: {epoch_acc:.4f}\tbatch_acc: {batch_acc:.4f}")"""

        epoch_loss += batch_loss

        epoch_acc += batch_acc

        

    valid_epoch_loss = 0

    valid_epoch_acc = 0

    with torch.no_grad():

        for valid_batch in valid_loader:

            valid_X, valid_Y = valid_batch

            valid_X, valid_Y = valid_X.to(device), valid_Y.to(device)

            

            valid_prediction = model.forward(valid_X)



            valid_batch_loss = criterion(valid_prediction, valid_Y)

            valid_batch_acc = calc_accuracy(valid_prediction.squeeze(), valid_Y.squeeze())



            valid_epoch_loss += valid_batch_loss

            valid_epoch_acc += valid_batch_acc

    

    epoch_loss /= len(train_loader)

    epoch_acc /= len(train_loader)

    

    valid_epoch_loss /= len(valid_loader)

    valid_epoch_acc /= len(valid_loader)

    

    

    loss.append(epoch_loss)   

    accuracy.append(epoch_acc)

    

    if i % 25 == 0:

        print("======================================================")

        print(f'Yo, this is epoch number {i}')

        print(f"Loss : {epoch_loss:.4f}\t\t\tAccuracy: {epoch_acc:.4f}")

        print(f"Validation Loss : {valid_epoch_loss:.4f}\tValidation Accuracy: {valid_epoch_acc:.4f}")

toc = time.time()
device = 'GPU'

if not use_gpu:

    device = 'CPU'

algorithms.append('Neural Network (' + device + ')')

# algorithms.append('Neural Network (gpu)')

model_accuracy.append(round(epoch_acc.item()*100, 2))

exec_time.append(round(toc-tic, 4))
accuracies['Algorithms'] = algorithms

accuracies['Accuracy'] = model_accuracy

accuracies['Training Time'] = exec_time

accuracies.reset_index(drop=True, inplace=True)

accuracies