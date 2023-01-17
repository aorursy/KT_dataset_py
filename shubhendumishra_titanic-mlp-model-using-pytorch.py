# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



import os

print(os.listdir("../input"))

import torch
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")

test_output_df = pd.read_csv("../input/gender_submission.csv")
#test_output_df.set_index('PassengerId',inplace=True)
test_df.head()
test_output_df.head()
print("Dimensions of train: {}".format(train_df.shape))

print("Dimensions of test: {}".format(test_df.shape))
train_df.info()
train_df.describe()
#preview of the data

train_df.head(10)
# check for duplicates in the data

sum(train_df.duplicated())

sum(test_df.duplicated())
def find_missing_data():

    print('Training Data\n')

    train_total = train_df.isnull().sum().sort_values(ascending=False)

    train_percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

    train_percent_2 = (round(train_percent_1, 1)).sort_values(ascending=False)

    train_missing_data = pd.concat([train_total, train_percent_2], axis=1, keys=['Total', '%'])

    print(train_missing_data.head(12))

    

    

    print('\n\nTest Data\n')

    test_total = test_df.isnull().sum().sort_values(ascending=False)

    test_percent_1 = test_df.isnull().sum()/test_df.isnull().count()*100

    test_percent_2 = (round(test_percent_1, 1)).sort_values(ascending=False)

    test_missing_data = pd.concat([test_total, test_percent_2], axis=1, keys=['Total', '%'])

    print(test_missing_data.head(12))

    



find_missing_data()
def merge_relatives():

    data = [train_df, test_df]

    for dataset in data:

        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    # help(train_df.loc)

    print(train_df.head())

    

merge_relatives()
def drop_useless_columns():

    train_df.drop(['SibSp', 'Parch', 'Ticket','Name','Cabin'], axis=1, inplace=True)

    test_df.drop(['SibSp', 'Parch', 'Ticket','Name','Cabin'], axis=1, inplace=True)

    

drop_useless_columns()
test_output_df.head()
test_df = test_df.merge(test_output_df, on='PassengerId')

test_df.head()
def set_index():

    train_df.set_index('PassengerId',inplace=True)

    test_df.set_index('PassengerId',inplace=True)



#set_index()
def encode_sex_column():

    train_df['Sex'] = train_df['Sex'].map(dict(zip(['male','female'],[0,1])))

    test_df['Sex'] = test_df['Sex'].map(dict(zip(['male','female'],[0,1])))



encode_sex_column()
train_df.head()
def fill_age_na():

    # Reblace NAN with age averages

    age_mean= train_df["Age"].mean()

    train_df["Age"] = train_df["Age"].fillna(age_mean)

    

fill_age_na()
train_df.dropna(inplace=True)

train_df.info()
def encode_embarked_column():

    train_df['Embarked'] = train_df['Embarked'].map(dict(zip(['S','C','Q'],[0,1,2])))

    test_df['Embarked'] = test_df['Embarked'].map(dict(zip(['S','C','Q'],[0,1,2])))



encode_embarked_column()
train_df.head()
def scale_age():

    min_age = train_df['Age'].min()

    max_age = train_df['Age'].max()



    train_df['Age'] = train_df['Age']/max_age

    

    test_min_age = test_df['Age'].min()

    test_max_age = test_df['Age'].max()

    

    test_df['Age'] = test_df['Age']/test_max_age

    



scale_age()
train_df.head()
test_df.head()
def scale_fare():

    min_age = train_df['Fare'].min()

    max_age = train_df['Fare'].max()



    train_df['Fare'] = train_df['Fare']/max_age

    

    test_min_age = test_df['Fare'].min()

    test_max_age = test_df['Fare'].max()

    

    test_df['Fare'] = test_df['Fare']/test_max_age

    



scale_fare()
train_df['relatives'].value_counts()
train_df['relatives'] = train_df['relatives'].apply(lambda x: 3 if x>=3 else x)

train_df['relatives'].value_counts()
from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler



# number of subprocesses to use for data loading

num_workers = 0



# percentage of training set to use as validation

valid_size = 0.2



# obtain training indices that will be used for validation

num_train = len(train_df)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



targets_df = pd.DataFrame(data=train_df['Survived'])

targets_df.columns = ['Survived']



del train_df['Survived']



train = torch.utils.data.TensorDataset(torch.Tensor(np.array(train_df)), torch.tensor(targets_df['Survived'].values))



train_loader = torch.utils.data.DataLoader(train,sampler=train_sampler, num_workers=num_workers)



valid_loader = torch.utils.data.DataLoader(train,sampler=valid_sampler, num_workers=num_workers)



test_targets_df = pd.DataFrame(data=test_df['Survived'])

test_targets_df.columns = ['Survived']



del test_df['Survived']



test = torch.utils.data.TensorDataset(torch.Tensor(np.array(test_df)), torch.tensor(test_targets_df['Survived'].values))



test_loader = torch.utils.data.DataLoader(test,num_workers=num_workers)
train_loader
import torch.nn as nn

import torch.nn.functional as F



# define the NN architecture

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # number of hidden nodes in each layer (512)

        hidden_1 = 10

        hidden_2 = 10

        # linear layer (784 -> hidden_1)

        self.fc1 = nn.Linear(6, hidden_1)

        # linear layer (n_hidden -> 10)

        self.fc2 = nn.Linear(hidden_1, hidden_2)

        

        self.fc3 = nn.Linear(hidden_2, 2)

        # dropout layer (p=0.2)

        # dropout prevents overfitting of data

        self.dropout = nn.Dropout(0.2)



    def forward(self, x):

        x = x.view(-1, 6)

        # add hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = torch.sigmoid(self.fc3(x))

        return x

         



# initialize the NN

model = Net()

print(model)
# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer (stochastic gradient descent) and learning rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model

n_epochs = 60



# initialize tracker for minimum validation loss

valid_loss_min = np.Inf # set initial "min" to infinity



for epoch in range(n_epochs):

    # monitor training loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train() # prep model for training

    for data, target in train_loader:

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        output = model(data[:,1:7])

        # calculate the loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval() # prep model for evaluation

    for data, target in valid_loader:

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data[:,1:7])

        # calculate the loss

        loss = criterion(output, target)

        # update running validation loss 

        valid_loss += loss.item()*data.size(0)

        

    # print training/validation statistics 

    # calculate average loss over an epoch

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch+1, 

        train_loss,

        valid_loss

        ))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model.pt')

        valid_loss_min = valid_loss
model.load_state_dict(torch.load('model.pt'))
test_df.head()
test_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval() # prep model for evaluation

results=[]

for data, target in test_loader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data[:,1:7])

    #print(data)

    # calculate the loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)

    results.append([int(data[:,0].item()), pred.item()])

    # compare predictions to true label

    correct = pred.eq(target.view_as(pred))

  

    # calculate test accuracy for each object class

    #break

    #print('Output %s Target %s Predicted %s Correct %s' % (output,target.item(),pred.item(),correct.item()))

    for i in range(len(target)):

        label = target.data

        class_correct[label] += correct.item()

        class_total[label] += 1



# calculate and print avg test loss

test_loss = test_loss/len(test_loader.sampler)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            str(i), 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
results
output = pd.DataFrame(results, columns=["PassengerId", "Survived"])
output.head()
output.to_csv("output.csv",index=False)