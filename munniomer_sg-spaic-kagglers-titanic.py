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
print("Dimensions of train: {}".format(train_df.shape))

print("Dimensions of test: {}".format(test_df.shape))
train_df.info()
train_df.describe()
#preview of the data

train_df.head(10)
# check for duplicates in the data

sum(train_df.duplicated())
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
train_df.columns.values
sns.countplot('Survived',data=train_df)

plt.show()
# Exploring the number of survivors by gender

train_df.groupby(['Sex', 'Survived'])['Survived'].count()
train_df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

sns.countplot('Sex',hue='Survived',data=train_df,)

plt.show()
#Exploring the Pclass column

class_pivot = train_df.pivot_table(index="Pclass",values="Survived")

class_pivot.plot.bar()

plt.show()
# Exploring age column

train_df["Age"].describe()
print('Oldest survivor:',train_df['Age'].max())

print('Youngest survivor:',train_df['Age'].min())

print('Average survivor Age:',train_df['Age'].mean())
#plotting a histogram of Age



#giving the figure size(width, height)

plt.figure(figsize=(9,5), dpi = 100)



#On x-axis 

plt.xlabel('Age', fontsize = 15)

#On y-axis 

plt.ylabel('No.of survivors', fontsize=15)

#Name of the graph

plt.title('Age of the Survivors', fontsize=18)



#giving a histogram plot

plt.hist(train_df['Age'], rwidth = 0.9, bins =35)

#displays the plot

plt.show()
survived = train_df[train_df["Survived"] == 1]

died = train_df[train_df["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
def merge_relatives():

    data = [train_df, test_df]

    for dataset in data:

        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    # help(train_df.loc)

    print(train_df.head())

    

merge_relatives()
train_df.head()
# help(sns.factorplot) - it is deprecated

# help(sns.catplot)

axes = sns.factorplot('relatives','Survived', 

                      data=train_df, aspect = 2.5, )
def drop_useless_columns():

    train_df.drop(['SibSp', 'Parch', 'Ticket','Name','Cabin'], axis=1, inplace=True)

    test_df.drop(['SibSp', 'Parch', 'Ticket','Name','Cabin'], axis=1, inplace=True)

    

drop_useless_columns()
train_df.head()
def set_index():

    train_df.set_index('PassengerId',inplace=True)

    test_df.set_index('PassengerId',inplace=True)



set_index()
def encode_sex_column():

    train_df['Sex'] = train_df['Sex'].map(dict(zip(['male','female'],[0,1])))

    test_df['Sex'] = test_df['Sex'].map(dict(zip(['male','female'],[0,1])))



encode_sex_column()
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
test_output_df.head()
# Save the final CLEAN dataset as our new file!

# train_df.to_csv('clean_train.csv', index=False)
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



test_targets_df = pd.DataFrame(data=test_output_df['Survived'])

test_targets_df.columns = ['Survived']



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

        hidden_1 = 4

        # linear layer (784 -> hidden_1)

        self.fc1 = nn.Linear(6, hidden_1)

        # linear layer (n_hidden -> 10)

        self.fc2 = nn.Linear(hidden_1, 2)

        # dropout layer (p=0.2)

        # dropout prevents overfitting of data

        self.dropout = nn.Dropout(0.2)



    def forward(self, x):

        x = x.view(-1, 6)

        # add hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add output layer

        x = torch.sigmoid(self.fc2(x))

        return x

         



# initialize the NN

model = Net()

print(model)
# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer (stochastic gradient descent) and learning rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model

n_epochs = 50



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

        # forward pass: compute predicted outputs by passing inputs to the model

        

       

        output = model(data)

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

        output = model(data)

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



for data, target in test_loader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the loss

    #print('Data %s' % data)

    #print('Output %s' % output)

    loss = criterion(output, target)

    #print('Target %s' % target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)

    #print('Predicted %s' % pred)

    # compare predictions to true label

    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    #print('Correct %s' % correct)

    # calculate test accuracy for each object class

    #break

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