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

import torchvision

from torchvision import transforms, datasets

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
# Load Data

train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test=pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

submission_set = pd.read_csv("../input/Kannada-MNIST/test.csv").iloc[:,1:]



train_data=train.drop('label',axis=1)

train_targets=train['label']



test_images=test.drop('label',axis=1)

test_labels=test['label']



# Train Test Split

train_images, val_images, train_labels, val_labels = train_test_split(train_data, 

                                                                     train_targets, 

                                                                     test_size=0.2)



# Reset Index

train_images.reset_index(drop=True, inplace=True)

val_images.reset_index(drop=True, inplace=True)

train_labels.reset_index(drop=True, inplace=True)

val_labels.reset_index(drop=True, inplace=True)



print("Train Set")

print(train_images.shape)

print(train_labels.shape)



print("Validation Set")

print(val_images.shape)

print(val_labels.shape)



print("Validation 2")

print(test_images.shape)

print(test_labels.shape)



print("Submission")

print(submission_set.shape)
print("Train Distribution")

print(train_labels.value_counts(normalize = True))



print("\nSubmission Distribution")

print(test_labels.value_counts(normalize = True))
# Train

train_data=torch.from_numpy(train_images.values).float().view(train_images.shape[0],28,28)

train_targets=torch.from_numpy(train_labels.values).long().view(train_labels.shape[0])



# Validation

val_data=torch.from_numpy(val_images.values).float().view(val_images.shape[0],28,28)

val_targets=torch.from_numpy(val_labels.values).long().view(val_labels.shape[0])



# Test

test_data=torch.from_numpy(test_images.values).float().view(test_images.shape[0],28,28)

test_targets=torch.from_numpy(test_labels.values).long().view(test_labels.shape[0])



submission_data=torch.from_numpy(submission_set.values).float().view(submission_set.shape[0],28,28)



print(train_targets.shape)

print(train_data.shape)



print(val_targets.shape)

print(val_data.shape)



print(test_targets.shape)

print(test_data.shape)



print(submission_data.shape)
print("Train Data")

fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10,10))

fig.subplots_adjust(hspace=.3)

for i in range(5):

    for j in range(5):

        rand_int = np.random.randint(train_data.shape[0])

        ax[i][j].axis('off')

        ax[i][j].imshow(train_data[rand_int])

        ax[i][j].set_title(train_targets[rand_int].item())

plt.show()



print("Valid 1 Data")

fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10,10))

fig.subplots_adjust(hspace=.3)

for i in range(5):

    for j in range(5):

        rand_int = np.random.randint(val_data.shape[0])

        ax[i][j].axis('off')

        ax[i][j].imshow(val_data[rand_int])

        ax[i][j].set_title(val_targets[rand_int].item())

plt.show()



print("Valid 2 Data")

fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10,10))

fig.subplots_adjust(hspace=.3)

for i in range(5):

    for j in range(5):

        rand_int = np.random.randint(test_data.shape[0])

        ax[i][j].axis('off')

        ax[i][j].imshow(test_data[rand_int])

        ax[i][j].set_title(test_targets[rand_int].item())

plt.show()
print("Submission Data")

fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10,10))

fig.subplots_adjust(hspace=.3)

for i in range(5):

    for j in range(5):

        rand_int = np.random.randint(submission_data.shape[0])

        ax[i][j].axis('off')

        ax[i][j].imshow(submission_data[rand_int], cmap='viridis')

#         ax[i][j].set_title(train_targets[rand_int].item())

plt.show()
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))



# I know these for loops look weird, but this way num_i is only computed once for each class

for i in range(10): # Column by column

    num_i = train_images[train_labels == i]

    ax[0][i].set_title(i)

    for j in range(10): # Row by row

        ax[j][i].axis('off')

        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='viridis')
batch_size = 8



train_set=torch.utils.data.TensorDataset(train_data, train_targets)

valid_set=torch.utils.data.TensorDataset(val_data, val_targets)

test_set=torch.utils.data.TensorDataset(test_data, test_targets)

submission_set=torch.utils.data.TensorDataset(submission_data) 



train_loader = torch.utils.data.DataLoader(train_set, 

                                           batch_size=batch_size, 

                                           shuffle=True)



val_loader = torch.utils.data.DataLoader(valid_set, 

                                           batch_size=batch_size, 

                                           shuffle=False)



test_loader = torch.utils.data.DataLoader(test_set,

                                          batch_size=batch_size, 

                                          shuffle=False)



submission_loader = torch.utils.data.DataLoader(submission_set,

                                          batch_size=batch_size, 

                                          shuffle=False)



classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
class Net(nn.Module):

    def __init__(self, dropout = .05):

        super().__init__()

        self.dropout = dropout

        

        self.fc1 = nn.Linear(28*28, 64)

        self.d1 = nn.Dropout(p= self.dropout)

        self.fc2 = nn.Linear(64, 128)

        self.d2 = nn.Dropout(p= self.dropout)

        self.fc3 = nn.Linear(128, 64)

        self.d3 = nn.Dropout(p= self.dropout)

        self.fc4 = nn.Linear(64, 10)



    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.d1(x)

        x = F.relu(self.fc2(x))

        x = self.d2(x)

        x = F.relu(self.fc3(x))

        x = self.d3(x)

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)



net = Net()

net.to(device)
EPOCHS = 13



loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)



nn_output = []



def get_num_correct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()



for epoch in range(EPOCHS): # 3 full passes over the data

    epoch_loss = 0

    epoch_correct = 0

    net.train()

    

    for data in train_loader:  # `data` is a batch of data

        X, y = data[0].to(device), data[1].to(device)  # X is the batch of features, y is the batch of targets.

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)

        tloss = F.nll_loss(output, y)  # calc and grab the loss value

        tloss.backward()  # apply this loss backwards thru the network's parameters

        optimizer.step()  # attempt to optimize weights to account for loss/gradients 

        

        epoch_loss += tloss.item()

        epoch_correct += get_num_correct(output, y)

    

    # Evaluation with the validation set

    net.eval() # eval mode

    val_loss = 0

    val_correct = 0

    with torch.no_grad():

        for data in val_loader:

            X, y = data[0].to(device), data[1].to(device)

            

            preds = net(X.view(-1,784)) # get predictions

            vloss = F.cross_entropy(preds, y) # calculate the loss

            

            val_correct += get_num_correct(preds, y)

            val_loss += vloss.item()

    

    tmp_nn_output = [epoch + 1,EPOCHS,

                     epoch_loss/len(train_loader.dataset),epoch_correct/len(train_loader.dataset)*100,

                     val_loss/len(val_loader.dataset), val_correct/len(val_loader.dataset)*100]

    nn_output.append(tmp_nn_output)

    

    # Print the loss and accuracy for the validation set

    print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} valid loss: {:.4f} acc: {:.4f}'

        .format(*tmp_nn_output))
pd_results = pd.DataFrame(nn_output, columns = ['epoch','total_epochs','train_loss','train_acc','valid_loss','valid_acc'])

pd_results.head()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(pd_results['epoch'],pd_results['valid_loss'], label='validation_loss')

axes[0].plot(pd_results['epoch'],pd_results['train_loss'], label='train_loss')



axes[1].plot(pd_results['epoch'],pd_results['valid_acc'], label='validation_acc')

axes[1].plot(pd_results['epoch'],pd_results['train_acc'], label='train_acc')

axes[1].legend()
num_classes = len(classes)



# Use the validation set to make a confusion matrix

net.eval() # good habit I suppose

predictions = torch.LongTensor().to(device) # Tensor for all predictions



# Goes through the val set

for images, _ in val_loader:

    preds = net(images.view(-1,784).to(device))

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)



# Make the confusion matrix

cmt = torch.zeros(num_classes, num_classes, dtype=torch.int32)

for i in range(len(val_labels)):

    cmt[val_labels[i], predictions[i]] += 1
cmt
# Time to get the network's predictions on the test set

# Put the test set in a DataLoader



net.eval() # Safety first

predictions = torch.LongTensor().to(device) # Tensor for all predictions



# Go through the test set, saving the predictions in... 'predictions'

for images in submission_loader:

    preds = net(images[0].view(-1,784).to(device))

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)
# Read in the sample submission

submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")



# Change the label column to our predictions 

# Have to make sure the predictions Tensor is on the cpu

submission['label'] = predictions.cpu().numpy()

# Write the dataframe to a new csv, not including the index

submission.to_csv("predictions.csv", index=False)
submission.head()