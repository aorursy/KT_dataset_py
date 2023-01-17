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

import torch.nn.functional as F

import torch.nn as nn

import torch.optim as optim

import math



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

train_labels.reset_index(drop=True, inplace=True)



val_images.reset_index(drop=True, inplace=True)

val_labels.reset_index(drop=True, inplace=True)



test_images.reset_index(drop=True, inplace=True)

test_labels.reset_index(drop=True, inplace=True)



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
print("Look at image means")

print(train_images.mean(axis = 1).mean())

print(val_images.mean(axis = 1).mean())

print(test_images.mean(axis = 1).mean())

print(submission_set.mean(axis = 1).mean())
print("Train Distribution")

print(train_labels.value_counts(normalize = True))



print("\nSubmission Distribution")

print(test_labels.value_counts(normalize = True))
IMGSIZE = 28



# Transformations for the train

train_trans = transforms.Compose(([

    transforms.ToPILImage(),

    transforms.RandomCrop(IMGSIZE),

    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),

    transforms.ToTensor(), # divides by 255

  #  transforms.Normalize((0.5,), (0.5,))

]))



# Transformations for the validation & test sets

val_trans = transforms.Compose(([

    transforms.ToPILImage(),

    transforms.ToTensor(), # divides by 255

   # transforms.Normalize((0.1307,), (0.3081,))

]))



class KannadaDataSet(torch.utils.data.Dataset):

    def __init__(self, images, labels,transforms = None):

        self.X = images

        self.y = labels

        self.transforms = transforms

         

    def __len__(self):

        return (len(self.X))

    

    def __getitem__(self, i):

        data = self.X.iloc[i,:]

        data = np.array(data).astype(np.uint8).reshape(IMGSIZE,IMGSIZE,1)

        

        if self.transforms:

            data = self.transforms(data)

            

        if self.y is not None:

            return (data, self.y[i])

        else:

            return data
batch_size = 128



train_data = KannadaDataSet(train_images, train_labels, train_trans)

val_data = KannadaDataSet(val_images, val_labels, val_trans)

test_data = KannadaDataSet(test_images, test_labels, val_trans)

submission_data = KannadaDataSet(submission_set, None, val_trans)





train_loader = torch.utils.data.DataLoader(train_data, 

                                           batch_size=batch_size, 

                                           shuffle=True)



val_loader = torch.utils.data.DataLoader(val_data, 

                                           batch_size=batch_size, 

                                           shuffle=False)



test_loader = torch.utils.data.DataLoader(test_data,

                                          batch_size=batch_size, 

                                          shuffle=False)



submission_loader = torch.utils.data.DataLoader(submission_data,

                                          batch_size=batch_size, 

                                          shuffle=False)



classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
class Net(nn.Module):

    def __init__(self, dropout = 0.40):

        super(Net, self).__init__()

        self.dropout = dropout

        

        # https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch

        #Our batch shape for input x is (1, 28, 28)

        # (Batch, Number Channels, height, width).

        #Input channels = 1, output channels = 18

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.conv1_bn = nn.BatchNorm2d(num_features=64)

        

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.conv1_1_bn = nn.BatchNorm2d(num_features=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d2_1 = nn.Dropout2d(p=self.dropout)

        

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv2_bn = nn.BatchNorm2d(num_features=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d2_2 = nn.Dropout2d(p=self.dropout)

        

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv3_bn = nn.BatchNorm2d(num_features=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d2_3 = nn.Dropout2d(p=self.dropout)

        

        #4608 input features, 256 output features (see sizing flow below)

        self.fc1 = nn.Linear(256 * 3 * 3, 512) # Linear 1

        self.d1_1 = nn.Dropout(p=self.dropout)

        #64 input features, 10 output features for our 10 defined classes

        self.fc2 = nn.Linear(in_features=512, out_features=256) # linear 2

        self.d1_2 = nn.Dropout(p=self.dropout)

        self.fc3 = nn.Linear(in_features=256, out_features=128) # linear 3

        self.d1_3 = nn.Dropout(p=self.dropout)

        self.out = nn.Linear(in_features=128, out_features=10) # linear 3

        

    def forward(self, x):

        #Computes the activation of the first convolution

        #Size changes from (1, 28, 28) to (18, 28, 28)

        x = self.conv1(x)

        x = self.conv1_bn(x)

        x = F.relu(x)

        x = self.conv1_1(x)

        x = self.conv1_1_bn(x)

        x = F.relu(x)       

        

        x = self.d2_1(x)

        x = self.pool1(x) # Size changes from (18, 28, 28) to (18, 14, 14)

        

        # Second Conv       

        x = self.conv2(x)

        x = self.conv2_bn(x)

        x = F.relu(x)

        x = self.d2_2(x)

        x = self.pool2(x) # Size changes from (18, 14, 14) to (18, 7, 7)

        

        # Third Conv       

        x = self.conv3(x)

        x = self.conv3_bn(x)

        x = F.relu(x)

        x = self.d2_3(x)

        x = self.pool3(x) # Size changes from (18, 7, 7) to (18, 3, 3)

        

        #Reshape data to input to the input layer of the neural net

        #Size changes from (18, 14, 14) to (1, 3528)

        #Recall that the -1 infers this dimension from the other given dimension

        x = x.view(-1, 256 * 3 * 3)



        #Computes the activation of the first fully connected layer

        #Size changes from (1, 4608) to (1, 64)

        #Computes the second fully connected layer (activation applied later)

        #Size changes from (1, 64) to (1, 10)

        x = F.relu(self.fc1(x))

        x = self.d1_1(x)

        

        x = F.relu(self.fc2(x))

        x = self.d1_2(x)

        

        x = F.relu(self.fc3(x))

        x = self.d1_3(x)

        

        x = self.out(x)

        return F.log_softmax(x, dim=-1)





net = Net().to(device)

net
def outputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1

    return(output)

# outputSize(64, 5, 1, 2)
# Learning Rate Finder https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

def find_lr(trn_loader, init_value = 1e-8, final_value=10., beta = 0.98):

    num = len(trn_loader)-1

    mult = (final_value / init_value) ** (1/num)

    lr = init_value

    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.

    best_loss = 0.

    batch_num = 0

    losses = []

    log_lrs = []

    for data in trn_loader:

        batch_num += 1

        #As before, get the loss for this mini-batch of inputs/outputs

        inputs = data[0].to(device)

        labels = data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        #Compute the smoothed loss

        avg_loss = beta * avg_loss + (1-beta)*loss.item()

        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Stop if the loss is exploding

        if batch_num > 1 and smoothed_loss > 4 * best_loss:

            return log_lrs, losses

        #Record the best loss

        if smoothed_loss < best_loss or batch_num==1:

            best_loss = smoothed_loss

        #Store the values

        losses.append(smoothed_loss)

        log_lrs.append(math.log10(lr))

        #Do the SGD step

        loss.backward()

        optimizer.step()

        #Update the lr for the next step

        lr *= mult

        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
net = Net().to(device)



# Loss Function

criterion = nn.CrossEntropyLoss()

# criterion = F.nll_loss



# Gradient Descent

# optimizer = optim.SGD(net.parameters(),lr=1e-1)

optimizer = optim.Adam(net.parameters(), lr=1e-1)



logs,losses = find_lr(trn_loader = train_loader)

plt.plot(logs[10:-5],losses[10:-5])
net = Net().to(device)



EPOCHS = 30

nn_output = []



# optimizer = optim.SGD(net.parameters(),lr=1e-2)

optimizer = optim.Adam(net.parameters(), lr=4e-3)

criterion = nn.CrossEntropyLoss()

# criterion = F.nll_loss



def get_num_correct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()



for epoch in range(EPOCHS):

    epoch_loss = 0

    epoch_correct = 0

    net.train()

    

    for data in train_loader:

        # `data` is a batch of data

        # Before using transforms, I used .unsqueeze(1) to enter a empty number channel array (Batch, Number Channels, height, width).

        X = data[0].to(device) # X is the batch of features

        # Unsqueeze adds a placeholder dimension for the color channel - (8, 28, 28) to (8, 1, 28, 28)

        y = data[1].to(device) # y is the batch of targets.

        

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X)  # pass in the reshaped batch (recall they are 28x28 atm)

        tloss = criterion(output, y)  # calc and grab the loss value

        tloss.backward()  # apply this loss backwards thru the network's parameters

        optimizer.step()  # attempt to optimize weights to account for loss/gradients 

        

        epoch_loss += tloss.item()

        epoch_correct += get_num_correct(output, y)

    

    # Evaluation with the validation set

    net.eval() # eval mode

    val_loss = 0

    val_correct = 0

    test_loss = 0

    test_correct = 0

    

    with torch.no_grad():

        # First Validation Set

        for data in val_loader:

            X = data[0].to(device)

            y = data[1].to(device)

            

            preds = net(X) # get predictions

            vloss = criterion(preds, y) # calculate the loss

            

            val_correct += get_num_correct(preds, y)

            val_loss += vloss.item()

        

        # Second Validation Set..

        for data in test_loader:

            X = data[0].to(device)

            y = data[1].to(device)

            

            preds = net(X) # get predictions

            tstloss = criterion(preds, y) # calculate the loss

            

            test_correct += get_num_correct(preds, y)

            test_loss += tstloss.item()

    

    tmp_nn_output = [epoch + 1,EPOCHS,

                     epoch_loss/len(train_loader.dataset),epoch_correct/len(train_loader.dataset)*100,

                     val_loss/len(val_loader.dataset), val_correct/len(val_loader.dataset)*100,

                     test_loss/len(test_loader.dataset), test_correct/len(test_loader.dataset)*100

                    ]

    nn_output.append(tmp_nn_output)

    

    # Print the loss and accuracy for the validation set

    print('Epoch [{}/{}] train loss: {:.6f} acc: {:.3f} - valid loss: {:.6f} acc: {:.3f} - Test loss: {:.6f} acc: {:.3f}'

        .format(*tmp_nn_output))
pd_results = pd.DataFrame(nn_output,

    columns = ['epoch','total_epochs','train_loss','train_acc','valid_loss','valid_acc','test_loss','test_acc']

                         )

display(pd_results)



print("Best Epoch: {}".format(pd_results.loc[pd_results.valid_acc.idxmax()]['epoch']))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(pd_results['epoch'],pd_results['valid_loss'], label='validation_loss')

axes[0].plot(pd_results['epoch'],pd_results['train_loss'], label='train_loss')

# axes[0].plot(pd_results['epoch'],pd_results['test_loss'], label='test_loss')



axes[0].legend()



axes[1].plot(pd_results['epoch'],pd_results['valid_acc'], label='validation_acc')

axes[1].plot(pd_results['epoch'],pd_results['train_acc'], label='train_acc')

# axes[1].plot(pd_results['epoch'],pd_results['test_acc'], label='test_acc')

axes[1].legend()
num_classes = len(classes)



# Use the validation set to make a confusion matrix

net.eval() # good habit I suppose

predictions = torch.LongTensor().to(device) # Tensor for all predictions



# Goes through the val set

for images, _ in val_loader:

    images = images.to(device)

    preds = net(images)

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

    images = images.to(device)

    preds = net(images)

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)
# Read in the sample submission

submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")



# Change the label column to our predictions 

# Have to make sure the predictions Tensor is on the cpu

submission['label'] = predictions.cpu().numpy()

# Write the dataframe to a new csv, not including the index

submission.to_csv("predictions.csv", index=False)
submission.head()