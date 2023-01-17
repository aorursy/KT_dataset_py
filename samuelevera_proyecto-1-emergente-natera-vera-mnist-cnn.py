# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import matplotlib.pyplot as plt # Plotting
import pandas as pd #Reading csv
from sklearn.model_selection import train_test_split

#Pytorch
import torch
from torch.autograd import Variable
from torch import nn,functional
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print(device)
#file paths
train_csv_path = '../input/digit-recognizer/train.csv'
test_csv_path = '../input/digit-recognizer/test.csv'

#Read the csv and make the dataframes
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

#Print dataframe structure
n_train = len(train_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))
print('M = {0}'.format(n_train))
print('WxH = {0}'.format(n_pixels))
print('# classes = {0}'.format(n_class))

#Print the test dataframe structure
n_test = len(test_df)
n_pixels = len(test_df.columns)
print('Number of test samples = {0}'.format(n_test))
print('Test WxH = {0}'.format(n_pixels))
#Build custom data set to allow data augmentation
class MNISTDataset(Dataset):
    """MNIST data set"""
    
    def __init__(self, dataframe, 
                 transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        df = dataframe
        #28x28 images
        self.n_pixels = 784
        
        if len(df.columns) == self.n_pixels:
            #Test data
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = None
        else:
            #Training data
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])
#Helper to build a dataset with a dataframe, a class dataset and composed transformations
def get_dataset(dataframe, dataset=MNISTDataset,
                transform=transforms.Compose([transforms.ToPILImage(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.5,), std=(0.5,))])):
    return dataset(dataframe, transform=transform)

#Split the data frmae in 2 (train - dev)
def split_dataframe(dataframe=None, fraction=0.9, rand_seed=1):
    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)
    df_2 = dataframe.drop(df_1.index)
    return df_1, df_2
batch_size = 128

#Train transformations include data augmentation
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)

#Dev and transformations only have batch normalization
dev_test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)

#Split train and dev dataframes
train_df_new, val_df = split_dataframe(dataframe=train_df, fraction=0.8)

#Create train data set
train_dataset = get_dataset(train_df_new, transform=train_transforms)
#Create dev data set
dev_dataset = get_dataset(val_df, transform=dev_test_transforms)

#Create train loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#Create dev loader
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)
#For plotting
print('m =',len(train_loader)*batch_size)
print('m_dev =',len(dev_loader)*batch_size)

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

print(images.shape) # Should be (batch_size, 28x28)

#Plot 10 in a row to do a sanity check over the data augmentation and train data
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(10): #Arange next 10
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[]) #Create subplot
    ax.imshow(np.squeeze(images[idx].reshape(28,28)), cmap='gray') #Show subplot
    ax.set_title(labels[idx]) #Set title of subplot
import torch.nn as nn
import torch.nn.functional as F

class ClassifierNetwork(nn.Module):
    def __init__(self, drop_rate = 0.0):
        super(ClassifierNetwork, self).__init__()
        ## encoder layers ##
        #Convolution 1st layer (Same convolution)
        #Inputs 28x28
        self.conv1 = nn.Conv2d(
            1, #In channels 
            64, #Out channels (amount of filters)
            3, #Filter size
            padding = 1 #Padding amount
        )
        self.bn1 = nn.BatchNorm2d(64)
        #Convolution 2nd layer (Same padding)
        #Inputs 14x14
        self.conv2 = nn.Conv2d(
            64, #In channels
            64, #Out channels
            3, #Filter size
            padding = 1 #Padding amount
        )
        self.bn2 = nn.BatchNorm2d(64)
        #Convolution 3rd layer (Same padding)
        #Inputs 7x7
        self.conv3 = nn.Conv2d(
            64, #In channels
            128, #Out channels
            3, #Filter size
            padding = 1 #Padding amount
        )
        self.bn3 = nn.BatchNorm2d(128)
        #Max pool
        #Reduces size to half
        self.pool = nn.MaxPool2d(
            2, #Filter size
            stride = 2 #Stride to reduce 
        )
        #Linear layers
        #3*3*64 (576 -> 256)
        self.fc4 = nn.Linear(3*3*128, 512)
        self.bn4 = nn.BatchNorm1d(512)
        #256 -> 10
        self.output = nn.Linear(512, 10)
        #Dropout
        self.dropout = nn.Dropout(p = drop_rate)
        #Convolution dropout
        self.dropout_conv = nn.Dropout2d(p = drop_rate)
    
    #Forward pass
    def forward(self, x):
        #Transform to fit the convolutional layers
        x = x.view(-1,1,28,28)
        # Convolutional layers
        x = self.pool(self.bn1(F.relu(self.conv1(x)))) #Convolution then pool
        x = self.dropout_conv(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x)))) #Convolution then pool
        x = self.dropout_conv(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x)))) #Convolution then pool
        x = self.dropout_conv(x)
        #Flatten for hidden layers
        x = x.view(-1, 3*3*128)
        #Hidden layers
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = F.log_softmax(self.output(x), dim=1)
        return x
selected_l2_regularization = 0
dropout_selected = 0
selected_learning_rate = 0.001
reduce_lr_every_epochs = 1
selected_lr_decay = 0.95
epochs = 35
#Models List
models = []

#Instanciate the first model
model = ClassifierNetwork(drop_rate = dropout_selected)
#Instanciate the optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr = selected_learning_rate, weight_decay = selected_l2_regularization)
#Instanciate the loss
criterion = nn.CrossEntropyLoss()
#Reduce in epoch
reduce_every = reduce_lr_every_epochs
#Model label
label = 'Modelo final (iteración 2)'
gamma = selected_lr_decay
#Save in the models list
models.append(( model, optimizer, criterion, label, reduce_every, gamma ))
#Print
print('Label', label)
print('Model structure:', model)
print('Optimizer', optimizer)
print('Criterion', criterion)
print('Learning rate decay every', reduce_every, 'epoch')
print('Learning rate decay gamma', gamma)
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')

#To plot all losses
all_models_losses = []
#Itearte over the models list
for model, optimizer, criterion, label, reduce_every, gamma in models:
    print('Model', label, 'started')
    #Use GPU
    model = model.float()
    model = model.to(device)
    criterion = criterion.to(device)
    #Model accuracies
    model_losses = []
    train_accs = []
    dev_accs = []
    
    #Build scheduler for learning rate decay
    if reduce_every != 0:
        #Exponential adaptative learning rate with gamma = 0.95
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
        print('Scheduler', scheduler)
    
    #Iterate the epochs
    for epoch in range(epochs):
        print('Epoch #'+str(epoch)+' started')
        #Track accuracies
        correct_train = 0
        total_train = 0
        #Track loss
        epoch_loss = 0
        #Set model mode to training
        model.train()
        #Iterate over the mini batchs
        for images, labels in train_loader:
            #Use GPU
            images_var = images.to(device)
            labels_var = labels.to(device)
            #Preven gradient accumulations
            optimizer.zero_grad()
            #Forward propagation
            outputs = model(images_var)
            #Calculate 
            loss = criterion(outputs, labels_var.long())
            #Backward propagation
            loss.backward()
            #Update parameters
            optimizer.step()
            
            #Keep track of the minibatch loss
            epoch_loss += loss.item()
            
            #Get the predicted class for each output
            _, predicted = torch.max(outputs.data, 1)
            #Count up the total amount of correct labels
            #for which the predicted and true labels are equal
            correct_train += (predicted == labels_var).sum()
            #Track total examples
            total_train += labels.size(0)
        #When mini batches iteration ends
        else:
            #Set evaluation mode
            model.eval()
            #Prevent grads
            with torch.no_grad():
                #Track accuracies
                correct_dev = 0
                total_dev = 0
                #Validate dev dataset
                for images, labels in dev_loader:
                    #Use GPU
                    images_var = images.to(device)
                    labels_var = labels.to(device)
                    #Get output
                    outputs = model(images_var)
                    #Get the predicted class for each output
                    _, predicted = torch.max(outputs.data, 1)
                    #Count up the total amount of correct labels
                    #for which the predicted and true labels are equal
                    correct_dev += (predicted == labels_var).sum()
                    #Track total examples
                    total_dev += labels.size(0)
                #Store epoch losses
                model_losses.append(epoch_loss/len(train_loader))
                #Store epoch accuracy
                train_accs.append(correct_train.item()/total_train)
                dev_accs.append(correct_dev.item()/total_dev)
                print('Epoch #'+str(epoch)+' ended')
                print('Epoch loss: '+str((epoch_loss/len(train_loader))))
                print('- Train accuracy: '+str((correct_train.item()/total_train)*100)+'%')
                print('- Dev accuracy: '+str((correct_dev.item()/total_dev)*100)+'%')
                print('-------------------------------------------------------------------------------')
        #Check if the model is one that reduces the learning rate
        if reduce_every != 0:
            #Check if the learning rate should actually be reduced
            if (epoch+1) % reduce_every == 0:
                scheduler.step()
                print('Learning rate reduced')
    
    #Plot here
    plt.plot(np.squeeze(train_accs), 'blue', label = label + ' - Train accuracy')
    plt.plot(np.squeeze(dev_accs), 'red', label = label + ' - Dev accuracy')
    plt.title('Comparación de precisiones para el modelo ' + label)
    plt.xlabel('Epochs')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()
    
    #Store models losses
    all_models_losses.append(( model_losses, label ))
    
    print('Model', label, 'ended')
    print('--------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------')
#Plot model losses
for losses, label in all_models_losses:
    plt.plot(np.squeeze(losses), label = label)

#Customize the plot
plt.title('Losses del modelo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#Get the model
model, _, _2, _3, _4, _5 = models[0]
print(model)
#Set the evalutaion mode
model.eval()
#Predictions
test_pred = torch.LongTensor()

#Create the test dataset with only normalization transformation
test_dataset = get_dataset(test_df, transform=dev_test_transforms)

#Create the test loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Iterate the test data
for i, data in enumerate(test_loader):
    #Check GPU
    if torch.cuda.is_available():
        data = data.cuda()

    #Outputs
    output = model(data)

    #Get predictions
    pred = output.cpu().data.max(1, keepdim=True)[1]
    test_pred = torch.cat((test_pred, pred), dim=0)

# tensor -> numpy.ndarray -> pandas.DataFrame
test_pred_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1), test_pred.numpy()], columns=['ImageId', 'Label'])

# show part of prediction dataframe
print(test_pred_df.head())

#Upload predictions
test_pred_df.to_csv('submission.csv', index=False)