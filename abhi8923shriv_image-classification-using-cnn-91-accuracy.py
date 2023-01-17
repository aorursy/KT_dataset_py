# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Libraries

import matplotlib.pyplot as plt

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.autograd import Variable
### To test whether GPU instance is present in the system of not.

use_cuda = torch.cuda.is_available()

print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

device
image_size = (128,128)



transformations = transforms.Compose(

        [transforms.Resize(list(image_size)),

            transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),

         transforms.Normalize(mean=[0.5], std=[0.5])])
batch_size = 130 



train_set = datasets.ImageFolder('/kaggle/input/cat-dog-dataset/Cat_Dog_data/train', transform = transformations)



# YOUR CODE HERE for the DataLoader

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
# Get input/output from data loader

for (X_train, y_train) in train_loader:

    print('X_train:', X_train.size(), 'type:', X_train.type())

    print('y_train:', y_train.size(), 'type:', y_train.type())

    break

# YOUR CODE HERE for plotting the images

pltsize=1

plt.figure(figsize=(15*pltsize, pltsize))



for i in range(10):

    plt.subplot(1,10,i+1)

    plt.axis('off')

    plt.imshow(X_train[i,:,:,:].numpy().reshape(128,128), cmap="gray")

    plt.title('Class:'+str(y_train[i].numpy()))
class Flatten(torch.nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()



        # Convolutional Layers

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=5, padding=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=5, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=5, padding=2),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=5, padding=2),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2))        

        self.layer5 = nn.Sequential(

            nn.Conv2d(128, 128, kernel_size=5, padding=2),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2))

        

        # Dropout to avoid overfitting

        self.drop_out = nn.Dropout()



        # Fully connected layers

        self.fc1 = nn.Linear(4*4*128, 512)

        self.fc2 = nn.Linear(512, 2)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.layer5(out)



        # Flatten

        out = out.view(out.size(0), -1)



        out = self.drop_out(out)

        out = self.fc1(out)

        out = self.fc2(out)

        return out
# Declaring the loss function and optimizer



model = CNN()

model = model.to(device)

print(model)



#criterion = # YOUR CODE HERE : Explore and declare loss function

# loss_fn = torch.nn.BCELoss()

loss_fn = torch.nn.CrossEntropyLoss()



#optimizer = # YOUR CODE HERE : Explore on optimizer and define with the learning rate

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

learning_rate = 0.001;

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate);
# YOUR CODE HERE. This will take time



# Record loss and accuracy of the train dataset

def train(epoch, log_interval=100):

    for batch_idx, (data, target) in enumerate(train_loader):

        data = Variable(data.float())

        target = Variable(target)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()



        # Track the accuracy

        total = target.size(0)

        _, predicted = torch.max(output.data, 1)

        correct = (predicted == target).sum().item()        



        if batch_idx % log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item(), (correct / total) * 100))
%%time

epochs = 11

lossv, accv = [], []

for epoch in range(1, epochs+1):

    train(epoch)

    # test(lossv, accv)
# Save model

torch.save(model.state_dict(), 'conv_net_model.ckpt')
# Load the model

loaded_model = CNN()

loaded_model.load_state_dict(torch.load('conv_net_model.ckpt'))

loaded_model.eval()
# Testing Evaluation for CNN model



val_set = datasets.ImageFolder('/kaggle/input/cat-dog-dataset/Cat_Dog_data/test',transform = transformations)



# YOUR CODE HERE for the DataLoader

test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=10)
# Get input/output from data loader

for (X_test, y_test) in test_loader:

    print('X_test:', X_test.size(), 'type:', X_test.type())

    print('y_test:', y_test.size(), 'type:', y_test.type())

    break
# YOUR CODE HERE for calculating the accuracy

loaded_model.eval()

correct = 0

total = 0

for images, labels in test_loader:

    images = Variable(images.float())

    # images, labels = images.to(device), labels.to(device)

    outputs = loaded_model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)

    correct += (predicted == labels).sum()

print('Test Accuracy of the model on test images: %.4f %%' % (100 * correct / total))
print(correct, total)
# Classify the model using different Classifiers



# # Loading the model for the entire dataset. This will take more time for loading.

train_Class = torch.utils.data.DataLoader(train_set) 

test_loader1 = torch.utils.data.DataLoader(val_set)

x_train =[]

y_train=[]

x_test=[]

y_test=[]

for data,target in train_Class:

  x_train.append(data.numpy())

  y_train.append(target.numpy())

for data, target in test_loader1:

  x_test.append(data.numpy())

  y_test.append(target.numpy())
import numpy as np

x_train=np.array(x_train)

x_test=np.array(x_test)
x_train.shape

x_train = x_train.reshape(22500,128*128)

x_test = x_test.reshape(2500,128*128)
x_test.shape
import pickle

pickle.dump(x_train, open('x_train.pkl', 'wb'))

pickle.dump(y_train, open('y_train.pkl', 'wb'))

pickle.dump(x_test, open('x_test.pkl', 'wb'))

pickle.dump(y_test, open('y_test.pkl', 'wb'))
# Use different algorithms (atleast 5 algorithms) for predicting the performance



from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score

from skimage.io import imread, imshow

from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

import glob
#function to calculate the accuracy

def accuracy(actual,predicted):

    return accuracy_score(actual,predicted, normalize=True)
# Common method to train/test and print accuracies for both

def train_test(model):

  model.fit(x_train, y_train)

  y_train_predicted = model.predict(x_train)

  y_test_predicted = model.predict(x_test)

  print("Model:", model)

  print("Training accuracy : ", accuracy(y_train,y_train_predicted))

  print("Test accuracy : ", accuracy(y_test,y_test_predicted))
# RandomForest

rf_model = RandomForestClassifier(n_estimators=90, random_state=0)

train_test(rf_model)
# RandomForest

rf_model = RandomForestClassifier(n_estimators=50, random_state=0)

train_test(rf_model)
#DT Classifier

dt_clf = DecisionTreeClassifier(max_depth=10)

train_test(dt_clf)
#DT Classifier

dt_clf = DecisionTreeClassifier(max_depth=12)

train_test(dt_clf)
# MLP Classifer

mlp_cf = MLPClassifier(activation='relu' ,solver='adam' ,hidden_layer_sizes=(5,),max_iter = 100 ,learning_rate = 'constant',learning_rate_init=0.01)

train_test(mlp_cf)
# MLP Classifer

mlp_cf = MLPClassifier(activation='tanh' ,solver='adam' ,hidden_layer_sizes=(5,),max_iter = 100 ,learning_rate = 'constant',learning_rate_init=0.01)

train_test(mlp_cf)
# MLP Classifer

mlp_cf = MLPClassifier(activation='tanh' ,solver='adam' ,hidden_layer_sizes=(5,),max_iter = 50 ,learning_rate = 'constant',learning_rate_init=0.1)

train_test(mlp_cf)
tree = DecisionTreeClassifier()

bag = BaggingClassifier(tree, n_estimators=10, max_samples=0.8,

                        random_state=1)

train_test(bag)
tree = DecisionTreeClassifier()

bag = BaggingClassifier(tree, n_estimators=20, max_samples=0.5,

                        random_state=1)

train_test(bag)