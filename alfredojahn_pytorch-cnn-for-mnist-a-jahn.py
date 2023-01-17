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
#Python Imports

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os



#PyTorch Util

import torch

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset

from torchvision.utils import make_grid



#PyTorch NN

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
#Convertir DataFrame a Numpy Array

train_labels = train_df['label'].values

train_images = (train_df.iloc[:,1:].values).astype('float32')

test_images = (test_df.iloc[:,:].values).astype('float32')



#Split del train y validation set (85/15 %)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,

                                                                     stratify=train_labels,

                                                                     test_size=0.15)
print("Tamaño Train Set: \t",len(train_images))

print("Tamaño Validation Set: \t",len(val_images))

print("Tamaño Test Set: \t",len(test_images))
#Reshape de los Array de Numpy (M, nChan, H, W)



#Reshape a (35700, 1, 28,28)

train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)

#Reshape a (6300, 1, 28,28)

val_images = val_images.reshape(val_images.shape[0], 1, 28, 28) 

#Reshape a (28000, 1, 28,28)

test_images = test_images.reshape(test_images.shape[0], 1, 28, 28) 
#Convertir de Numpy Array a Tensors y Tensor Datasets



#Pasar a Tensor y Normalizar

train_images_tensor = torch.tensor(train_images)/255.0 #Normalizar

train_labels_tensor = torch.tensor(train_labels)

#Train Tensor Dataset

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)



#Pasar a Tensor y Normalizar

val_images_tensor = torch.tensor(val_images)/255.0 #Normalizar

val_labels_tensor = torch.tensor(val_labels)

#Test Tensor Dataset

val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)



#Pasar a Tensor y Normalizar

test_images_tensor = torch.tensor(test_images)/255.0 #Normalizar



#Hiperparametros

batch_size = 128

learning_rate = 0.0005

dropout_prob = 0.5

weight_decay = 1e-5

epochs = 15





train_loader = DataLoader(train_tensor, batch_size=batch_size, num_workers=2, shuffle=True)

val_loader = DataLoader(val_tensor, batch_size=batch_size, num_workers=2, shuffle=True)

test_loader = DataLoader(test_images_tensor, batch_size=batch_size, num_workers=2, shuffle=False)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        #Conv Layer 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)

        

        #Conv Layer 2

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)

        self.bn2 = nn.BatchNorm1d(64*4*4)

        

        #Fully Connected 1

        self.fc1 = nn.Linear(64*4*4, 128)

        

        #Fully Connected 2

        self.fc2 = nn.Linear(128, 64)

        

        #Fully Connected 3 Output

        self.fc3 = nn.Linear(64, 10)

        

        #Dropout

        self.dropout = nn.Dropout(p=dropout_prob)



    def forward(self, x):

        

        #Conv 1

        x = self.pool(F.relu(self.conv1(x)))

        x = self.bn1(x)

        #Conv 2

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 4 * 4) #Flatten Image a dim de fc1

        x = self.bn2(x)

        #Fully Connected 1

        x = self.dropout(F.relu(self.fc1(x)))

        #Fully Connected 2

        x = self.dropout(F.relu(self.fc2(x)))

        #Output

        x = self.fc3(x)

        return x



#Inicializacion Xavier a los Weights

def xavier_init(m):

    if type(m) == nn.Linear:

        torch.nn.init.xavier_uniform_(m.weight)

        m.bias.data.fill_(0.01)

        

#Instanciar Modelo

net=Net()

net.apply(xavier_init)

# Loss Function

criterion = nn.CrossEntropyLoss()

# Optimizador Adam

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

net
#Listas de losses por epoch

train_losses, val_losses = [], []

for e in range(epochs):

    train_loss = 0

    for images, labels in train_loader:



        optimizer.zero_grad()

        #forward prop

        preds = net(images)

        loss = criterion(preds, labels)

        #back prop

        loss.backward()

        optimizer.step()



        train_loss += loss.item()

            

    else:

        #Pase de validacion/evaluacion        

        val_loss = 0

        correct = 0

        total = 0

        with torch.no_grad():

            net.eval()

            for images, labels in val_loader:

                #Calcular validation loss

                preds = net(images)

                val_loss += criterion(preds, labels)

                #Calcular Accuracy

                _, predicted = torch.max(preds.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

    

    net.train()

    #Agregar a listas por epoch

    train_losses.append(train_loss/len(train_loader))

    val_losses.append(val_loss/len(val_loader))



    print("Epoch {}/{}".format(e+1,epochs),

          "Training Loss: {:.4f}".format(train_loss/len(train_loader)),

          "Validation Loss: {:.4f}".format(val_loss/len(val_loader)),

          "Accuracy: {:.4f}".format(correct/total))





#Plot Train vs Val Loss

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

plt.plot(train_losses, label='Training loss')

plt.plot(val_losses, label='Validation loss')

plt.legend(frameon=False)
# Calcular predicciones en el test_loader

# y agregarlas en el tensor test_preds



def predictions(data_loader):

    net.eval()

    test_preds = torch.LongTensor()

    

    for i, data in enumerate(data_loader):

            

        output = net(data)

        preds = output.data.max(1, keepdim=True)[1]

        test_preds = torch.cat((test_preds, preds), dim=0)

        

    return test_preds



ts_predictions = predictions(test_loader)



submission_df = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")



submission_df['Label'] = ts_predictions.numpy().squeeze()



submission_df.to_csv('submission.csv', index=False)