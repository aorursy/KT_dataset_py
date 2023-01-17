# Imports



import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim



import torchvision.transforms as transforms

from torchvision import datasets



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importación de datos

# Data import

d_train = pd.read_csv("../input/digit-recognizer/train.csv")

d_test = pd.read_csv("../input/digit-recognizer/test.csv")
# Formato actual de d_train

# d_train current format

d_train.head()
# Formato actual de d_test

# d_test current format

d_test.head()



#Nota: estas imágenes no tinen 'label' porque son las que queremos predecir con el modelo!

#Note: these images don't have 'label' because they're the ones we want to predict with the model!
# Transformación de datos a tensor

# Data transformation to tensor



# 1. Conversión de los DataFrame a NumPy arrays utilizando pandas.DataFrame.to_numpy. Las imágenes y los labels se separan

# 1. Convert DataFrame to NumPy arrays using pandas.DataFrame.to_numpy. Images and labels are separated



# Train dataset

x_train = d_train.iloc[:, 1:].to_numpy(dtype = np.float)

x_train = np.reshape(x_train, newshape = (len(x_train), 1, 28, 28))

y_train = d_train['label'].to_numpy()



print(x_train.shape)

print(y_train.shape)



# Test dataset

x_test = d_test.to_numpy(dtype = np.float)

x_test = np.reshape(x_test, newshape = (len(x_test), 1, 28, 28))



print(x_test.shape)
# 2. Normalización de los datos. Se dividirán los valores entre 255 ya que se encuentran entre 0 y 255,

# haciendo que los nuevos valores se encuentre entre 0 y 1.

# 2. Data normalization. Values will be divided by 255 because they're set between 0 and 255,

# making it so the new values are between 0 and 1.



x_train = x_train / 255

x_test = x_test / 255



# Imprimamos algunos valores

# Let's print some values



print(x_train[0])

print(x_test[0])
# 3. Conversión de los arrays a tensor

# 3. Convert array to tensor



# Train data

X_tr = torch.from_numpy(x_train).type(torch.FloatTensor)

Y_tr = torch.from_numpy(y_train)



# Test data

X_te = torch.from_numpy(x_test).type(torch.FloatTensor)

Y_te = torch.from_numpy(np.zeros(x_test.shape))



# Validation

X_train, X_va, Y_train, Y_va = train_test_split(X_tr, Y_tr, test_size = 0.2)



# Sets

train_set = torch.utils.data.TensorDataset(X_train, Y_train)

val_set = torch.utils.data.TensorDataset(X_va, Y_va)

test_set = torch.utils.data.TensorDataset(X_te, Y_te)



trainloader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = False)

valloader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle = False)

testloader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False)
# Ver un ejemplo del dataset MNIST

# Seeing an example from the MNIST dataset

dataiter = iter(trainloader)

images, labels = dataiter.next()

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
# Ahora se planteará la arquitectira del modelo CNN que se encargará de clasificar las imágenes

# Now I'll build the architecture for the CNN model that will clasify the images



# La arquitectura será: 2 capas convolucionales, y se seguirá con 2 capas lineales. (inicialmente intenté con 3 

# capas convolucionales pero no funcionaba el modelo). 

# Se utilizará Batch Normalization, Weight Initialization, Pooling y Dropout.



class NMIST_CNN(nn.Module):

    

    def __init__(self):

        

        super(NMIST_CNN, self).__init__()

        

        # Capa 1: capa convolucional 1, 1 --> 16

        self.cn1 = nn.Conv2d(1, 16, kernel_size = 4, padding = 2)

        nn.init.xavier_normal_(self.cn1.weight)

        nn.init.zeros_(self.cn1.bias)

        self.bn1 = nn.BatchNorm2d(num_features = 16)

        

        # Capa 2: capa convolucional 2, 16 --> 32

        self.cn2 = nn.Conv2d(16, 32, kernel_size = 4)

        nn.init.xavier_normal_(self.cn2.weight)

        nn.init.zeros_(self.cn2.bias)

        self.bn2 = nn.BatchNorm2d(num_features = 32)

        

        # Capa 3: capa lineal 1, 800 --> 200

        self.fc1 = nn.Linear(800, 200)

        nn.init.xavier_normal_(self.fc1.weight)

        nn.init.zeros_(self.fc1.bias)

        self.bn3 = nn.BatchNorm1d(num_features = 200)

        

        # Capa 4: capa lineal 2 (output), 200 --> 10

        self.fc2 = nn.Linear(200, 10)

        self.bn4 = nn.BatchNorm1d(num_features = 10)

        

        # Dropout, utilizaré 0.2 como siempre se hizo en las tareas

        self.dropout = nn.Dropout(p = 0.2)

        

        #Pooling

        self.pooling = nn.MaxPool2d((2,2))

        

        # (parece que me gustan los numeros 2^n)

        

    def forward(self, x):

        

        # Capa 1

        x = F.relu(self.bn1(self.cn1(x)))

        x = self.pooling(x)

        x = self.dropout(x)

        

         # Capa 2

        x = F.relu(self.bn2(self.cn2(x)))

        x = self.pooling(x)

        x = self.dropout(x)

        

        # Flatten tensor

        x = x.view(-1, 800)

        

        # Capa 3

        x = F.relu(self.bn3(self.fc1(x)))

        x = self.dropout(x)

        

        # Capa 4

        x = F.log_softmax(self.bn4(self.fc2(x)), dim = 1)

        

        return x
model = NMIST_CNN()

print(model)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 30



train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []



for e in range(epochs):

    

    tr_loss = 0

    tr_accuracy = 0

    

    model.train()

    

    for images, labels in trainloader:

        

        optimizer.zero_grad()

        

        results = model(images)

        

        loss = criterion(results, labels)

        

        loss.backward()

        

        optimizer.step()

        

        tr_loss += loss.item()

        

        top_p, top_class = results.topk(1, dim=1)

        equals = top_class == labels.view(*top_class.shape)

        tr_accuracy += torch.mean(equals.type(torch.FloatTensor))

        

    else:

        

        te_loss = 0

        te_accuracy = 0

        

        with torch.no_grad():

            

            model.eval()

            

            for images, labels in valloader:

                

                test_results = model(images)

                

                loss2 = criterion(test_results, labels)

                

                te_loss += loss2.item()

                

                top_p, top_class = test_results.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                te_accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        train_losses.append(tr_loss/len(trainloader))

        train_accuracies.append(tr_accuracy.item()/len(trainloader))

        

        test_losses.append(te_loss/len(valloader))

        test_accuracies.append(te_accuracy.item()/len(valloader))

                

        print("Epoch: " + str(e+1))

        print("Train loss: " + str(tr_loss/len(trainloader)))

        print(f'Train accuracy: {tr_accuracy.item()*100/len(trainloader)}%')

        print("Validation loss: " + str(te_loss/len(valloader)))

        print(f'Validation accuracy: {te_accuracy.item()*100/len(valloader)}%')

        print('')
# Graficar Loss

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.plot(train_losses, label='Train Loss')

plt.plot(test_losses, label='Test Loss')

plt.legend()

plt.show()
# Graficar Accuracy

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.plot(train_accuracies, label='Train Accuracy')

plt.plot(test_accuracies, label='Test Accuracy')

plt.legend()

plt.show()
# Correr el modeo para obtener outputs

prediction = []



with torch.no_grad():



    model.eval()

    

    for images, x in testloader:



        predictions = model(images)

        top_p, top_class = predictions.topk(1, dim=1)



        for n in range(len(predictions)):

            prediction.append(top_class[n].item())

        



submit = pd.DataFrame(data={'Label':prediction})

submit.insert(0, 'ImageId', range(1, 1 + len(submit)), True)

submit.head()
# Subir output a Kaggle

submit.to_csv('submission.csv', index = False)

submit.head()