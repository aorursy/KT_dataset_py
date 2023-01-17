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
import torch
import numpy as np
import time
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split


train_data = pd.read_csv('../input/digit-recognizer/train.csv', dtype=np.float32)
test_data = pd.read_csv('../input/digit-recognizer/test.csv', dtype=np.float32)

start_time = time.time()
train_Y = train_data['label']
train_X = train_data.drop(labels=['label'], axis=1)

train_X = torch.tensor(train_X.values)
train_Y = torch.tensor(train_Y.values)

train_set = torch.utils.data.TensorDataset(train_X, train_Y)
num_workers = 0
batch_size = 64 # Tamaño del batch
valid_size = 0.2 # Porcentaje de los inputs de entrenamiento, que serán usados para la validación 

# Se obtienen de los datos de entrenamiento, los índices de los datos que se utilizarán para la validación
num_train = len(train_X)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Se construyen las variables de datos, en base a los índices de entrenamiento y validación obtenidos 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                    shuffle=True, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                    shuffle=True, num_workers=num_workers)
import matplotlib.pyplot as plt
%matplotlib inline
    
# Se obtiene un bacth de los datos de entrenamiento 
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
images = images.reshape(64, 1, 28, 28)
print(images.shape)

# Se obtiene una imagen del batch de datos 
img = np.squeeze(images[0])

# Se imprime la imagen obtenida anteriormente
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Arquitectura del Modelo
class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 4, 4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.conv2 = nn.Conv2d(4, 8, 4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        
        self.fc1 = nn.Linear(8*6*6, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(num_features=32)
        self.fc4 = nn.Linear(32, 10)
        
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = x.view(-1, 8*6*6)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn4(x)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.bn5(x)
        x = self.dropout(x)
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
    
    
model = Net() # Inicialización del modelo
print(model) # Visualización del modelo
from torch.autograd import Variable
torch.manual_seed(1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
epochs = 30
training_losses, validation_losses, validation_accuracy_list, training_accuracy_list = [], [], [], []

for epoch in range(epochs):

    training_loss = 0.0
    validation_loss = 0.0
    training_accuracy = 0.0
    validation_accuracy = 0.0
    
    # Entrenamiento del Modelo    
    for images, labels in train_loader:
        
        # Se "limpian" los gradientes que quedaron almacenados en memoria
        optimizer.zero_grad()
        
        images = Variable(images.view(images.shape[0], 1, 28, 28))
        labels = labels.type(torch.LongTensor)
        
        logps = model(images)
        
        # Se calcula el Loss de entrenamiento 
        loss = criterion(logps, labels)
        
        # Se aplica Backward Propagation
        loss.backward()
        
        optimizer.step()
        
        training_loss += loss.item()
        
        # Se calcula el accuracy del entrenamiento
        top_p, top_class = logps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        training_accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    else:
        
        with torch.no_grad():
            
            model.eval()
            for images, labels in valid_loader:
                
                images = Variable(images.view(images.shape[0], 1, 28, 28))
                labels = labels.type(torch.LongTensor)
                
                logps = model(images)
                loss = criterion(logps, labels)
                
                validation_loss += loss.item()
                
                # Se calcula el accuracy con los datos usados para la validación del modelo
                top_p, top_class = logps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                validation_accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        training_loss = training_loss/len(train_loader)
        validation_loss = validation_loss/len(valid_loader)
        training_accuracy = training_accuracy/len(train_loader)
        validation_accuracy = validation_accuracy/len(train_loader)
        
        # Se guardan los datos en listas para posteriormente graficarlos
        training_losses.append(training_loss/len(train_loader))
        validation_losses.append(validation_loss/len(valid_loader))
        training_accuracy_list.append(training_accuracy*100)
        validation_accuracy_list.append(validation_accuracy*100)
        
        model.train()

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, 
        training_loss,
        validation_loss,
        ))
    
    print(f'\tTraining Accuracy: {training_accuracy*100}%')
    print(f'\tValidation Accuracy: {validation_accuracy*100}%')
from matplotlib import pyplot

pyplot.plot(training_losses)
pyplot.plot(validation_losses)
pyplot.title('Train Loss vs Validation Loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Training Loss', 'Validation Loss'], loc='upper right')
pyplot.show()
pyplot.plot(training_accuracy_list)
pyplot.plot(validation_accuracy_list)
pyplot.title('Train Accuracy vs Validation Accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
pyplot.show()
test_X = test_data.loc[:,test_data.columns != "label"].values

test_set = torch.from_numpy(test_X)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                    shuffle=False, num_workers=num_workers)
predictions = []

with torch.no_grad():
    
    model.eval()

    # Test del Modelo    
    for images in test_loader:

        images = Variable(images.view(images.shape[0], 1, 28, 28))
        output = model(images)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        predictions += top_class.numpy().tolist()

results = np.array(predictions).flatten()
print(len(results))
submissions = pd.DataFrame({'ImageId': list(range(1, len(results) + 1)),
                            'Label': results})

submissions.to_csv("my_submissions.csv", index=False, header=True)
print('Tiempo de Ejecución del Notebook (en segundos): ', (time.time() - start_time))