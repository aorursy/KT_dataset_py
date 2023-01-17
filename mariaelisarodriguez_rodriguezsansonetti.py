#Librerías para la implementación de la CNN
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

#Librería para separar train set del validation set
from sklearn.model_selection import train_test_split

#Librería para el uso de gráficas
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#Manejo de data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Random seed
np.random.seed(1)
#Import de la Data

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


#Dimensiones del train y test set
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
#Separación del train set en: train set y validation set (80% - 20%)
x_train, x_validation, y_train, y_validation = train_test_split(train.values[:,1:], train.values[:,0], test_size=0.2)
aux = np.zeros(test.shape)

#Se estandariza la data al dividirlos entre 255 
train_set = torch.utils.data.TensorDataset(torch.from_numpy(x_train.astype(np.float32)/255), torch.from_numpy(y_train))

val_set = torch.utils.data.TensorDataset(torch.from_numpy(x_validation.astype(np.float32)/255), torch.from_numpy(y_validation))

test_set = torch.utils.data.TensorDataset(torch.from_numpy(test.values[:,:].astype(np.float32)/255), torch.from_numpy(aux).type(torch.FloatTensor))

#Definición del batch size
batch_size = 64

#Data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
#Mostrar ejemplo de la data
dataiter = iter(train_loader)
images, labels = dataiter.next()

img = images[0]
img = img.numpy()
img = img.reshape(28,28)


fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        
        #Pooling layer
        self.pool = nn.AvgPool2d((2,2), stride = 2)
        
        #Linear Layers with Batch Normalization
        self.fc1 = nn.Linear(256, 240)
        self.bn3 = nn.BatchNorm1d(240)
        self.fc2 = nn.Linear(240, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 10)
        
        #Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten image
        x = x.view(-1, 1, 28, 28)
        #Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        #Flatten image
        x = x.view(-1, 4*4*16)
        #Linear layers
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim = 1)
        
        return x
def weights_init_xavier_normal(m):
    
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if (classname.find('Linear') != -1) or (classname.find('Conv2d') != -1):
      nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
      m.bias.data.fill_(0)
model = Net()
model.apply(weights_init_xavier_normal)
print(model)
#Loss Function
criterion = nn.CrossEntropyLoss()
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
#number of epochs to train the model
epochs = 60
steps = 0
m_train = len(train_loader)
m_validation = len(val_loader)

train_losses, val_losses, train_acc, val_acc = [],[],[],[]

for epoch in range(epochs):
    print('epoch: ', epoch+1)
    train_loss = 0.0
    val_loss = 0.0
    total_train = 0.0
    correct_train = 0.0
    total_validation = 0.0
    correct_validation = 0.0
    train_accuracy = 0.0
    val_accuracy = 0.0
    
    model.train()
    for images, labels in train_loader:
        
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
        
        _, predicted = torch.max(out, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum()
        
    
    else:
        #Validacion
        with torch.no_grad():
            model.eval()
            for images, labels in val_loader:
                
                out = model(images)
                loss = criterion(out, labels)
                
                val_loss += loss.item()*images.size(0)
                
                _, predicted = torch.max(out, 1)
                total_validation += labels.size(0)
                correct_validation += (predicted == labels).sum()
            
            #Calculo de loss y accuracy
            
            #loss
            train_loss = train_loss / m_train
            val_loss = val_loss / m_validation
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            #accuracy
            train_accuracy = correct_train.item() / total_train
            val_accuracy = correct_validation.item() / total_validation
            
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            
            print('Train loss: ', train_loss)
            print('Validation loss: ', val_loss)
            print('Train accuracy: ', train_accuracy)
            print('Validation accuracy: ', val_accuracy)
            print('------------------------------------')
plt.plot(train_losses)
plt.plot(val_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss','Validation Loss'])
plt.title("Loss")
plt.show()
plt.plot(train_acc)
plt.plot(val_acc)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.title("Accuracy")
plt.show()
dataiter = iter(train_loader)
images, labels = dataiter.next()

img = images[0]
img = img.numpy()
img = img.reshape(28,28)


fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
image_id, label = [],[]

for images,labels in test_loader:
    
    out = model(images)
    labels, pred_test = torch.max(out,1)
    
    for i in range(len(pred_test)):        
        image_id.append(len(image_id)+1)
        label.append(pred_test[i].numpy())
        
final = pd.DataFrame(data={'ImageId':image_id, 'Label':label})
final.describe
print(final.head())
final.to_csv('submission.csv', index=False)
