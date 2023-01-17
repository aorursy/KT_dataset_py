#Librerías 
import numpy as np 
import pandas as pd 
import torch
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim

# Utilizada para la división de la data del training set
from sklearn.model_selection import train_test_split

# Librerías para Gráficos
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Random Seed para resultados predecibles
np.random.seed(1)
# Import de Data
training_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")

print('→ Train Shape:', training_data.shape)
print('→ Test Shape:', test_data.shape)        
# Separación de imágenes y sus respectivos labels del Training Set, conversión en Array
dif_x = np.array(training_data.drop('label',axis=1), dtype=np.float)
y = np.array(training_data['label'])
x = np.reshape(dif_x, newshape = (dif_x.shape[0],1,28,28))

# Rearreglo del Test Set en un Array
dif_x_test = np.array(test_data, dtype=np.float)
x_test = np.reshape(dif_x_test, newshape = (dif_x_test.shape[0],1,28,28))

print('→ Train Array Shape:', x.shape)
print('→ Test Array Shape:',x_test.shape)         
# Conversión de las imágenes y las etiquetas en Tensors de Pytorch del Training Set

# Normalización de Datos divididos entre 255
X = torch.from_numpy(x).type(torch.FloatTensor)/255
Y = torch.from_numpy(y)

# Conversión del Test Set a Tensor
X_test = torch.from_numpy(x_test).type(torch.FloatTensor)/255
temp = np.zeros(x_test.shape)
Y_test = torch.from_numpy(temp).type(torch.FloatTensor)

# Separación de los datos en Train y Validation Sets (80% - 20%)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20)

# Creación del Dataset de Train, Valudation y Test
train = torch.utils.data.TensorDataset(X_train, Y_train)
validation = torch.utils.data.TensorDataset(X_val, Y_val)
test = torch.utils.data.TensorDataset(X_test, Y_test)

# Creación de los DataLoaders
train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = False)
val_loader = torch.utils.data.DataLoader(validation, batch_size = 32, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size= 32, shuffle = False)

print('→ Train Length:', len(train))
print('→ Validation Length:', len(validation))
print('→ Test Length:', len(test))     
# Obtención de 1 Batch para graficar ejemplos del Training Set
dataiter = iter(train_loader)
images_train, labels_train = dataiter.next()
img = images_train.numpy()

# Impresión del primer ejemplo

fig = plt.figure(figsize = (25,5)) 
ax = fig.add_subplot(111)
ax.imshow(np.squeeze(img[0]), cmap='gray')
class coolNet(nn.Module):
    def __init__(self):
        super(coolNet, self).__init__()
        
        # Convolutional Layers
        
        # Convolutional Layer 1 (1 a 6 channels) con Xavier Normal Initialization y Batch Normalization (Capa 1 de la red)
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, padding = 2) 
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        self.conv1_bn = nn.BatchNorm2d(6)
        
        # Convolutional Layer 2 (6 a 16 channels) con Batch Normalization (Capa 2 de la red)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # layer2
        self.conv2_bn = nn.BatchNorm2d(16)
        
        #Linear Layers
        
        # Linear Layer 1 (400 a 120) con Batch Normalization (Capa 3 de la red)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc1_bn = nn.BatchNorm1d(120)
        
        # Linear Layer 2 (120 a 84) con Batch Normalization (Capa 4 de la red)
        self.fc2 = nn.Linear(120, 84) 
        self.fc2_bn = nn.BatchNorm1d(84)
        
        # Output Layer 3 (84 a 10) con Batch Normalization (Capa 5 de la red)
        self.fc3 = nn.Linear(84, 10) 
        self.fc3_bn = nn.BatchNorm1d(10)
        
        # MaxPool Layer 2x2
        self.maxPool = nn.MaxPool2d((2,2)) 
        
        # Dropout con 0.2 
        self.dropout = nn.Dropout(p=0.2)
    
    # Forward Propagation
    def forward(self, x):
    
        x = self.maxPool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.maxPool(F.relu(self.conv2_bn(self.conv2(x)))) 
        x = self.dropout(x)
        
        # Flatten Tensor
        x = x.view(-1, 16*5*5)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        
        # Output y Clasificación
        x = F.log_softmax(self.fc3_bn(self.fc3(x)), dim=1)
        
        return x
        
# Definición del Modelo
model = coolNet()
print(model)
# Definición de la Función de Pérdida
criterion = nn.CrossEntropyLoss()

# Definición del Optimizador con learning rate = 0.003
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Número de Iteraciones
epochs = 50

training_loss, validation_loss = [],[]
training_acc, validation_acc = [],[]

for epoch in range(epochs):
    
    # Inicialización de Métricas
    tloss = 0.0
    vloss = 0.0
    total_val = 0
    correct_val = 0
    total_train = 0
    correct_train = 0
    train_accuracy = 0.0
    validation_accuracy =0.0
    
    # ---- Entrenamiento -----
    model.train()
    for images, labels in train_loader:
    
        # Inicializar Gradientes en 0
        optimizer.zero_grad()

        # Obtención de Outputs
        outputs = model(images)
      
        # Cálculo de Pérdida
        loss = criterion(outputs, labels)

        # Backward Pass
        loss.backward()

        # Actualización de Parámetros
        optimizer.step()

        # Actualización de función de pérdida
        tloss += loss.item()*images.size(0)
        
        # Cálculo de Accuracy
        _, predicted_train = torch.max(outputs,1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum()
    
    # ---- Validación ----
    model.eval()
    for images, labels in val_loader:
    
        # Obtención de Outputs
        outputs = model(images)

        # Cálculo de Pérdida
        loss = criterion(outputs, labels)

        # Backward Pass
        loss.backward()

        # Actualización de función de pérdida
        vloss += loss.item()*images.size(0)
        
        # Cálculo de Accuracy del Validation Set
        _, predicted_val = torch.max(outputs,1)
        total_val += labels.size(0)
        correct_val += (predicted_val == labels).sum()
    
    # Métricas del Modelo
    
    # Pérdidas de ambos sets
    tloss = tloss/len(train_loader)
    vloss = vloss/len(val_loader)
    training_loss.append(tloss)
    validation_loss.append(vloss)
    
    #Cálculo del Accuracy del Set
    train_accuracy = correct_train.item() / total_train
    validation_accuracy = correct_val.item() / total_val
    
    training_acc.append(train_accuracy)
    validation_acc.append(validation_accuracy)
    
    
    print('→Epoch: {}/{} \t →Training Loss: {:.6f} \t →Validation Loss: {:.6f} \t →Train Accuracy: {:.6f} \t →Validation Accuracy: {:.6f}'.format(
        epoch+1,epochs,tloss,vloss,train_accuracy,validation_accuracy))
    
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.legend()
plt.ylabel('Pérdidas')
plt.xlabel('Epoch')
plt.title('Comparación de Pérdidas')
plt.show()

plt.plot(training_acc, label='Training Accuracy')
plt.plot(validation_acc, label='Validation Accuracy')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Comparación de Accuracy')
plt.show()
# Obtención de 1 Batch para graficar ejemplos del Test Set

dataitertest = iter(test_loader)
images_test,labels_test = dataitertest.next()
imgtest = images_test.numpy()

# Impresión del primer ejemplo

fig = plt.figure(figsize = (25,5)) 
ax = fig.add_subplot(111)
ax.imshow(np.squeeze(imgtest[0]), cmap='gray')
# Objetos para almacenar las predicciones
ImageId, Label = [],[]

# ---- Predicciones ----
for images,_ in test_loader:
    
    outputs = model(images)
    _, predicted_test = torch.max(outputs,1)
    
    for i in range(len(predicted_test)):        
        ImageId.append(len(ImageId)+1)
        Label.append(predicted_test[i].numpy())
    
sub = pd.DataFrame(data={'ImageId':ImageId, 'Label':Label})
sub.describe
# Escritura del archivo CSV
sub.to_csv("submission.csv", index=False)
sub.head()