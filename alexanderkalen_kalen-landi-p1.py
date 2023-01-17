import numpy as np

import matplotlib.pyplot as plt

import torch

import time



print(f'Using PyTorch v{torch.__version__}')





t0= time.clock()
#Cargamos la data para Training



import pandas as pd

train_data = pd.read_csv("../input/digit-recognizer/train.csv")
#Cargamos el device donde se va a trabajar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#Se construye la Transformacion usando TorchVision



import torchvision.transforms as transforms

from   PIL import Image

transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize((0.5,), (0.5,)) #---> Hiperparametro por defecto de MNIST

    ])



#Con esta funcion extraemos los valores y labels del dataset en el pandas dataframe



def obtener_valores(df):

   

    labels = []

    index = 0

    if 'label' in df.columns: # Se recorren las columnas del archivo de entrada para obtener los labels

        labels = [v for v in df.label.values]

        index = 1

        

    # Se recorren los diferentes valores del archivo de entrada para transformalos en data usable para la rn

    datas = []

    for i in range(df.pixel0.size):

        data = df.iloc[i].astype(float).values[index:] 

        data = np.reshape(data, (28,28)) #--> Aseguramos que este correcto el valor de la matriz de valores

        data = transform(data).type('torch.FloatTensor') #--> Transformamos la data

        if len(labels) > 0:

            datas.append([data, labels[i]])

        else:

            datas.append(data)



    return datas
batch_size  = 64   

validation  = 0.15

num_classes = 10



#Cargamos la data



X_train = obtener_valores(train_data)



#Dividimos la data (split) entre el el training y el validation(test)

m = len(X_train)

indices   = list(range(m))

split = int(np.floor(validation* m))  

index_train, index_test = indices[split:], indices[:split] #---> Split



# Define los samplers para obtener los batches de el train y validation loader

from torch.utils.data.sampler import SubsetRandomSampler

train_sampler = SubsetRandomSampler(index_train)

valid_sampler = SubsetRandomSampler(index_test)



# Construimos los data loaders

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,

                    sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, 

                    sampler=valid_sampler)
import torch.nn as nn



#Calculamos el Tamano del output 'nxn' con stride:



def stride_tam(in_layers, stride, padding, tam_kernel, tam_pool):

    return int((1 + (in_layers - tam_kernel + (2 * padding)) / stride) / tam_pool)



class Net(nn.Module): 

    

    

    def __init__(self):

        super(Net, self).__init__()

        

        #Pasandole tuplas correspondientes a los canales de entrada, tamaño del kernel y pool

        #definimos diferentes valores para nuestra arquitectura que nos permiten construir modelos para

        #nuestra arquitectura.



        #### ARQUITECTURA ####

        

        inputs     = [1,16,32,64]

        tam_kernel = [5,5,3]

        tam_pool   = [2,2,2]

        

        #####################



        layers_conv = []

        self.out   = 28         #--> inicializamos en 28 para caclular el output layer de cada layer

        self.depth = inputs[-1] #--> Ultimo layer 

        

        for i in range(len(tam_kernel)):

            

            padding = int(tam_kernel[i]/2)



            # Output para cada layer

            self.out = stride_tam(self.out, 1, padding, tam_kernel[i], tam_pool[i])

            

            #Utilizamos 1x1 Convolution Arquitecture Tecnique:



            # Convolutional layer 1

            layers_conv.append(nn.Conv2d(inputs[i], inputs[i+1], tam_kernel[i], 

                                       1, padding=padding))

            layers_conv.append(nn.BatchNorm2d(inputs[i+1])) #--> BathNorm

            layers_conv.append(nn.ReLU())

            

            

            # maxpool layer

            layers_conv.append(nn.MaxPool2d(tam_pool[i],tam_pool[i]))



        self.convolutional_layers = nn.Sequential(*layers_conv)

                

        # Fully connected Layers

        fc_layers = []

        fc_layers.append(nn.Dropout(p=0.2))

        fc_layers.append(nn.Linear(self.depth*self.out*self.out, 256))

        fc_layers.append(nn.Dropout(p=0.2))

        fc_layers.append(nn.Linear(256, 128))

        fc_layers.append(nn.Dropout(p=0.2))

        fc_layers.append(nn.Linear(128, 64))

        fc_layers.append(nn.Dropout(p=0.2))

        fc_layers.append(nn.Linear(64, 10))



        self.fully_connected_layers = nn.Sequential(*fc_layers)



    def forward(self, x):

        

        x = self.convolutional_layers(x)

        

        #Aplanamos

        x = x.view(-1, self.depth*self.out*self.out)

        

        x = self.fully_connected_layers(x)

        return x

    



model = Net()

model
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr= 0.00069) #--> Hiperparametro
#Nota --> Revisar que diga cuda al correr esta celda (GPU)



epochs = 20



valid_loss_min = np.Inf



print(device)

model.to(device)

tLoss, vLoss = [], []

for epoch in range(epochs):



    train_loss = 0.0

    valid_loss = 0.0

    

#Training

    

    model.train()

    for data, target in train_loader:

        

        # Utilizamos GPU para agilizar el entrenamiento

        data   = data.to(device)

        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*data.size(0)

        

#Validamos



    model.eval()

    for data, target in valid_loader:

       

        # Utilizamos GPU para agilizar el entrenamiento

        data   = data.to(device)

        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        valid_loss += loss.item()*data.size(0)

    

    

    train_loss = train_loss/len(train_loader.dataset)

    valid_loss = valid_loss/len(valid_loader.dataset)

    tLoss.append(train_loss)

    vLoss.append(valid_loss)

        

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch + 1, train_loss, valid_loss))

    

    # Salvamos el modelo con el Validation Loss mas bajo (Early Stopping)

    if valid_loss <= valid_loss_min:

        print('Mejora ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'modelo_optimo.pt')

        valid_loss_min = valid_loss
#Loss

plt.plot(tLoss, label='Training Loss')

plt.plot(vLoss, label='Validation Loss')

plt.legend();



model.load_state_dict(torch.load('modelo_optimo.pt'));



test_loss  = 0.0

Y_hat      = [0]*10

Y          = [0]*10



model.eval()



for data, target in valid_loader:



    data   = data.to(device)

    target = target.to(device)

    output = model(data)

    loss = criterion(output, target)

    test_loss += loss.item()*data.size(0)

    _, pred = torch.max(output, 1)    

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())



    for i in range(target.size(0)):

        label = target.data[i]

        Y_hat[label] += correct[i].item()

        Y[label] += 1



test_loss = test_loss/len(valid_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



print('Test Accuracy por Digito:')



for i in range(10):

    if Y[i] > 0:

        print('Test Accuracy of %3s: %.2f%% (%2d/%2d)' % (

            i, 100 * Y_hat[i] / Y[i],

            np.sum(Y_hat[i]), np.sum(Y[i])))

    else:

        print('Test Accuracy of %3s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %.4f%% (%2d/%2d)' % (

    100. * np.sum(Y_hat) / np.sum(Y),

    np.sum(Y_hat), np.sum(Y)))
test_data        = pd.read_csv("../input/digit-recognizer/test.csv")

X_test           = obtener_valores(test_data)

test_loader      = torch.utils.data.DataLoader(X_test, batch_size=batch_size)
t1 = time.clock() - t0



ImageId, Label = [],[]



for data in test_loader:

    data = data.to(device)

    salida = model(data)

    _, pred = torch.max(salida, 1)

    

    for i in range(len(pred)):        

        ImageId.append(len(ImageId)+1)

        Label.append(pred[i].cpu().numpy())



sub = pd.DataFrame(data={'ImageId':ImageId, 'Label':Label})

sub.describe



print("Tiempo pasado: ", t1)



#Hacemos submit

sub.to_csv("submission.csv", index=False)