# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import DataLoader
# descargando datos de entrenamiento

train = pd.read_csv(r"../input/digit-recognizer/train.csv",dtype = np.float32)

# se dividen los datos en caracteristicas y etiquetas 

targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization



# picamos el set entranamiento con la finalidad de tener un nuvo set de validacion (80%, 20%) 

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42) 



#creamos funciones y tagets tensor para el set de entrenamiento y el de validacion

#con la finalidad de accumular los gradieantes.

featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long





# vizualizamos una imagen del set

plt.imshow(features_numpy[22].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[22]))

plt.savefig('graph.png')

plt.show()
#Creando modelo CNN

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        

        # Convolution 1

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=2)

        self.drop1=nn.Dropout(.3)

        self.relu1 = nn.ReLU()

        

         # Convolution 2

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2)

        self.relu2 = nn.ReLU()

        

        # Max pool 1

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

     

        # Convolution 3

        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2)

        self.drop2=nn.Dropout(.15)

        self.relu3 = nn.ReLU()

        

        # Convolution 4

        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.relu4 = nn.ReLU()

        

        

        # Max pool 2

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        

         # Convolution 5

        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1)

        self.drop3=nn.Dropout(.1)

        self.relu5 = nn.ReLU()

        

        

        # Max pool 3

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        

        # Convolution 6

        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.relu6 = nn.ReLU()

        

        

        # Fully connected 1

        self.fc1 = nn.Linear(64, 10) 

    

    def forward(self, x):

        # Convolution 1

        out = self.cnn1(x)

        out=self.drop1(out)

        out = self.relu1(out)

        

        # Convolution 2

        out = self.cnn2(out)

        out = self.relu2(out)

        

        # Max pool 1

        out = self.maxpool1(out)

        

        # Convolution 3

        out = self.cnn3(out)

        out = self.drop2(out)

        out = self.relu3(out)

        

        # Convolution 4 

        out = self.cnn4(out)

        out = self.relu4(out)

        

        # Max pool 2 

        out = self.maxpool2(out)

        

        # Convolution 5

        out = self.drop3(out)

        out = self.cnn5(out)

        out = self.relu5(out)

        

        # Max pool 3 

        out = self.maxpool3(out)

        

        # Convolution 6 

        out = self.cnn6(out)

        out = self.relu6(out)



        

        # flatten

        out = out.view(out.size(0), -1)



        # Linear function (readout)

        out = self.fc1(out)

        

        return out



# tamano de lote, cantidadmde iteraciones

batch_size = 120

n_iters = 2000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch set de entrenamiento y validacion

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



# cargador de datos

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)



    

# creando el modelo CNN

model = CNNModel()



# perdida 

error = nn.CrossEntropyLoss()



# Optimizador ADAM

learning_rate = 98e-5

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# modelo entrenamiento

count = 0

loss_list = []

loss_list1 = []

iteration_list = []

accuracy_list = []

accuracy_list1 = []

for epoch in range(num_epochs):

    correcto1=0

    total1= 0

    for i, (images, labels) in enumerate(train_loader):

        

        train = Variable(images.view(120,1,28,28))

        labels = Variable(labels)

        

        # Gradiante

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        # calculando salida del softmax y del loss

        loss = error(outputs, labels)

        # calculando gradiantes

        loss.backward()

        # actualizando parametros

        optimizer.step()

        predicted1 = torch.max(outputs.data, 1)[1]

        total1 += len(labels)

        correcto1 += (predicted1 == labels).sum()

        accuracy1 = 100 * correcto1 / float(total1)

        count += 1

        if count % 10 == 0:

            # Calculando el Accuracy         

            correct = 0

            total = 0

            # iterando en el set de validacion

            for images, labels in test_loader:

                

                test = Variable(images.view(120,1,28,28))

                labels = Variable(labels)

                # Forward propagation

                outputs = model(test)

                loss1 = error(outputs, labels)

                # obteniendo una prediccion

                predicted = torch.max(outputs.data, 1)[1]

                

                # sumando uno al numero de etiquetas

                total += len(labels)

                

                correct += (predicted == labels).sum()

             

            accuracy = 100 * correct / float(total)

            

            # guardando la iteracion de la perdida

            loss_list.append(loss.data)

            loss_list1.append(loss1.data)

            iteration_list.append(count)

            accuracy_list.append(accuracy)

            accuracy_list1.append(accuracy1)

            if count % 20 == 0:

                # imprimiendo

                print('Iteration: {}  Loss: {}  Accuracy train: {} Accuracy test: {} %'.format(count, loss.data, accuracy, accuracy1))

            



# grafica loss 

plt.plot(iteration_list,loss_list1,color = "blue")

plt.plot(iteration_list,loss_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("CNN: Loss test=red train=blue vs Number of iteration")

plt.show()



# gafica accuracy 

plt.plot(iteration_list,accuracy_list,color = "blue")

plt.plot(iteration_list,accuracy_list1,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("CNN: Accuracy test=red train=blue vs Number of iteration")

plt.show()
import torchvision.transforms as transforms

from   PIL import Image

transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize((0.5,), (0.5,))

    ])



# Get the device we're training on

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_digits(df):

    """Loads images as PyTorch tensors"""

    # Load the labels if they exist 

    # (they wont for the testing data)

    labels = []

    start_inx = 0

    if 'label' in df.columns:

        labels = [v for v in df.label.values]

        start_inx = 1

        

    # Load the digit information

    digits = []

    for i in range(df.pixel0.size):

        digit = df.iloc[i].astype(float).values[start_inx:]

        digit = np.reshape(digit, (28,28))

        digit = transform(digit).type('torch.FloatTensor')

        if len(labels) > 0:

            digits.append([digit, labels[i]])

        else:

            digits.append(digit)



    return digits
# Define the test data loader

kag        = pd.read_csv("../input/digit-recognizer/test.csv")

kagg = pd.read_csv(r"../input/digit-recognizer/test.csv",dtype = np.float32)

kaggg = kagg.loc[:,kagg.columns != "label"].values/255 # normalization

test_X      = get_digits(kag)

test_loader = torch.utils.data.DataLoader(test_X, batch_size=batch_size, shuffle = False)

# Create storage objects

ImageId, Label = [],[]



# Loop through the data and get the predictions

for data in test_loader:

    # Move tensors to GPU if CUDA is available

    data = data.to(device)

    # Make the predictions

    output = model(data)

    # Get the most likely predicted digit

    _, pred = torch.max(output, 1)

    

    for i in range(len(pred)):        

        ImageId.append(len(ImageId)+1)

        Label.append(pred[i].cpu().numpy())



sub = pd.DataFrame(data={'ImageId':ImageId, 'Label':Label})
sub.to_csv("submission.csv", index=True)
sub.head(10)
plt.imshow(kaggg[1255].reshape(28,28))

plt.axis("off")

plt.title(str(Label[1255]))

plt.savefig('graph.png')

plt.show()
