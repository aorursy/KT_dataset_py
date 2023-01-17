import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import random



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, LSTM

from keras.layers import Conv2D

from keras.utils import plot_model



import torch

from torch import nn,functional

from torch.autograd import Variable

import torchvision.models as models

from torchvision.models.resnet import  model_urls





from fastai import *

from fastai.vision import *





from numpy.random import seed

seed(48)

from tensorflow import set_random_seed

set_random_seed(438)



from sklearn.model_selection import train_test_split



from IPython.display import Image
def generate_route(route_no):

        if route_no == 0:

            nine_route_x,nine_route_y,direction = 0*np.arange(40)+random.random()*36+8, 0.5*np.arange(40)-2*random.random(),270*np.ones(40)

            return np.vstack((nine_route_x,nine_route_y,direction))



        elif route_no == 1:

            eight_route_x,eight_route_y,direction = np.hstack((0*np.arange(20),0.25*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0.3*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),220*np.ones(20)))

            

            if(eight_route_x[0] > 25):

                eight_route_x = eight_route_x - 2* (eight_route_x - eight_route_x[0])   

                direction[20:] += 100

            return np.vstack((eight_route_x,eight_route_y,direction))



        elif route_no == 2:

            seven_route_x,seven_route_y,direction = np.hstack((0*np.arange(20),-0.25*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0.3*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),320*np.ones(20)))

            if(seven_route_x[0] > 25):

                seven_route_x += 2* (seven_route_x[0] - seven_route_x)     

                direction[20:] -= 100

            return np.vstack((seven_route_x,seven_route_y,direction))



        elif route_no == 3:

            six_route_x,six_route_y,direction = np.hstack((0*np.arange(20),0.3*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),180*np.ones(20)))

            

            if(six_route_x[0] > 25):

                six_route_x = six_route_x - 2* (six_route_x - six_route_x[0])

                direction[20:] -= 180

            return np.vstack((six_route_x,six_route_y,direction))



        elif route_no == 4:

            five_route_x,five_route_y,direction = np.hstack((0*np.arange(20),-0.3*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),0*np.ones(20)))

            if(five_route_x[0] > 25):

                five_route_x += 2* (five_route_x[0] - five_route_x)

                direction[20:] += 180

            return np.vstack((five_route_x,five_route_y,direction))

num_play = 200



route_coordinate = []

play_type = []



for play in range(num_play):

    temp_route = []

    play_type_int = random.randrange(3)

    play_type.append(play_type_int)

    

    for i in range(3): #Generate 3 routes with same play type

        if(play_type_int == 0):

            route_type = 0

        elif(play_type_int == 1):

            route_type = np.round(random.random()+1)

        else:

            route_type = np.round(random.random()+3)



        temp_route.append(generate_route(route_type))

    for i in range(2): #Generate 2 random routes

        temp_route.append(generate_route(random.randrange(5)))

    #print(len(temp_route))

    route_coordinate.append(temp_route)

# route_coordinate = np.array(route_coordinate).reshape(200,5,40,3)
x_rescale = np.array(route_coordinate)[:,:,0]/50

y_rescale = (np.array(route_coordinate)[:,:,1]- np.array(route_coordinate)[:,:,1].min())/( np.array(route_coordinate)[:,:,1].max()- np.array(route_coordinate)[:,:,1].min())

direction_rescale = np.array(route_coordinate)[:,:,2]
route_data = np.stack((x_rescale, y_rescale, direction_rescale),axis=3)
plt.scatter(route_data[6,:,:,0],route_data[6,:,:,1])
plt.imshow(route_data[6])
X_train, X_test, y_train, y_test = train_test_split(route_data,play_type,test_size=0.2,random_state=1583)
y_train_transform = keras.utils.to_categorical(y_train,3)

y_test_transform = keras.utils.to_categorical(y_test,3)
num_classes = 3

model = keras.Sequential()

model.add(Conv2D(50, kernel_size=(1,5),

                 activation='relu',

                 input_shape=(5,40,3)))

model.add(Conv2D(100, (5, 5), activation='relu'))



model.add(Dropout(0.25))

model.add(keras.layers.Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
model.fit(X_train, y_train_transform,

          batch_size=64,

          epochs=200,

          verbose=1,

          validation_data=(X_test, y_test_transform))

score = model.evaluate(X_test, y_test_transform, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        self.conv_block = nn.Sequential(

            nn.Conv2d(3, 50, kernel_size=(1,5), stride=1, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(50, 100, kernel_size=(5,5), stride=1, padding=1),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.ReLU(inplace=True))



        self.linear_block = nn.Sequential(

            nn.Linear(100*180, 128),

            nn.Dropout(0.5),

            nn.ReLU(inplace=True),

            nn.Linear(128, 3)

        )

        

    def forward(self, x):

        x = self.conv_block(x)

        x = x.view(x.size(0), -1)

        x = self.linear_block(x)

        return x

model = CNN()

model = model.double()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
train = torch.utils.data.TensorDataset(torch.from_numpy(X_train.reshape(-1,3,5,40)),torch.LongTensor(y_train))

test = torch.utils.data.TensorDataset(torch.from_numpy(X_test.reshape(-1,3,5,40)),torch.LongTensor(y_test))





train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = 16, shuffle = False)
epochs = 100

train_losses, test_losses = [] ,[]

for epoch in range(epochs):

    running_loss = 0

    for images,labels in train_loader:





        labels = Variable(labels)

        

        optimizer.zero_grad()

        

        output = model(images)

        loss = criterion(output,labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    else:

        test_loss = 0

        accuracy = 0

        

        with torch.no_grad(): #Turning off gradients to speed up

            model.eval()

            for images,labels in test_loader:

                

                test = Variable(images.view(-1,3,5,40))

                labels = Variable(labels)

                

                log_ps = model(test)

                test_loss += criterion(log_ps,labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim = 1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()        

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))



        print("Epoch: {}/{}.. ".format(epoch+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(objective='multiclass', random_state=5)



lgbm.fit(X_train.reshape(-1,600), y_train)



y_pred = lgbm.predict(X_test.reshape(-1,600))
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)