#This kernel aims to investigate the performance of trasfer learning in solving the Dogs vs. Cats image recognition problem

#In the previous kernel we have seen that the amount of data available during training phase is substantial to the performance of model applying to unknown data

#In the following we are going to use pretrained convolutional base from the VGG16 model which is trained with a huge dataset in ImageNet

#and see if this can achieve a better performance than the model in the previous kernel
%xmode Plain

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns;sns.set()
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras import models

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.applications import VGG16

from tensorflow.keras import backend as Back
def result_visualisation(loss,acc,val_loss,val_acc):

    #Result visualisation

    epochs = np.arange(1,len(loss)+1)

    fig,ax = plt.subplots(1,2,figsize=(20,5))

    ax[0].plot(epochs,loss,label='loss')

    ax[0].plot(epochs,val_loss,label='val_loss')

    ax[0].set_title('Loss')

    ax[0].set_xlabel('Epochs')

    ax[0].legend(['loss','val_loss'])

    ax[1].plot(epochs,acc,label='acc')

    ax[1].plot(epochs,val_acc,label='val_acc')

    ax[1].set_title('Accuracy')

    ax[1].set_xlabel('Epochs')

    ax[1].legend(['acc','val_acc'])

    plt.tight_layout()

    plt.show();
#Directories of data

train_dir = '../input/dogs-vs-cats/train/train'

test_dir = '../input/dogs-vs-cats/test1/test1'
#Create labels for the images

filenames = os.listdir(train_dir)

labels = []

for filename in filenames:

    if filename.startswith('dog'):

        labels.append(1)

    else:

        labels.append(0)
#Import the filenames and labels into a dataframe

#Only 5000 images would be used for a shorter training time

df_train = pd.DataFrame({'Filenames':filenames,'Labels':labels})

#Stratified sampling

df_train = df_train.groupby('Labels').apply(lambda x:x.sample(frac=0.2,random_state=100))

#Read data from the directory

X_filenames = df_train['Filenames'].values

y = df_train['Labels'].values

X = [image.load_img(os.path.join(train_dir,filename),target_size=(150,150)) for filename in X_filenames]

X = np.array([image.img_to_array(item) for item in X])
#Standardization of training data

X = X/255.
#Split the data into train and val set

Xtrain,Xval,ytrain,yval = train_test_split(X,y,stratify=y,test_size=0.2)
#Data Augmentation

imggen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,

                            shear_range=0.2,horizontal_flip=True,fill_mode='nearest')

train_gen = imggen.flow(Xtrain,ytrain,batch_size=128)
#Extract convolutional base from VGG16 and construct the network

#As this kernel is applying to a binary classification problem, so we are going to use only the convolutional base from the VGG16 model

#A new fully connected layer will be constructed and trained

#Convolutional base will be freezed and only the newly created fully connected layer will be trained in this phase

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

conv_base.trainable = False

CNN = Sequential()

CNN.add(conv_base)

CNN.add(Flatten())

CNN.add(Dropout(0.5))

CNN.add(Dense(512,activation='relu'))

CNN.add(Dense(1,activation='sigmoid'))
#Define the optimizer and compile the CNN

optimizer = RMSprop(lr=1e-4)

CNN.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
#Examine the structure of the CNN model

CNN.summary()
#Model Training

history1 = CNN.fit_generator(train_gen,steps_per_epoch=80,validation_data=(Xval,yval),epochs=80,verbose=2)
##Extract loss and acc for visualisation

loss1 = history1.history['loss']

acc1 = history1.history['acc']

val_loss1= history1.history['val_loss']

val_acc1 = history1.history['val_acc']
#Result visualisation

result_visualisation(loss1,acc1,val_loss1,val_acc1)
#From the results above, we can see that they are better than those produced by a CNN network trained from the beginning

#The main reason of having such great improvement would be due to the availability of training dataset during the training phase of the convolutional base

#Another interesting observation here is the validation accuracy is better than the training accuracy

#This is mainly due to the existence of the dropout layer, during training phase not all activation nodes are available for prediction, while they are all available during

#the validation phase, so the performance on validation data is better than on training data
#Examine the activation outputs of the convolutional base

layer_names = ['block1_conv2','block3_conv2','block5_conv2']

img = Xtrain[0].reshape(1,150,150,3)

outputs = [conv_base.layers[index].output for index in [2,8,16]]

activation_model = models.Model(inputs=conv_base.input,outputs=outputs)

activation_outputs = activation_model.predict(img)

for i,output in enumerate(activation_outputs):

    plt.figure(i,figsize=(20,20))

    for j in range(1,4):

        plt.subplot(1,4,j)

        plt.imshow(output[0,:,:,j],cmap='viridis')

        plt.title(layer_names[i]+' activation output {}'.format(j))

        plt.grid(False)

        plt.axis('off');
#Visualise the convolutional filters

filter_num = 25

learning_step = 1.

for i,layer_name in enumerate(layer_names):

    plt.figure(i,figsize=(20,20))

    for j in range(filter_num):

        layer_output = conv_base.get_layer(layer_name).output

        loss = Back.mean(layer_output[:,:,:,j])

        grad = Back.gradients(loss,conv_base.input)[0]

        grad = grad/Back.sqrt(Back.mean(Back.square(grad))+1e-5)

        func = Back.function([conv_base.input],[loss,grad])

        input_img = np.random.random((1,150,150,3))*10+128.

        #Using gradient ascent to maximize the loss

        for k in range(30):

            loss_value,grad_value = func([input_img])

            input_img += grad_value*learning_step

        #Standardize the image data

        input_img -= np.mean(input_img)

        input_img /= np.std(input_img)

        input_img += 0.5

        input_img = np.clip(input_img,0,1)

        #Change to RGB format

        input_img *= 255.

        input_img = np.clip(input_img,0,255).astype('uint8')

        #Visualise the filter output

        plt.subplot(5,5,j+1)

        plt.imshow(input_img[0])

        plt.title(layer_name + ' filter {}'.format(j))

        plt.grid(False)

        plt.axis('off')
#From the visualisation of filers observed above, we can see that the deeper the layer, the more abstract and complicated the filter is

#This suggests that the filters in the shallower layers are of more general purpose, such as edge detection, so they are more suitable to be used in transfer learning

#The deeper layers are of less general function, and this suggests that we may be able to train them in order to achieve better results for specific problems
#In the following, we are going to investigate whether we can further improve the performance of this model by fine-tuning part of the convolutional base
#Examine the structure of the convolutional base

conv_base.summary()
#Unfreeze part of the convoluational base

layer_unfreeze = ['block5_conv1','block5_conv2','block5_conv3']

conv_base.trainable = True

for layer in conv_base.layers:

    if layer.name in layer_unfreeze:

        layer.trainable = True

    else:

        layer.trainable = False

CNN.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
CNN.summary()
#Model Training

history2 = CNN.fit_generator(train_gen,steps_per_epoch=80,validation_data=(Xval,yval),epochs=25,verbose=2)
##Extract loss and acc for visualisation

loss2 = history2.history['loss']

acc2 = history2.history['acc']

val_loss2 = history2.history['val_loss']

val_acc2 = history2.history['val_acc']
#Result visualisation

result_visualisation(loss2,acc2,val_loss2,val_acc2)
#From the results above, we can observe that unfreezing part of the convolutional base and fine tuning the weights would improve the performance of  the network