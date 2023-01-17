import pandas as pd

import numpy as np

import tensorflow as tf
#Training Dataset

train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print("Training Dataset Shape:",train.shape)

train.head()
#Testing Dataset

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print('Test Dataset Shape:',test.shape)

test.head()
#Extracting the dependent and independent variables

y=train['label'].values #To be predicted

x=train.drop('label',axis=1).values #Independent Variables

print('Variables Shape:',x.shape)

print('Labels Shape:',y.shape)
#Displaying some images from the dataset

import matplotlib.pyplot as plt

x=x.reshape(x.shape[0],28,28)

fig=plt.figure(figsize=(12,5))

for i in range(30):

    plt.subplot(3,10,i+1)

    plt.axis('off')

    plt.imshow(x[i],cmap=plt.cm.binary)
#Reshaping and Normalization

from keras.utils.np_utils import to_categorical



#Reshape images to input into CNN layers as a 4D Tensor

x=x.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)

#Normalization

x=x/255

test=test/255

#One Hot Encoding the labels

y=to_categorical(y)
#Data Augmentation to increase number of input images

from keras.preprocessing.image import ImageDataGenerator



train_datagen=ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    fill_mode='nearest'

)
#CNN Ensemble Models

#To learn about deep learning ensemble models:

# https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/

from keras.models import Sequential

from keras.layers import Conv2D,Dropout,Dense,GlobalAveragePooling2D,MaxPool2D,Flatten,BatchNormalization



ensem=10

model=[0]*ensem

for i in range(ensem):

    model[i]=Sequential()

    model[i].add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(28,28,1)))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(MaxPool2D(2,2))

    model[i].add(Dropout(0.2))



    model[i].add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(MaxPool2D(2,2))

    model[i].add(Dropout(0.2))



    model[i].add(GlobalAveragePooling2D())

    model[i].add(Dense(128,activation='relu'))

    model[i].add(Dropout(0.2))

    model[i].add(Dense(10,activation='softmax'))

    

    model[i].compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
import tensorflow as tf

from sklearn.model_selection import train_test_split



callback=tf.keras.callbacks.EarlyStopping(monitor='accuracy',min_delta=0,patience=5,mode='auto',restore_best_weights=True,verbose=0)

lrs=tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)



history=[0]*ensem

for i in range(ensem):

    train_x,valid_x,train_y,valid_y=train_test_split(x,y,test_size=0.2)

    history[i]=model[i].fit_generator(train_datagen.flow(train_x,train_y,batch_size=128),

                      epochs=100,

                      steps_per_epoch=train_x.shape[0]//128,

                      verbose=0,

                      validation_data=(valid_x,valid_y),

                      validation_steps=valid_x.shape[0]//128,

                      callbacks=[callback,lrs])

    print('Model {}: Epochs=100, Train_Accuracy:{}, Val_Accuracy:{}'.format(i+1,max(history[i].history['accuracy']),max(history[i].history['val_accuracy'])))
#Models Accuracy

import matplotlib.pyplot as plt



styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']

names=['Model {}'.format(i) for i in range(ensem)]

plt.figure(figsize=(15,5))

for i in range(ensem):

    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])

plt.title('Models accuracy')

plt.ylabel('Accuracy')

plt.xlabel('# Epoch')

plt.legend(names, loc='lower right')

axes = plt.gca()

axes.set_ylim([0.8,1])

plt.show()
#Plotting the loss and accuracy for the first model.

#All the modesl follow a similar trend



def plot_model(history):

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    fig.suptitle('Model Accuracy and Loss')



    ax1.plot(history[0].history['accuracy'])

    ax1.plot(history[0].history['val_accuracy'])

    ax1.title.set_text('Accuracy')

    ax1.set_ylabel('Accuracy')

    ax1.set_xlabel('Epoch')

    ax1.legend(['Train','Valid'],loc=4)



    ax2.plot(history[0].history['loss'])

    ax2.plot(history[0].history['val_loss'])

    ax2.title.set_text('Loss')

    ax2.set_ylabel('Loss')

    ax2.set_xlabel('Epoch')

    ax2.legend(['Train','Valid'],loc=1)



    fig.show()



plot_model(history)
#Classification Report and Confusion Matrix for the first model



from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns



#Predicting values from the validation dataset and one hot encoding the predicted and true labels

y_pred=model[0].predict(valid_x)

y_pred_classes=np.argmax(y_pred, axis=1)

y_true=np.argmax(valid_y, axis=1)



#Classification Report

print('Classification Report')

report=classification_report(y_true, y_pred_classes)

print(report)



#Computuing and Plotting Confusion Matrix

confusion_mtx=confusion_matrix(y_true, y_pred_classes)

f,ax=plt.subplots(figsize=(16,8))

sns.heatmap(confusion_mtx,annot=True,fmt='')

plt.xlabel("Predicted",size=12)

plt.ylabel("True",size=12)

plt.title("Confusion Matrix",size=20)

plt.show()
prediction=np.zeros((test.shape[0],10)) 

for i in range(ensem):

    prediction=prediction+model[i].predict(test)

prediction=np.argmax(prediction,axis = 1)

prediction=pd.Series(prediction,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name="ImageId"),prediction],axis=1)

submission.to_csv("digit_recognizer.csv",index=False)