# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load The Libraries



import matplotlib.pyplot as plt 

import cv2 as cv



from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D

from keras import models

from keras.optimizers import Adam,RMSprop 

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import pickle



%matplotlib inline
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train = train.iloc[np.random.permutation(len(train))]

train.head()
train.shape
#Preparing Training and Validation data



sample_size = train.shape[0] # Training set size

validation_size = int(train.shape[0]*0.1) # Validation set size 



# train_x and train_y

train_x = np.asarray(train.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0

train_y = np.asarray(train.iloc[:sample_size-validation_size,0]).reshape([sample_size-validation_size,1]) # taking column 0



# val_x and val_y

val_x = np.asarray(train.iloc[sample_size-validation_size:,1:]).reshape([validation_size,28,28,1])

val_y = np.asarray(train.iloc[sample_size-validation_size:,0]).reshape([validation_size,1])
train_x.shape,train_y.shape
#Loading test.csv



test = pd.read_csv("../input/digit-recognizer/test.csv")

test_x = np.asarray(test.iloc[:,:]).reshape([-1,28,28,1])
train_x = train_x/255

val_x = val_x/255

test_x = test_x/255
counts = train.iloc[:sample_size-validation_size,:].groupby('label')['label'].count()

# df_train.head(2)

# counts

f = plt.figure(figsize=(10,6))

f.add_subplot(111)



plt.bar(counts.index,counts.values,width = 0.8,color="orange")

for i in counts.index:

    plt.text(i,counts.values[i]+50,str(counts.values[i]),horizontalalignment='center',fontsize=14)



plt.tick_params(labelsize = 14)

plt.xticks(counts.index)

plt.xlabel("Digits",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Frequency Graph training set",fontsize=20)

plt.show()



counts = train.iloc[sample_size-validation_size:,:].groupby('label')['label'].count()

# df_train.head(2)

# counts

f = plt.figure(figsize=(10,6))

f.add_subplot(111)



plt.bar(counts.index,counts.values,width = 0.8,color="orange")

for i in counts.index:

    plt.text(i,counts.values[i]+5,str(counts.values[i]),horizontalalignment='center',fontsize=14)



plt.tick_params(labelsize = 14)

plt.xticks(counts.index)

plt.xlabel("Digits",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Frequency Graph Validation set",fontsize=20)

plt.show()
rows = 5 # defining no. of rows in figure

cols = 6 # defining no. of colums in figure



f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 



for i in range(rows*cols): 

    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration

    plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 

    plt.axis("off")

    plt.title(str(train_y[i]), y=-0.15,color="green")
model = models.Sequential()
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model.add(LeakyReLU())

model.add(Conv2D(32,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Block 2

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation="sigmoid"))
initial_lr = 0.001

loss = "sparse_categorical_crossentropy"

model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

model.summary()
epochs = 20

batch_size = 256

history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])
# Diffining Figure

f = plt.figure(figsize=(20,7))



#Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()

#Confusion Matrix



val_p = np.argmax(model.predict(val_x),axis =1)



error = 0

confusion_matrix = np.zeros([10,10])

for i in range(val_x.shape[0]):

    confusion_matrix[val_y[i],val_p[i]] += 1

    if val_y[i]!=val_p[i]:

        error +=1

        

confusion_matrix,error,(error*100)/val_p.shape[0],100-(error*100)/val_p.shape[0],val_p.shape[0]



print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])

print("\nValidation set Shape :",val_p.shape[0])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(train_x)
#Learning rate



lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.00001)
#Further Traning



epochs = 20

history_2 = model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=[val_x,val_y],callbacks=[lrr])
#Training Performance.**



f = plt.figure(figsize=(20,7))

f.add_subplot(121)



#Adding Subplot 1 (For Accuracy)

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['accuracy']+history_2.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_accuracy']+history_2.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()





#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['loss']+history_2.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_loss']+history_2.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()





#Confusion Matrix



val_p = np.argmax(model.predict(val_x),axis =1)



error = 0

confusion_matrix = np.zeros([10,10])

for i in range(val_x.shape[0]):

    confusion_matrix[val_y[i],val_p[i]] += 1

    if val_y[i]!=val_p[i]:

        error +=1

        

confusion_matrix,error,(error*100)/val_p.shape[0],100-(error*100)/val_p.shape[0],val_p.shape[0]



print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])

print("\nValidation set Shape :",val_p.shape[0])

#Visualizing Result



rows = 4

cols = 9



f = plt.figure(figsize=(2*cols,2*rows))

sub_plot = 1

for i in range(val_x.shape[0]):

    if val_y[i]!=val_p[i]:

        f.add_subplot(rows,cols,sub_plot) 

        sub_plot+=1

        plt.imshow(val_x[i].reshape([28,28]),cmap="Blues")

        plt.axis("off")

        plt.title("T: "+str(val_y[i])+" P:"+str(val_p[i]), y=-0.15,color="Red")

plt.show()
#Predict on test set¶





test_y = np.argmax(model.predict(test_x),axis =1)



rows = 5

cols = 10



f = plt.figure(figsize=(2*cols,2*rows))



for i in range(rows*cols):

    f.add_subplot(rows,cols,i+1)

    plt.imshow(test_x[i].reshape([28,28]),cmap="Blues")

    plt.axis("off")

    plt.title(str(test_y[i]))

df_submission = pd.DataFrame([test.index+1,test_y],["ImageId","Label"]).transpose()

df_submission.to_csv("submission.csv",index=False)