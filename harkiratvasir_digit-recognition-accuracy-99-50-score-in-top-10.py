import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Importing  the  dataset using pandas 

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
#Checking the first 5 columns in the train dataset

print(train.head())

print('Shape of the train is {}'.format(train.shape))

print('Shape of the test is {}'.format(test.shape))
# Visualising the label using the  bar plot to get the  idea of the  distribution of the target feature

import seaborn as sns

sns.countplot(train['label'])
#Breaking the training dataset into matrix of independent features and dependent feature vector

X = train.drop('label',axis = 1)

y = train.label
# Let's check the shape 

print('Shape of the X is {}'.format(X.shape))

print(f'Shape of y is {y.shape}')
#Using the matplotlib to display the image and the  label of the  image 

import matplotlib.pyplot as plt

plt.imshow(X.values.reshape(-1,28,28)[3],cmap = 'gray')

plt.show()

print(y[3])
#Reshaping the size of the image to 28 x 28 pixels

X = X.values.reshape(-1,28,28,1)
#Rechecking the shape of the image

print('Shape of the X is {}'.format(X.shape))
#Converting the dependent feature vector into one hot  encoding 

from keras.utils import to_categorical

y = to_categorical(y)
#Dividing the dataset into training and validation set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 123)
#Model building using keras

from keras.models import Sequential

from keras.layers import Dense, Flatten,MaxPooling2D,Dropout

from keras.layers.convolutional  import Conv2D



model = Sequential()



model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',input_shape = (28,28,1)))

model.add(Conv2D(64,kernel_size = (3,3),activation = 'relu'))

model.add(Flatten())

model.add(Dense(10,activation = 'softmax'))

#Compiling  the  model

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
#Training

history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 20)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs   = range(len(acc)) 



# Plotting training and validation accuracy per epoch

plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



# Plotting training and validation loss per epoch

plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.legend()

plt.title('Training and validation loss')
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# Compile the model

#from keras.optimizers import RMSprop

#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
#Data agumentation to preventing the overfitting

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.1,  

        width_shift_range=0.1, 

        height_shift_range=0.1,  

        horizontal_flip=False,  

        vertical_flip=False)  





datagen.fit(X_train)
# Set a learning rate annealer

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

batch_size = 256

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = 30, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs   = range(len(acc)) 



# Diffining Figure

fig = plt.figure(figsize=(20,7))



#Subplot 1 (For Accuracy)

fig.add_subplot(121)



plt.plot(epochs, acc,label = 'Accuracy')

plt.plot(epochs, val_acc,label = 'Validation Accuracy')

plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Subplot 2 (For Loss)

fig.add_subplot(122)

plt.plot(epochs, loss,label = 'Loss')

plt.plot(epochs, val_loss,label = 'Validation Loss')

plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()

plt.show()
#Reshaping the test dataset

test = test.values.reshape(-1,28,28,1)
#Predicting the test using our model

predictions = model.predict_classes(test, verbose=1)
predictions
#Converting our predictions into dataframe

prediction = pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),"Label":predictions})
#Finally exporting to the csv file as the output

prediction.to_csv('kaggle_submission.csv',index=False,header=True)

prediction
from keras.models import Sequential

from keras.layers import Dense, Flatten,MaxPooling2D,Dropout,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.layers.convolutional  import Conv2D



model = Sequential()



model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(Conv2D(128,(3,3),activation='relu'))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(128,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(64,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(10,activation='softmax'))



#Compiling the model

model.compile(RMSprop(lr=0.001,rho=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=False,

                                   fill_mode='nearest')

train_datagen.fit(X_train)

train_generator = train_datagen.flow(X_train,y_train,batch_size=128)

from keras.callbacks import ReduceLROnPlateau,EarlyStopping



earlystop = EarlyStopping(monitor='val_loss',patience=2,verbose=1)

learning_reduce = ReduceLROnPlateau(patience=2,monitor="val_acc",verbose=1,min_lr=0.00001,factor=0.5)

#callbacks = [earlystop,learning_reduce]

callbacks = [learning_reduce]

history = model.fit_generator(train_generator,epochs=30,verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs   = range(len(acc)) 



# Diffining Figure

fig = plt.figure(figsize=(20,7))



#Subplot 1 (For Accuracy)

fig.add_subplot(121)



plt.plot(epochs, acc,label = 'Accuracy')

plt.plot(epochs, val_acc,label = 'Validation Accuracy')

plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Subplot 2 (For Loss)

fig.add_subplot(122)

plt.plot(epochs, loss,label = 'Loss')

plt.plot(epochs, val_loss,label = 'Validation Loss')

plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()