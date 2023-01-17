import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns
# Read to train dataset.

train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

train.head()
test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

test = test.drop(labels=['label'],axis=1) # I drop label column

test.head()
# Y_train and X_train seperate

Y_train = train['label']

X_train = train.drop(labels=['label'],axis=1)
# Visualize Labels count

plt.figure(figsize=(15,7))

g = sns.countplot(Y_train, palette="RdBu")

plt.title("Number Of Labels")

Y_train.value_counts()
# Plot some samples

plt.subplot(2,2,1)

img = X_train.iloc[0].to_numpy()

img = img.reshape((28,28))

plt.imshow(img)

plt.title('Samples 1')

plt.axis('off')

# -----------------------

plt.subplot(2,2,2)

img1 = X_train.iloc[8].to_numpy()

img1 = img1.reshape((28,28))

plt.imshow(img1)

plt.title('Sample 2')

plt.axis('off')

# -----------------------

plt.subplot(2,2,3)

img2 = X_train.iloc[12].to_numpy()

img2 = img2.reshape((28,28))

plt.imshow(img2)

plt.title('Sample 3')

plt.axis('off')

# -----------------------

plt.subplot(2,2,4)

img3 = X_train.iloc[67].to_numpy()

img3 = img3.reshape((28,28))

plt.imshow(img3)

plt.title('Sample 4')

plt.axis('off')

plt.show()
# Normalization

X_train = X_train / 255.0

test = test / 255.0



# Reshape

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

#----------------------------------

print('X_train SHAPE:',X_train.shape)

print('Test SHAPE:',test.shape)
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train,num_classes=10)
# Seperate train and val.

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.1,random_state=42)

# - - - - - - - - - - - - - - - - - - - - -

print('x_train SHAPE:',x_train.shape)

print('x_val SHAPE:',x_val.shape)

print('y_train SHAPE:',y_train.shape)

print('y_val SHAPE:',y_val.shape)
# Dics for CNN

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import Adam,RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# Create Model

model = Sequential()

model.add(Conv2D(filters=8,kernel_size=(7,7),padding='same',activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#

model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#

model.add(Conv2D(filters=64,kernel_size=(2,2),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Fully Connection

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10,activation='softmax'))



# Optimizer

optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

epochs = 10

batch_size = 250



# Datagen

datagen = ImageDataGenerator(featurewise_center=False,

                            samplewise_center=False,

                            featurewise_std_normalization=False,

                            samplewise_std_normalization=False,

                            zca_whitening=False,

                            rotation_range=0.5,

                            zoom_range=0.5,

                            width_shift_range=0.5,

                            height_shift_range=0.5,

                            horizontal_flip=False,

                            vertical_flip=False)

datagen.fit(x_train)



# Fit model

history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),

                             epochs=epochs,validation_data=(x_val,y_val),steps_per_epoch=x_train.shape[0]//batch_size)
# Loss Graph

plt.plot(history.history['val_loss'], color='Lime', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# confusion matrix

import seaborn as sns

# Predict the values from the validation dataset

y_pred = model.predict(x_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Reds",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()