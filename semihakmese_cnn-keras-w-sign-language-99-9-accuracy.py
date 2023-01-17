# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings 

warnings.filterwarnings("ignore")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")

test =  pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")
train.head()
test.head()
X_train = train.drop(labels=["label"],axis=1)

Y_train = train["label"]

print("X_train shape",X_train.shape)

print("Y_train shape",Y_train.shape)
X_test = test.drop(labels=["label"],axis=1)

Y_test = test.iloc[:,0]

print("X_test shape",X_test.shape)

print("Y_test shape",Y_test.shape)
print("Label Value Counts\n",Y_train.value_counts())

plt.figure(figsize =(15,10))

sns.countplot(Y_train, palette = "GnBu_d")

plt.title("Number of Digits Label Pixels")
#plotting some of the samples  

plt.subplot(2,2,1)

img1 = X_train.iloc[0].to_numpy().reshape((28,28))

plt.imshow(img1,cmap='gray')

plt.subplot(2,2,2)

img2 = X_train.iloc[1].to_numpy().reshape((28,28))

plt.imshow(img2,cmap='gray')

plt.subplot(2,2,3)

img3 = X_train.iloc[2].to_numpy().reshape((28,28))

plt.imshow(img3,cmap='gray')

plt.subplot(2,2,4)

img4 = X_train.iloc[3].to_numpy().reshape((28,28))

plt.imshow(img4,cmap='gray')

plt.show()
#Normalization

X_train = X_train.astype("float32")/255.0

X_test = X_test.astype("float32")/255.0

print("X_train shape is  >>> ",X_train.shape)

print("X_test shape  is  >>> ",X_test.shape)
#Reshape

#When we want to reshape our data firstly we need to convert the data to Numpy by using .values method

X_train = X_train.values.reshape(-1,28,28,1) #28x28 >> 784 px

X_test = X_test.values.reshape(-1,28,28,1)

print("X_train shape : ",X_train.shape)

print("Test shape : ",X_test.shape)
#Label Encoding 

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 25 ) #We got 25 labels (0 to 24)

Y_test = to_categorical(Y_test, num_classes = 25 )
#Lets see all the pictures below

f, ax = plt.subplots(4,6) 

f.set_size_inches(10, 10)

k = 0

for i in range(4):

    for j in range(6):

        ax[i,j].imshow(X_train[k].reshape(28, 28) , cmap = "gray")

        k += 1

        plt.axis("off")

        plt.savefig("graph.png")

    plt.tight_layout()    

    
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size = 0.15, random_state = 42)

print("X_train shape",X_train.shape)

print("X_val shape",X_val.shape)

print("Y_train shape",Y_train.shape)

print("Y_val shape",Y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools 



from keras.utils.np_utils import to_categorical #Converting to one hot encoding 

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ReduceLROnPlateau



epochs = 15 

batch_size = 150

num_classes = 25



model = Sequential()

#Convutional Layer 1 

model.add(Conv2D(75, (3,3), strides = 1, padding = "Same",activation ="relu", input_shape =(28,28,1)))

model.add(BatchNormalization())



#Pooling Layer 1 

model.add(MaxPool2D((2,2),strides = 2,padding ="Same"))



#Convutional Layer 2

model.add(Conv2D(50, (3,3), strides = 1, padding = "Same",activation ="relu", input_shape =(28,28,1)))

model.add(Dropout(0.2))

model.add(BatchNormalization())



#Pooling Layer 2

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))



#Convutional Layer 3

model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(BatchNormalization())



#Pooling Layer 3

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))



#Fully Connected Layer

model.add(Flatten())

    #Hidden Layer 1 

model.add(Dense(units = 512 , activation = 'relu'))

model.add(Dropout(0.3))

    #Hidden Layer 2

model.add(Dense(units = 25 , activation = 'softmax'))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
#Compiler

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

#Model Summary

model.summary()
# With data augmentation to prevent overfitting



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180 by 5 degrees)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
history = model.fit(datagen.flow(X_train,Y_train,

                                 batch_size = batch_size),

                                 epochs = epochs, 

                                 validation_data =(X_val,Y_val),

                                 steps_per_epoch = X_train.shape[0]//batch_size,

                                 callbacks = [learning_rate_reduction])
score = model.evaluate(X_test,Y_test,verbose = 0)

print("Test Loss : ",score[0])

print("Test Accuracy : ",score[1])
import matplotlib.pyplot as plt

%matplotlib inline

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
# confusion matrix

import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Spectral",linecolor="blue", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()