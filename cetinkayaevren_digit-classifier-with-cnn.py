# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Train Dataset



train = pd.read_csv('../input/digit-recognizer/train.csv')

print(train.shape)

train.head(15)
#Test Dataset



test = pd.read_csv('../input/digit-recognizer/test.csv')

print(test.shape)

test.head(15)
#Seperating labels

y_train_first = train["label"]



#Dropping label to the main dataset

x_train_first = train.drop(labels = ["label"], axis = 1)
print(x_train_first.shape)

x_train_first
#Visualizing numbers

plt.figure(figsize = (15,7))

g = sns.countplot(y_train_first, palette = "icefire")

plt.title("Number of digits")

y_train_first.value_counts()
img = x_train_first.iloc[0].to_numpy() #as_matrix() method is deprecated !!

img = img.reshape((28,28))

plt.imshow(img, cmap = 'gray')

plt.title(train.iloc[0,0])

plt.axis('off')

plt.show()
img = x_train_first.iloc[70].to_numpy() #as_matrix() method is deprecated !!

img = img.reshape((28,28))

plt.imshow(img, cmap = 'gray')

plt.title(train.iloc[70,0])

plt.axis('off')

plt.show()
#Normalize the data

x_train_first = x_train_first / 255.0 #colors can take 255 numbers (max: 255)

test = test / 255.0

print("x_train shape: ",x_train_first.shape)

print("test shape: ",test.shape)

#Reshaping

x_train_first = x_train_first.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape: ",x_train_first.shape)

print("test shape: ",test.shape)
#Encoding labels

from keras.utils.np_utils import to_categorical #Converts to one-hot-encoding

y_train_first = to_categorical(y_train_first, num_classes = 10)
y_train_first
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x_train_first, y_train_first, test_size = 0.1, random_state = 42)

print("x_train shape: ",X_train.shape)

print("x_test shape: ",X_val.shape)

print("y_train shape: ",Y_train.shape)

print("y_test shape: ",Y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential() #Add Layers to the model

#***************



model.add(Conv2D(filters = 8, kernel_size= (5,5), padding = 'Same',

                activation = 'relu', input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25)) #Yüzde 25 node u active et ya da etme

#***************



model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same',

                activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2))) #Strides, 2 adım atlayarak gez pixellerde

model.add(Dropout(0.25))

#Fully Connected



model.add(Flatten())

model.add(Dense(256, activation = "relu")) #Hidden Layere

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) #Hidden Layer



#Softmax function is generilazed version of sigmoid function.

#Softmax is generally used for multiple classifications instead of binary classification



optimizer = Adam(lr= 0.001, beta_1 = 0.9, beta_2 = 0.999) #Beta parameters are determines the learning rate's variability'
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
epochs = 10 # for better result increase the epochs

batch_size = 250
# Data augmentation

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.5,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.5, # Randomly zoom image 5%

        width_shift_range=0.5,  # randomly shift images horizontally 5%

        height_shift_range=0.5,  # randomly shift images vertically 5%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),

                             epochs = epochs, validation_data = (X_val, Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'], color = 'b', label = "validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

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

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()