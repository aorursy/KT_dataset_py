# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#%% load data

train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")



train.shape



train.head()

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

test_data_id = test_data.id.values

test_data.drop(["id"],axis = 1,inplace= True)
#%% set label data

y_train = train["label"]

x_train = train.drop(labels = ["label"],axis = 1)
#%% visualization

plt.figure(figsize=(15,7))

sns.countplot(y_train , palette = "icefire")

plt.title("Number of digits")

y_train.value_counts()
img = x_train.iloc[1].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap = "gray")

plt.title(train.iloc[1,0])

plt.axis("off")

plt.show()

img = x_train.iloc[107].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap = "gray")

plt.title(train.iloc[107,0])

plt.axis("off")

plt.show()
#%% normalization - reshape - label encoding

# Normalize the data

x_train = x_train / 255.0

print("x_train shape: ",x_train.shape)



test_data = test_data / 255.0

# Reshape

x_train = x_train.values.reshape(-1,28,28,1)

print("x_train shape: ",x_train.shape)



test_data = test_data.values.reshape(-1,28,28,1)

# Label Encoding 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)
#%% train-test split (traini val olarak bölecez)

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

print("x_train shape",X_train.shape)

print("x_test shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_test shape",Y_val.shape)
#%% model

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()

#

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

# fully connected

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

# Define the optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



epochs = 10  # for better result increase the epochs

batch_size = 250
# data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

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
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)

#%% evaluate the model

# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
Y_pred = model.predict(test_data)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 
submission = np.vstack((test_data_id,Y_pred_classes))

submission = submission.T
submission1 = pd.DataFrame(submission,columns = ["id","target"])
submission1.to_csv("submission.csv",index=False)