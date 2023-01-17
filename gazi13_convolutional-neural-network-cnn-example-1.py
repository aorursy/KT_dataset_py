

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Load train 

train = pd.read_csv("../input/train.csv") 

# Load test 

test= pd.read_csv("../input/test.csv")



# Let's take a look at the data. 

print(train.shape)  

train.head() # show first 5 elements
print(test.shape)

test.head()
# put labels into y_train

Y_train = train["label"]

# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1)
# Let's look at a sample

img = X_train.iloc[10].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[10,0])

plt.axis("off")

plt.show()
# visualize number of digits classes

plt.figure(figsize=(12,5))

g = sns.countplot(Y_train, palette="icefire")

plt.title("Digit Classes")

Y_train.value_counts()

Y_train.shape
# Normalize the data (for color - gri)

X_train = X_train / 255.0

test = test / 255.0

print("x_train shape: ",X_train.shape)

print("test shape: ",test.shape)
#Our image already black-white but the data has R-G-B values and we reduce the size.

img = X_train.iloc[0].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[0,0])

plt.axis("off")

plt.show()
# Reshape

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape: ",X_train.shape)

print("test shape: ",test.shape)
# Label Encoding 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

Y_train = to_categorical(Y_train, num_classes = 10)
# Split the train(X) and the validation (Y) set for the fitting

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

print("x_train shape",X_train.shape)

print("x_test shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_test shape",Y_val.shape)
# 

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
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()