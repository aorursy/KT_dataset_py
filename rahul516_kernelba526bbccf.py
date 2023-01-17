# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing the libraries

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
train_df = pd.read_csv("../input/sign_mnist_train.csv")

test_df = pd.read_csv("../input/sign_mnist_test.csv")
train_df.head()
test_df.head()
train_df.info()
print(train_df.shape)
print(test_df.shape)
label = train_df.label
label_test = test_df.label
train_df = train_df.drop("label", axis =1)

test_df = test_df.drop("label", axis = 1)

print(train_df.shape, "\n",test_df.shape)
sns.countplot(label)
sns.countplot(label_test)
label.value_counts()
x_train = train_df.values

x_test = test_df.values

y_train = label

y_test = label_test
y_train.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28)
for i in range(0, 10):

    plt.subplot(1, 10, i+1)

    plt.imshow(x_train[i])

    plt.title(y_train[i])
x_train.shape
y_train.shape
plt.imshow(x_train[10])
y_train[10]
x_train_c = x_train.reshape(x_train.shape[0], 28, 28,1)
x_train_c.shape
x_test_c = x_test.reshape(x_test.shape[0], 28, 28, 1)
#normalizing the data

x_train_c = x_train_c/254.0

x_test_c = x_test_c/254.0
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 25)
y_train.shape
y_train[:5]
#spliting the training and test set

#set the reandom seed

SEED = 2

# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(x_train_c, y_train, test_size = 0.1, random_state=SEED)
y_train.shape
plt.imshow(X_train[0][:, :, 0])

for i in range(25):

    if Y_train[0, i] == 1:

        y = i

        print(y)

plt.title(y)
#building the cnn model

model = Sequential()

#adding 2 convolution layer followed by pooling layer and dropout layer 

model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = "same", activation = "relu", input_shape = (28, 28, 1)))

model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = "same", activation = "relu", ))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
#adding 2 convolution layer followed by pooling layer and dropout layer

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',  activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',  activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), ))

model.add(Dropout(0.25))
#adding a flattning layer and followed by ANN neural layer

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(25, activation = "softmax"))
#adding the optimizer, I here used msprop but we can use adam or sgd also. But I find rms prop useful as it faster than sgd 

#better than adam

# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
from keras.optimizers import SGD, Adam

optimizer_1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)

optimizer_2 = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=[ "accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 30 

batch_size = 64
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Fit the model

classifier = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
plt.plot(classifier.history['acc'])

plt.plot(classifier.history['val_acc'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
y_test = to_categorical(y_test, num_classes = 25)
y_test.shape
x_test_c.shape
y_pred = model.predict(x_test_c)
from sklearn.metrics import accuracy_score
y_pred[:2]
accuracy_score(y_test, y_pred.round())