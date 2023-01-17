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
## Getting the required libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



## Ignore Warnings

import warnings

warnings.filterwarnings('ignore')
## Importing the required datasets



digit = pd.read_csv("../input/train.csv")
digit.head()
digit.shape
digit.iloc[0].value_counts()
## Loading the test data



test = pd.read_csv("../input/test.csv")
## Viewing the data

print(test.shape)

test.head()
## Separating the X and Y variable



Y_train = digit['label']



## Dropping the variable 'label' from X variable 

X_train = digit.drop(columns = 'label')



## Printing the size of data 

print(X_train.shape)
## Visualizing the number of class and counts in the datasets



plt.plot(figure = (16,5))

g = sns.countplot(Y_train, palette = 'icefire')

plt.title('NUmber of digit classes')

Y_train.value_counts()
## Plotting some samples as well as converting into matrix



img = X_train.iloc[3].as_matrix()

img = img.reshape(28,28)

plt.imshow(img)

plt.title("Digit")
## Normalization



X_train = X_train/255.0

test = test/255.0



print("X_train:", X_train.shape)

print("X_test:", test.shape)
## Reshaping the value 



X_train = X_train.values.reshape(-1, 28,28,1)

test = test.values.reshape(-1,28,28,1)



print("X_train:", X_train.shape)

print("test:", test.shape)
## Label Encoding



from keras.utils import to_categorical



Y_train = to_categorical(Y_train, num_classes = 10)
## Splitting the datasets



## Importing the libraries



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size = 0.75, test_size = 0.25,

                                                    random_state = 100)



## Viewing the size

print("X_train :", X_train.shape)

print("Y_train:", Y_train.shape)

print("X_test:", X_test.shape)

print("Y_test:", Y_test.shape)
plt.imshow(X_train[4][: , : , 0], cmap = 'gray')

plt.show()
## Building a Convolutional Neural Networks



## Importing the required libraries

from keras.models import Sequential                  ## To create a CNN model



from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation ## layers to built a models



from keras.optimizers import Adam, RMSprop     ## Optimizers



from keras.layers.normalization import BatchNormalization                ## Normalization

from keras.callbacks import ReduceLROnPlateau
## Building a model



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size= (5,5), padding = 'Same', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size = (2,2)))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size = (2,2)))



model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same'))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size = (2,2)))

                    

## Fully Connected

model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation = 'softmax'))
model.summary()
## Defining the Optimizers



##Adam optimizers : Changing the learning rate



## Defining the learning rate



optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
## Compile the model



model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
## Set a learning rate annealer



learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 

                                           patience = 3, 

                                           varbose = 1, 

                                           factor=0.5, 

                                           min_lr = 0.00001)
## Epochs and Batch Size 



epochs = 50     ## For better results, increase epochs

batch_size = 50
# data augmentation



from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(   

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=10,  # randomly rotate images in the range 10 degrees

        zoom_range = 0.1, # Randomly zoom image 10%

        width_shift_range=0.1,  # randomly shift images horizontally 10%

        height_shift_range=0.1,  # randomly shift images vertically 10%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False  # randomly flip images

        )



datagen.fit(X_train)
plt.imshow(X_train[4][: , : , 0], cmap = 'gray')

plt.show()
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                              epochs=50, validation_data = (X_test, Y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks=[learning_rate_reduction]) 
## Evaluate the model by using Confusion Matrix and Validation & Loss Visualization



# Plot the loss curve for training

plt.plot(history.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curve for training

plt.plot(history.history['acc'], color='g', label="Train Accuracy")

plt.title("Train Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Plot the loss curve for validation 

plt.plot(history.history['val_loss'], color='r', label="Validation Loss")

plt.title("Validation Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curve for validation 

plt.plot(history.history['val_acc'], color='g', label="Validation Accuracy")

plt.title("Validation Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print('Train accuracy of the model: ',history.history['acc'][-1])
print('Train loss of the model: ',history.history['loss'][-1])

print('Validation accuracy of the model: ',history.history['val_acc'][-1])

print('Validation loss of the model: ',history.history['val_loss'][-1])
test = pd.read_csv('../input/test.csv')

test = test.values.reshape(-1,28,28,1)

test.shape
# predict results

results = model.predict(test)
# select the index with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,len(test)+1),name = "ImageId"),results],axis = 1)



submission.to_csv("Digit_Recognizer_CNN_Result_2.csv",index=False)
submission.head(20)