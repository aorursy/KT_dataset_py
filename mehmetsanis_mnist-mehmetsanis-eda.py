# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns





np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



testIds = pd.Series(test.index.tolist(), name = 'ImageId') + 1
train.head()
train.shape
test.shape
Y_train = train['label']



X_train = train.drop( labels=['label'], axis = 1)
g = sns.countplot( Y_train)

plt.show()
Y_train.value_counts()
# plot some samples

img = X_train.iloc[0].values

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[0,0])

plt.axis("off")

plt.show()
X_train.isnull().any().describe()
X_train = X_train / 255.0



test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)



X_train = X_train.values.reshape(-1,28,28,1)

test  = test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=42)
# Some examples

g = plt.imshow(X_train[0][:,:,0])
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()





model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))

model.add(MaxPool2D((2, 2) , strides=(2,2)))

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2) , strides=(2,2)))





model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dense(256, activation = "relu"))

model.add(Dense(128, activation = "relu"))

model.add(Dense(10, activation = "softmax"))
optimizer = Adam(lr=0.001)
model.compile( optimizer = optimizer , loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs = 30



batch_size = 64
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
history = model.fit_generator( datagen.flow( X_train, Y_train, batch_size = batch_size ),

                              epochs = epochs,

                              validation_data = (X_val, Y_val),

                              steps_per_epoch=X_train.shape[0] // batch_size

                              )
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

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
# Predict the values from the validation dataset

y_head = model.predict(test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(y_head,axis = 1) 
Y_pred_classes = pd.Series(Y_pred_classes, name = 'Label')
results = pd.concat( [testIds , Y_pred_classes], axis = 1 )

results.to_csv("submission.csv", index = False)
results