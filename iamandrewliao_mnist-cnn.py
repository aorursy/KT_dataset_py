# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



sns.set(style='white', context='notebook', palette='deep')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
y_train = train.label

x_train = train.drop('label', 1)
train.shape
train.isnull().values.any() #checking to see if there are nulls
sns.countplot(y_train) #checking the balance of the data
n = np.random.randint(0, 42000)

img = x_train.iloc[n].to_numpy() #past tutorials use as_matrix() which is now deprecated: https://stackoverflow.com/questions/60164560/attributeerror-series-object-has-no-attribute-as-matrix-why-is-it-error

img = img.reshape((28,28))

plt.imshow(img)

plt.title(train.iloc[n,0])

plt.show()
#grayscale normalization of features

x_train /= 255.0

test /= 255.0
#reshaping feature data

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape: ",x_train.shape)

print("test shape: ",test.shape)
#encoding labels as arrays e.g. 2 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

y_train = to_categorical(y_train, num_classes = 10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

#for imbalanced datasets, use stratify=True: https://stackoverflow.com/a/38889389
x_train.shape
plt.imshow(x_train[0].reshape(28,28)) #using [:,:,0] instead of reshape does the same

plt.show()
#The model used here is from this experiment: https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist\

model = Sequential()



model.add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32,kernel_size=5,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.40))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.40))



model.add(Flatten())

model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.40))

model.add(Dense(10, activation = "softmax"))



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 10  # for better result increase the epochs

batch_size = 100
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



datagen.fit(x_train)
LR_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience = 2, 

                                            verbose = 1, 

                                            factor = 0.5, 

                                            min_lr = 0.00001)
model_datagen = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),

                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size, callbacks=[LR_reduction])
plt.plot(model_datagen.history['val_loss'], color='b', label="validation loss")

plt.title("Validation Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.show()
results = model.predict(test)



# select the prediction with the maximum probability

results = np.argmax(results, axis = 1)



results = pd.Series(results, name="Label")



submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)



submission.to_csv("submission.csv", index=False)