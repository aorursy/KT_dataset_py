# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the dependencies

from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt

%matplotlib inline 
#Reading the Lookup Table

IdLookupTable = pd.read_csv('/kaggle/input/IdLookupTable.csv')

IdLookupTable.info()
IdLookupTable.head()
#Reading the training dataset

training = pd.read_csv('/kaggle/input/training/training.csv')

training.info()
training.head()
test = pd.read_csv('/kaggle/input/test/test.csv')

test.info()
test.head()
#There are lot of null values in the training dataset. Let's drop all the rows which have null values. 

training = training.dropna()
training.info()
training.shape, type(training)
#We have in total of 2140 images for training.
#Let us reshape the value if the Image column to (96 x 96) basically of height and width

training['Image'] = training['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
training['Image'].shape
#Let us see the data of the Image columns

training['Image']
#Let us write a function to combine the images and the facial points of the images.



def get_image_and_points(df, index):

    image = plt.imshow(df['Image'][index],cmap='gray') #Converting the image to grayscale

    

    l = []

    for i in range(1,31,2):

        l.append(plt.plot(df.loc[index][i-1], df.loc[index][i], 'ro'))

        

    return image, l
#Let us plot the images along with the facial points for visualizing before training the model

fig = plt.figure(figsize=(8, 8))

fig.subplots_adjust(

    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



for i in range(16):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    get_image_and_points(training, i)



plt.show()
X = np.asarray([training['Image']], dtype=np.uint8).reshape(training.shape[0],96,96,1)

y = training.drop(['Image'], axis=1)
X.shape
y.shape
type(X), type(y)
#We need to convert the y datatype to numpy array

y2= y.to_numpy()
type(y2), y2.shape
# Lets us split the training dataset for train and validation



X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.3, random_state=42)



# reshape to be [samples][width][height][channels]

X_train = X_train.reshape((X_train.shape[0], 96, 96, 1))

X_test = X_test.reshape((X_test.shape[0], 96, 96, 1))



# convert from int to float

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



#Normalize the input image

X_train = X_train / 255

X_test = X_test / 255
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout, Activation, MaxPooling2D

from keras.optimizers import Adam, SGD, RMSprop

from keras import regularizers

from keras.layers.advanced_activations import ReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D



model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

# model.add(BatchNormalization())

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
from keras import optimizers

adam =optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=adam, 

              loss='mse', 

              metrics=['accuracy'])
import keras

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

    def on_batch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))



history = LossHistory()

hist = model.fit(X_train, y_train, epochs=200,batch_size =64, validation_data=(X_test,y_test), callbacks=[history])
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.save('keypoint_model4.h5')
test['Image'] = test['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
test_X = np.asarray([test['Image']], dtype=np.uint8).reshape(test.shape[0],96,96,1)



test_X = test_X / 255



test_res = model.predict(test_X)
header = list(y.columns)
test_predicts = pd.DataFrame(test_res, columns = header)
for i in range(IdLookupTable.shape[0]):

    IdLookupTable.Location[i] = test_predicts.loc[IdLookupTable.ImageId[i]-1][IdLookupTable.FeatureName[i]]
SampleSubmission = pd.read_csv('/kaggle/input/SampleSubmission.csv')



SampleSubmission.Location = IdLookupTable.Location

my_submission = SampleSubmission

my_submission.to_csv('submission4.csv', index=False)
test_res.shape
test_res[0]
test_res[0][1]
from keras.preprocessing.image import img_to_array, array_to_img

plt.imshow(array_to_img(test_X[0]))
#Let us plot the images along with the facial points for visualizing  the test images

fig = plt.figure(figsize=(8, 8))

fig.subplots_adjust(

    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



for i in range(16):

    

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    image = plt.imshow(array_to_img(test_X[i])) #Converting the image to grayscale

    for j in range(1,31,2):

        plt.plot(test_res[i][j-1], test_res[i][j], 'ro')

plt.show()