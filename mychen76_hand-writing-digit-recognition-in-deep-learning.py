# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

import keras.layers.core as core

import keras.layers.convolutional as conv

import keras.models as models

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

# one-hot-encoding convertion

from keras.utils.np_utils import to_categorical 

import keras.utils.np_utils as kutils

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Set seed

seed = 5

np.random.seed(seed)
#read data files into pandas dataframe

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#does some basic data quality checking

print(train.shape, test.shape)

train.tail()
#any empty values?

# train.isnull().sum()

train.isnull().any().describe()
#test.isnull().sum()

test.isnull().any().describe()
# ensure traing data using correct data type. 

# note: use 16bits could potentially save more GPU memory space

X_train = (train.iloc[:,1:].values).astype('float32') 

y_train = train.iloc[:,0].values.astype('int32') 



# change train datset format to keras format 

X_train = X_train.reshape(-1, 28, 28,1)

# normalize the data to grayscale to reduce effect of illumination's diffencens

X_train = X_train / 255.0

print(X_train.shape , y_train.shape)



#make same update for test data

test = test.values.reshape(-1, 28, 28, 1)

test = test.astype(float)

# normalize the data to grayscale to reduce effect of illumination's diffencens

test /= 255.0

print (test.shape)
# one-hot vector for training labels classes

from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)

num_classes = y_train.shape[1]

print ("Number of classes: ",num_classes)
# split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state=seed)

print ("Shapes of train, validation dataset ")

print(X_train.shape , Y_train.shape)

print(X_val.shape , Y_val.shape)
# view few example data

for i in range(1,5):

    plt.subplot(2,2,i)

    g = plt.imshow(X_train[i][:,:,0], cmap=plt.get_cmap('gray'))

plt.show()
# adj parameters

filters_1 = 32 

filters_2 = 64 

filters_3 = 128 



# Create model

model = models.Sequential()

model.add(conv.Convolution2D(filters_1, (3,3),  activation="relu", input_shape=(28, 28, 1), border_mode='same'))

model.add(conv.Convolution2D(filters_1, (3,3), activation="relu", border_mode='same'))

model.add(conv.MaxPooling2D(strides=(2,2)))

model.add(conv.Convolution2D(filters_2,(3,3), activation="relu", border_mode='same'))

model.add(conv.Convolution2D(filters_2, (3,3), activation="relu", border_mode='same'))

model.add(conv.MaxPooling2D(strides=(2,2)))

model.add(core.Flatten())

model.add(core.Dropout(0.2))

model.add(core.Dense(128, activation="relu"))

model.add(core.Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()
%%time

# best practice tips

print ("apply augumentation or data noisy...")

# apply data augmentation to create noisy on data which increase accuracy

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

datagen.fit(X_train)



print ("training started...")

# Train

epochs = 15 

batch_size = 128

# callback checkpoint

checkpoint = ModelCheckpoint('model-best-trained.h5', verbose=0, monitor='loss',save_best_only=True, mode='auto')  

# callback learning rate reducer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction,checkpoint])

print ("training finished!")
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)



score = model.evaluate(X_val, Y_val, verbose=0)

print("model evaluation score: %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
# Predict

print ("Running prediction test....")

predictions = model.predict_classes(test,verbose=1)

print ("done")
# save file using numpy

np.savetxt('digits-mnist-cnn-3.csv', np.c_[range(1,len(predictions)+1),predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

print ("saved prediction to file")

sub = pd.read_csv("digits-mnist-cnn-3.csv")

sub.tail(10)
# save file using data-frame

#submission_result_file="digits-mnist-cnn-4.csv"

#submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

#submissions.to_csv(submission_result_file, index=False, header=True)

#print ("saved prediction to file")

#submissions.tail(10)