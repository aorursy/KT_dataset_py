import pandas as pd

import numpy as np

import cv2

import pathlib

import imageio

from skimage.transform import resize



#converting normal images into numpy array



training_paths = pathlib.Path('../input/chest-xray-pneumonia/chest_xray').glob('train/NORMAL/*.jpeg')

training_sorted = sorted([x for x in training_paths])



training_images = np.zeros(2352)

for index in range(301):

    im_path = training_sorted[index]

    img = cv2.imread(str(im_path))

    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    res = res.flatten()

    training_images = np.vstack((training_images, res))

    

#converting pneumonia images into numpy array



training_paths = pathlib.Path('../input/chest-xray-pneumonia/chest_xray').glob('train/PNEUMONIA/*.jpeg')

training_sorted = sorted([x for x in training_paths])



training_images_affected = np.zeros(2352)

for index in range(301):

    im_path = training_sorted[index]

    img = cv2.imread(str(im_path))

    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    res = res.flatten()

    training_images_affected = np.vstack((training_images_affected, res))

    
#copying and removing unnecessary rows





train_images_dummy = training_images

train_images_aff_dummy = training_images_affected



train_images_dummy = train_images_dummy[2:]

train_images_aff_dummy = train_images_aff_dummy[2:]
#concatenating the pixel values into the data frame





dfcop = train_images_dummy

dfcop = np.concatenate([dfcop, train_images_aff_dummy])





df_images = pd.DataFrame(dfcop)

df_images = pd.DataFrame(df_images.loc[:, :])
#copying to a new dataframe



df = pd.DataFrame(df_images)

df = pd.DataFrame(df.loc[:, :])

df
#adding target column



normal = 300

affected = 300



temp = pd.DataFrame()

temp = np.ones(normal)

temp = np.concatenate([temp, np.zeros(affected)])

temp = pd.DataFrame(temp) 





# df = pd.DataFrame(temp)

# df = pd.DataFrame(df.loc[:, 0])

# df.columns = ['target']

# df.target.value_counts()



df['target'] = pd.DataFrame(temp)
df
#shuffling the rows



df = df.sample(frac=1, axis=0).reset_index(drop=True)

df
#diving rows into training and testing



size = len(df)

training_set = df.loc[0: 0.8*size,]

testing_set = df.loc[0.8*size:, ]





training = np.array(training_set, dtype = 'float32')

testing = np.array(testing_set, dtype = 'float32')

training.shape
#separating xtrain and ytrain





ytrain = training[:, -1]

xtrain = training[:, :-1]
#separating xtest and ytest



ytest = testing[:, -1]

xtest = testing[:, :-1]
#normalizing the values between 0-1



xtrain = xtrain/255

xtest = xtest/255
#splitting into train and validation set





from sklearn.model_selection import train_test_split





xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size = 0.2, random_state = 42)

xtrain.shape
xtrain.shape[0]
xtrain = xtrain.reshape(xtrain.shape[0], *(28,28,3))

xval = xval.reshape(xval.shape[0], *(28,28,3))

xtest = xtest.reshape(xtest.shape[0], *(28,28,3))
import tensorflow

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard



model = Sequential()

model.add(Conv2D(24,3,3, input_shape = (28,28,3), activation = 'elu'))

model.add(Conv2D(36,3,3, activation = 'elu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(36,3,3,  activation = 'elu'))



# model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())



model.add(Dense(output_dim = 64, activation = 'relu'))

model.add(Dense(output_dim = 32, activation = 'relu'))

model.add(Dense(output_dim = 1, activation = 'sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.005), metrics = ['accuracy'])
history = model.fit(xtrain, 

          ytrain,

          batch_size = 70,

          epochs = 50,

          verbose = 1,

          validation_data = (xval, yval))
#plot of validation vs training accuracy over the epochs

import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training', 'validation'])

plt.title('Loss')

plt.xlabel('Epoch')
ypred = model.predict(xtest)



evaluation = model.evaluate(xtest, ytest)

print('Test accuracy : {:.3f}'.format(evaluation[1]))
for index in range(len(ypred)):

    if ypred[index] > 0.5:

        ypred[index] = 1

    else:

        ypred[index] = 0





from sklearn.metrics import confusion_matrix

import seaborn as sns



cm = confusion_matrix(ytest, ypred)

plt.figure(figsize = (14,10))

sns.heatmap(cm, annot = True)
from sklearn.metrics import classification_report

 

num_classes = 2

target_names = ['Class {}'.format(i) for i in range(num_classes)]



print(classification_report(ytest, ypred, target_names = target_names))
model.save('../model.h5')
download('model.h5')