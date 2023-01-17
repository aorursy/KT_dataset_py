import numpy as np

import pandas as pd

import seaborn as sns

import os

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

print("TF version ", tf.__version__)

from tensorflow import keras as kr

from sklearn.model_selection import StratifiedKFold



print(os.listdir("../input"))

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
Y_train = train_df['label'] # keep labels

X_train = train_df.iloc[:,1:train_df.shape[1]].values # dataframe to numpy array

X_test = test_df.values # dataframe to numpy array
X_train = X_train.astype(float) / 255.

X_test = X_test.astype(float) / 255.



print("Current Shape => X_train: %s | Y_train %s | X_test %s" % (str(X_train.shape), str(Y_train.shape), str(X_test.shape) ))
X_train = X_train.reshape(-1,28, 28, 1)

X_test = X_test.reshape(-1,28, 28, 1)

print("After Reshape => X_train: %s | X_test %s" % (str(X_train.shape), str(X_test.shape)))



plt.imshow(X_train[25].reshape(28,28)) # let's see a sample of the Data
""" Use Keras image data generator """



AUGMENTED_SAMPLES = 10000 # how many new samples

original_size = X_train.shape[0] # training samples before augmentation



image_generator = kr.preprocessing.image.ImageDataGenerator(

        rotation_range=22, zoom_range = 0.05, width_shift_range=0.05,

        height_shift_range=0.05, horizontal_flip=False, vertical_flip=False, 

        data_format="channels_last", zca_whitening=False, featurewise_center=True, featurewise_std_normalization=True)



image_generator.fit(X_train, augment=True) # fit generator on training data



augm = np.random.randint(original_size, size=AUGMENTED_SAMPLES) # get random samples from the original dataset

X_augmented = X_train[augm].copy()

Y_augmented = Y_train[augm].copy()

X_augmented = image_generator.flow(X_augmented, np.zeros(AUGMENTED_SAMPLES), batch_size=AUGMENTED_SAMPLES, shuffle=False).next()[0]



# append new data to our already existing train set

X_train = np.concatenate((X_train, X_augmented))

Y_train = np.concatenate((Y_train, Y_augmented))



print('New Trainset Size: X %s - Y %s' % (str(X_train.shape), str(Y_train.shape)))



""" Let's take a look at an augmented sample and the original """

im = 0 # select image

print('Original Image')

plt.imshow(X_train[augm[im]].reshape(28,28))

plt.show()

print('Augmented Image')

plt.imshow(X_train[original_size+im].reshape(28,28))

plt.show()
"""

Labels to hot encoded for cross-entropy



[1 , 5 , ...]  -> [[0,1,0,0,0,0 .. ], [0,0,0,0,0,1,0,0 ...]]

"""

Y_train_cat = kr.utils.to_categorical(Y_train, num_classes=10)



print("Categorical Y shape:  %s " % str(Y_train_cat.shape) )
model = kr.models.Sequential()



model.add(kr.layers.Conv2D(64, kernel_size= (3,3), activation='relu', input_shape=(28,28,1)))

model.add(kr.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones'))

model.add(kr.layers.MaxPooling2D(pool_size = (2,2)))

model.add(kr.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))

model.add(kr.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones'))

model.add(kr.layers.MaxPooling2D(pool_size = (2,2)))

model.add(kr.layers.Flatten())

model.add(kr.layers.Dense(64, activation='relu'))

model.add(kr.layers.Dense(10, activation='softmax'))



opt = kr.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # Adam Optimizer

# early_stop = kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto', baseline=None)
BATCH_SIZE = 128

EPOCHS = 50

LOGS = 2

VALIDATION_SPLIT = 0.15



model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,Y_train_cat, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=LOGS, shuffle=False, class_weight=None, sample_weight=None) # train
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
res = model.predict(X_test)

res = res.argmax(axis=1)



print('Prediction Label Distribution')

g = sns.countplot(Y_train)

plt.show()
for i in range(0,10):

    print('Predicted', res[i])

    plt.imshow(X_test[i].reshape(28,28))

    plt.show()
out = pd.Series(res,name="Label")

out = pd.concat([pd.Series(range(1,28001),name = "ImageId"),out],axis = 1)

out.head()

out.to_csv('keras_cnn_results.csv',index=False, sep=',')