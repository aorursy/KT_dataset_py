# first load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
# load data

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
# split our data into features & target

X_train = df_train.drop('label', axis=1).values

y_train = df_train['label'].values.reshape(-1,1)



X_test = df_test.values
# rescale variables

X_train = X_train.astype('float32')/255.0

X_test = X_test.astype('float32')/255.0
# check first few images

plt.figure(figsize=(15,15))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(X_train[i].reshape(28,28), cmap='gray')

    plt.title('Number:' + str(y_train[i][0]))

    plt.axis('off')
# reshape features for tensorflow

X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)



# one hot encode for target variable

y_train = to_categorical(y_train)

target_count = y_train.shape[1]
# image augmentation 

datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=10,

    zoom_range = 0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False)



# fit generator on our train features

datagen.fit(X_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(BatchNormalization())



model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(BatchNormalization())



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(target_count, activation='softmax'))





optimizer = RMSprop(learning_rate=0.001,rho=0.99)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1,patience=2, min_lr=0.00000001)



callback = EarlyStopping(monitor='loss', patience=5)

history = model.fit(datagen.flow(X_train,y_train, batch_size=64), epochs = 50, validation_data=(X_val, y_val), verbose = 1, callbacks=[reduce_lr, callback])
# prepare data for evaluation

y_val_m = y_val.argmax(axis=1)

y_val_hat_prob = model.predict(X_val)

y_val_hat = y_val_hat_prob.argmax(axis=1)

X_val_inc = X_val[y_val_m != y_val_hat, :, :, :]

y_val_inc = y_val_m[y_val_m != y_val_hat]

y_val_hat_inc = y_val_hat[y_val_m != y_val_hat]

y_val_hat_prob_inc = y_val_hat_prob[y_val_m != y_val_hat]
plt.figure(figsize=(15,15))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.imshow(X_val_inc[i, :, :, :].reshape(28,28), cmap='gray')

    plt.axis('off')

    plt.title('Actual: {}; Predicted: {}'.format(y_val_inc[i], y_val_hat_inc[i]))
for i in range(0,10):

    act = y_val_inc[i]

    pred = y_val_hat_inc[i]

    print('Actual: {}; Confidence (act/pred): \t{} - {:.0f}%  \t{} - {:.0f}%'.format(act, act, y_val_hat_prob_inc[i][act]*100, pred, y_val_hat_prob_inc[i][pred]*100))
# predict our test data

y_test_hat = model.predict(X_test).argmax(axis=1)



df_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

df_submission['Label'] = y_test_hat.astype('int32')

df_submission.to_csv('Submission.csv', index=False)

print('Submission saved!')