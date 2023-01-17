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
%matplotlib inline

from keras.models import Sequential, load_model

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers import LeakyReLU

from keras.callbacks import ModelCheckpoint,History,EarlyStopping,LearningRateScheduler

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam, Adadelta, RMSprop

from keras.utils import to_categorical

import matplotlib.pyplot as plt



from sklearn import metrics
train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

print(train_data.shape)
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

print(test_data.shape)
dig_data = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

print(dig_data.shape)
train_data.head()
test_data.head()
dig_data.head()
validate_data = train_data[55000:]

# find label column

train_label = np.float32(train_data.label)

validate_label = np.float32(validate_data.label)

test_label = np.float32(test_data.id)

dig_label = np.float32(dig_data.label)



# find image values 

train_image = np.float32(train_data[train_data.columns[1:]])

validate_image = np.float32(validate_data[validate_data.columns[1:]])

test_image = np.float32(test_data[test_data.columns[1:]])

dig_image = np.float32(dig_data[dig_data.columns[1:]])
print('train_data shape:%s' %str(train_data.shape))

print('validate_data shape:%s' %str(validate_data.shape))

print('train_label shape:%s' %str(train_label.shape))

print('validate_label shape:%s' %str(validate_label.shape))

print('test_label shape:%s' %str(test_label.shape))

print('dig_label shape:%s' %str(dig_label.shape))

print('train_image shape:%s' %str(train_image.shape))

print('validate_image shape:%s' %str(validate_image.shape))

print('test_image shape:%s' %str(test_image.shape))

print('dig_image shape:%s' %str(dig_image.shape))
datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range = 10,

    horizontal_flip = False,

    zoom_range = 0.15)
# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(sparse=False,categories='auto')

# yy = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

# encoder.fit(yy)



# # transform

# train_label = train_label.reshape(-1,1)

# validate_label = validate_label.reshape(-1,1)



# dig_label = dig_label.reshape(-1,1)



# train_label_transform = encoder.transform(train_label)

# validate_label_transform = encoder.transform(validate_label)



# dig_label_transform = encoder.transform(dig_label)



# print('train_label_transform shape: %s'%str(train_label_transform.shape))

# print('validate_label_transform shape: %s'%str(validate_label_transform.shape))

# print('dig_label_transform shape: %s'%str(dig_label_transform.shape))
train_label_transform = to_categorical(train_data.iloc[:,0])

validate_label_transform = to_categorical(validate_data.iloc[:,0])

dig_label_transform = to_categorical(dig_data.iloc[:,0])



print('train_label_transform shape: %s'%str(train_label_transform.shape))

print('validate_label_transform shape: %s'%str(validate_label_transform.shape))

print('dig_label_transform shape: %s'%str(dig_label_transform.shape))
n_row = 1

n_col = 10



plt.figure(figsize=(13,12))

for i in list(range(n_row * n_col)):

    offset =0

    plt.subplot(n_row, n_col, i+1)

    plt.imshow(train_image[i].reshape(28,28))

    title_text = 'Eigenvalue ' + str(i + 1)

    plt.title(title_text, size=6.5)

    plt.xticks(())

    plt.yticks(())

plt.show()
train_image = train_image / 255.0

validate_image = validate_image / 255.0

test_image = test_image / 255.0

dig_image = dig_image / 255.0



train_image_reshape = train_image.reshape(train_image.shape[0],28,28,1)

validate_image_reshape = validate_image.reshape(validate_image.shape[0],28,28,1)

test_image_reshape = test_image.reshape(test_image.shape[0],28,28,1)

dig_image_reshape = dig_image.reshape(dig_image.shape[0],28,28,1)



print('train_image_reshape shape %s' %str(train_image_reshape.shape))

print('validate_image_reshape shape %s' %str(validate_image_reshape.shape))

print('test_image_reshape shape %s' %str(test_image_reshape.shape))

print('dig_image_reshape shape %s' %str(dig_image_reshape.shape))
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1),padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(128, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(256, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(256,kernel_regularizer=regularizers.l2(0.02)))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.3))

model.add(Dense(256,kernel_regularizer=regularizers.l2(0.02)))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.3))

model.add(Dense(512,kernel_regularizer=regularizers.l2(0.02)))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))



model.summary()
BATCH_SIZE = 512

EPOCHS = 60

#EPOCHS = 5
model.compile(loss='categorical_crossentropy',optimizer=Adadelta(),metrics=['accuracy'])

# fit data

datagen.fit(train_image_reshape)
# training

history = model.fit_generator(datagen.flow(train_image_reshape,train_label_transform, batch_size=BATCH_SIZE),

                              epochs = EPOCHS,

                              shuffle=True,

                              validation_data = (validate_image_reshape,validate_label_transform),

                              verbose = 1,

                              steps_per_epoch=train_image_reshape.shape[0] // BATCH_SIZE)
pred_dig = history.model.predict_classes(dig_image_reshape)

print(metrics.accuracy_score(pred_dig, np.argmax(dig_label_transform, axis = 1)))

print(metrics.accuracy_score(pred_dig, dig_label))
validate_labels = []

for i in validate_label_transform:

    for j, val in enumerate(i):

        if val == 0.:

            pass

        else:

            validate_labels.append(j)
pred_validate = history.model.predict_classes(validate_image_reshape)

metrics.accuracy_score(pred_validate, np.array(validate_labels))
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
pred_test = history.model.predict_classes(test_image_reshape)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = pred_test
submission.head(10)
submission.to_csv("submission.csv",index=False)