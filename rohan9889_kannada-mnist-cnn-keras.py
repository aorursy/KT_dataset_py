import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

y_train = train_data['label']

X = train_data.drop(['label'],axis=1)

del train_data
Id = test_data['id']
test_data = test_data.drop(['id'],axis=1)
X.isnull().all().unique()
y_train.isnull().any()
test_data.isnull().all().unique()
label_val = y_train.value_counts()

plt.figure(figsize=(12,6))

sns.barplot(x=label_val.index,y=label_val.values)
X_temp = X.values.reshape(X.shape[0], 28, 28)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

num_classes = len(classes)

samples_per_class = 6

plt.figure(0,figsize=(16,10))

for y, cls in enumerate(classes):

    idxs = np.flatnonzero(y_train == y)

    idxs = np.random.choice(idxs, samples_per_class, replace=False)

    for i, idx in enumerate(idxs):

        plt_idx = i * num_classes + y + 1

        plt.subplot(samples_per_class, num_classes, plt_idx)

        plt.imshow(X_temp[idx])

        plt.axis('off')

        if i == 0:

            plt.title(cls)

plt.show()
X = X.values.reshape(X.shape[0], 28, 28,1)

test_data = test_data.values.reshape(test_data.shape[0], 28, 28,1)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

        rotation_range= 8,  

        zoom_range = 0.12,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

datagen.fit(X)
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train,num_classes=10)



from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('BWeight.md5',monitor='val_loss',

                            save_best_only=True)
model = Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(MaxPool2D())

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
from sklearn.model_selection import train_test_split

X_train, X_val1, y_train, y_val1 = train_test_split(

    X, y_train, test_size=0.05, random_state=42)
size_batch = 64
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=size_batch),

                              epochs = 60,

                              validation_data = (X_val1,y_val1),

                              verbose = 2,

                              steps_per_epoch = X_train.shape[0] // size_batch,

                              callbacks=[checkpoint])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
model.load_weights('BWeight.md5')
extra_validation = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
y_extra_validate = extra_validation['label']

X_extra_validate = extra_validation.drop(['label'],axis=1)
X_extra_validate = X_extra_validate.values.reshape(X_extra_validate.shape[0], 28, 28,1)
from sklearn.metrics import classification_report
print(classification_report(y_extra_validate,model.predict_classes(X_extra_validate)))
FINAL_PREDS = model.predict_classes(test_data)
submission = pd.DataFrame({ 'id': Id,

                            'label': FINAL_PREDS })

submission.to_csv(path_or_buf ="Kannada_MNIST_KERAS.csv", index=False)