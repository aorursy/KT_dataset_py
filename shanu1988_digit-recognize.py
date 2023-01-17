# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv').values

test[:5]
print("shape of the train dataset:", train.shape)

print("Shape of the test dataset:", test.shape)
X = train.drop(['label'], axis='columns').values

y = train['label'].values



#Split data into train and validation

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
X=X.reshape(len(X),28,28,-1)



X_train, X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2)
from keras.preprocessing.image import ImageDataGenerator

data_generation = ImageDataGenerator(

                featurewise_center=False,

                samplewise_center=False,

                featurewise_std_normalization=False,

                samplewise_std_normalization=False,

                zca_whitening=False,

                rotation_range=10,

                zoom_range=0.1,

                width_shift_range=0.1,

                height_shift_range=0.1,

                horizontal_flip=False,

                vertical_flip=False)



data_generation.fit(X_train)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.5 ** x)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(32,(3,3), padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64,(3, 3), padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),#######

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

    ])



optimizer = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



%%time

batch_size = 64

epochs = 5

history = model.fit_generator(data_generation.flow(X_train, y_train, batch_size = batch_size), epochs = epochs, 

                              validation_data = (X_valid, y_valid), verbose=2, 

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks = [reduce_lr])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('loss')

plt.legend(['Train', 'Valid'])

plt.show();
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Valid'])

plt.show()
model.evaluate(X_valid, y_valid)
from sklearn.metrics import accuracy_score

y_preds = model.predict(X_valid)

y_preds=[np.argmax(pred) for pred in y_preds]

accuracy_score(y_preds, y_valid)
test_df = test.reshape(len(test),28, 28, -1)

y_test = model.predict((test_df)/255.0)

#y_test = model.predict(test_df)

y_test = [np.argmax(pred) for pred in y_test]
    

submission=pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submission["Label"]=y_test

submission.head()
submission.head()
submission.to_csv("final22.csv",index=False,float_format='%.4f')