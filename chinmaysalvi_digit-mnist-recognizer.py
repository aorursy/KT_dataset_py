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
train_url = '/kaggle/input/digit-recognizer/train.csv'

test_url = '/kaggle/input/digit-recognizer/test.csv'

train = pd.read_csv(train_url)

test = pd.read_csv(test_url)
print(train.shape,"\n",test.shape)
train.head(3)
X = train.drop('label', axis=1).astype('float32')

y = train['label']

X_sub = test.astype('float32')
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras

from keras import Sequential,layers,callbacks,preprocessing
X = X.values.reshape(-1,28,28,1)

X_sub = X_sub.values.reshape(-1,28,28,1)

y = keras.utils.to_categorical(y, 10)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, shuffle=True)
def spatial_cnn():

    model = Sequential([

        layers.experimental.preprocessing.Rescaling(scale=1.0/255.0, input_shape=(28,28,1)),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Conv2D(32, (5,5), (2,2), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Conv2D(64, (5,5), (2,2), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),

        layers.BatchNormalization(),

        layers.SpatialDropout2D(0.25),

        layers.Flatten(),

        layers.Dense(300, activation="relu"),

        layers.BatchNormalization(),

        layers.Dropout(0.4),

        layers.Dense(10, activation='softmax')],

        name='spatial_cnn')

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    model.summary()

    return model



model = spatial_cnn()
train_datagen = preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1,

                                                       height_shift_range=0.1, shear_range=0.1,

                                                       zoom_range=0.1)



train_datagen.fit(X_train)
batch_size = 64

epochs = 45

steps = X_train.shape[0]//batch_size

history = dict()

flow = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

changed = False



class cbs(callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        global changed

        if logs['val_loss']<=0.0205 and not changed:

            changed = True

            print("\nChanging LR\n")

            global reduce_lr

            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2, min_lr=1.0e-10)

            

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1.0e-10)

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto")

model_hist = model.fit_generator(flow, validation_data=(X_test, y_test), verbose=1, shuffle=True,

                               steps_per_epoch=steps, epochs=epochs, callbacks=[cbs(), reduce_lr, early_stop])
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.plot(model_hist.history['loss'],'r*-')

plt.plot(model_hist.history['val_loss'],'b*-')

plt.title('Training and Validation Losses')

plt.legend(labels=['Loss','Validation Loss'])

plt.subplot(1,2,2)

plt.plot(model_hist.history['accuracy'],'mo-')

plt.plot(model_hist.history['val_accuracy'],'co-')

plt.title('Training and Validation Accuracy')

plt.legend(labels=['Accuracy','Validation Accuracy'])

plt.show()
preds_sub = pd.DataFrame(data={"ImageId":list(range(1,X_sub.shape[0]+1)),"Label":(model.predict_classes(X_sub))}).astype(int)
preds_sub.head()
preds_sub.to_csv("DigitRecog.csv", index=False, header=True)