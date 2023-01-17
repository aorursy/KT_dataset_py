import numpy as np

import pandas as pd

import time



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

 



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(context='notebook',

        style='whitegrid',

        palette='deep',

        font='sans-serif',

        font_scale=1,

        color_codes=True,

        rc=None)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train_dataset.head(3)
test_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_dataset.head(3)
X_train, y_train = np.array(train_dataset.iloc[:,1:]), np.array(train_dataset.label)

test_data = np.array(test_dataset)
print(X_train.shape)

print(y_train.shape)

print(test_data.shape)
num_classes = len(np.unique(y_train))

X_train = X_train.reshape(42000, 28,28, 1)

y_train = (y_train).astype(int).reshape(42000, 1)





test_data = test_data.reshape(28000, 28, 28, 1)



y_train = keras.utils.to_categorical((y_train).astype(int), 10)
val_test = int(X_train.shape[0] * 0.20)

X_test = X_train[:val_test]

X_train = X_train[val_test:]

y_test = y_train[:val_test]

y_train = y_train[val_test:]
print('No. of classes = '+str(num_classes))

print('\n')

print('Train Data: '+str(X_train.shape))

print('Val Data: '+str(X_test.shape))

print('\n')

print('Train Label: '+str(y_train.shape))

print('Val Label: '+str(y_test.shape))

print('\n')

print('Test Data: '+str(test_data.shape))
plt.imshow(X_train[100].reshape(28,28))

plt.show()
input_shape = X_train[0].shape



model = Sequential(name="Conv2D_Model")



model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu'))



model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(1000, activation="relu"))

model.add(Dense(1000, activation="relu"))

model.add(Dense(1000, activation="relu"))

model.add(Dense(1000, activation="relu"))

model.add(Dense(1000, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
keras.utils.plot_model(model, show_shapes=True, rankdir="TB",expand_nested=True, dpi=100)
from tensorflow.keras.callbacks import EarlyStopping



early_stop = EarlyStopping(monitor='val_loss',

                        min_delta=0,

                        patience=30,

                        verbose=1,  

                        mode='auto',

                        baseline=None,  

                                               

                        restore_best_weights=False)
hist = model.fit(X_train, y_train,

            batch_size=50,

            epochs=500,

            verbose=1,

            callbacks=[early_stop],

            validation_split=0.2,

            validation_data=[X_train, y_train],

            shuffle=True,

            class_weight=None,

            sample_weight=None,

            initial_epoch=0,

            steps_per_epoch=None,

            validation_steps=None,

            validation_batch_size=None,

            validation_freq=1,

            max_queue_size=10,

            workers=1)
print(hist.history.keys())
performance = pd.DataFrame(model.history.history)
print(min(performance.loss))

print(min(performance.val_loss))

print('\n')

print(max(performance.accuracy))

print(max(performance.val_accuracy))

plt.rcParams["figure.dpi"] = 100

performance.plot(figsize=(10,5))

plt.title('Losses')

plt.show()
y_pred = model.predict_classes(test_data)



predictions=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),

                         "Label": y_pred})


