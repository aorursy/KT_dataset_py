# Linear algebra

import numpy as np



# Data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd



# Deep Learning

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.utils import np_utils



# Visualization

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.describe()
x_train = train_df.drop('label', axis=1).values

y_train = train_df['label'].values



x_test = test_df.values
img_height, img_width = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)

x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
num_classes = 10

y_train = np_utils.to_categorical(y_train, num_classes)
y_train[:5]
x_test = x_test / 255.0

x_train = x_train / 255.0
model = Sequential()



model.add(Conv2D(

    filters=32,

    kernel_size=(5,5),

    input_shape=(img_height, img_width, 1), 

    padding="Same",

    activation="relu"

))

model.add(Conv2D(

    filters=32,

    kernel_size=(5,5),

    input_shape=(img_height, img_width, 1), 

    padding="Same",

    activation="relu"

))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(

    filters=64,

    kernel_size=(3,3),

    input_shape=(img_height, img_width, 1), 

    padding="Same",

    activation="relu"

))

model.add(Conv2D(

    filters=64,

    kernel_size=(3,3),

    input_shape=(img_height, img_width, 1), 

    padding="Same",

    activation="relu"

))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
checkpoint = ModelCheckpoint(

    'best_model.hdf5',

    monitor='val_acc',

    save_best_only=True,

    verbose=1

)
lr_reduction = ReduceLROnPlateau(

    monitor='val_acc',

    patience=3,

    factor=0.5,

    min_lr=0.00001,

    verbose=1

)
batch_size = 96

epochs = 40

validation_size = 0.3
history = model.fit(

    x=x_train, 

    y=y_train, 

    batch_size=batch_size, 

    epochs=epochs,

    verbose=1,

    validation_split=validation_size,

    callbacks=[checkpoint, lr_reduction]

)
plt.figure(figsize=(16,9))

plt.plot(history.history['acc'], label='Train')

plt.plot(history.history['val_acc'], label='Test')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
model.load_weights('best_model.hdf5')
probabilities = model.predict(x_test)
predictions = np.argmax(probabilities, axis=1)
submission = pd.DataFrame(data={

    'ImageId': list(range(1, predictions.shape[0]+1)),

    'Label': predictions

})
submission.head()
submission.to_csv('submission.csv', index=False)