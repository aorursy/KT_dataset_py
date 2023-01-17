# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau



from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
input_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
input_data = input_df.drop(['label'], axis=1).values / 255.0

input_labels = input_df['label']

test_data = test_df.values / 255.0



# Split the train and the validation set for the fitting

train_data, valid_data, train_labels, valid_labels = train_test_split(input_data, input_labels, test_size = 0.15, random_state=2)



train_data = train_data.reshape(-1,28,28,1)

valid_data = valid_data.reshape(-1,28,28,1)

test_data = test_data.reshape(-1,28,28,1)



train_labels = to_categorical(train_labels, 10)

valid_labels = to_categorical(valid_labels, 10)

print("Train: ",train_data.shape, train_labels.shape)

print("Valid: ",valid_data.shape, valid_labels.shape)

print("Test: ",test_data.shape)
data_augment = ImageDataGenerator(rotation_range=10,

                                 width_shift_range=0.1,

                                 height_shift_range=0.1,

                                 zoom_range=0.1)
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(10, activation='softmax')

])

model.summary()
# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

annealer = ReduceLROnPlateau(

    monitor='val_acc', 

    patience=3, 

    verbose=1, 

    factor=0.5, 

    min_lr=0.00001)
optim = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])



num_epochs = 30

batch_size = 86

history = model.fit_generator(

    data_augment.flow(train_data, train_labels, batch_size=batch_size),

    steps_per_epoch=train_data.shape[0]//batch_size,

    validation_data=(valid_data, valid_labels),

    epochs=num_epochs,

    callbacks=[annealer]

)
loss = history.history['loss']

dev_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



from matplotlib import pyplot as plt

plt.plot(epochs, loss, 'bo', label='training loss')

plt.plot(epochs, dev_loss, 'b', label='validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
predictions = model.predict(test_data)

pred_list = []

for index, pred in enumerate(predictions):

    pred_list.append({"ImageId": index+1, "Label": np.argmax(pred)})

sub_df = pd.DataFrame(pred_list)

sub_df.to_csv("submission.csv", index=False)