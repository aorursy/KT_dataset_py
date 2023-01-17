# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import sklearn.model_selection

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

print(tf.__version__)
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
batch_size = 100

num_class = 10

epochs = 45



img_rows = 28

img_cols = 28



input_shape = (img_rows, img_cols, 1)
y = train["label"]

x = train.drop(["label"], axis = 1)



x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x.values, y.values, test_size = 0.10)



print(y_train.shape)

print(x_train.shape)

print(y_val.shape)

print(x_val.shape)
print(x_train.shape[0])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')

x_train /= 255



print(x_val.shape[0])

x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

x_val = x_val.astype('float32')

x_val /= 255
y_train = tf.keras.utils.to_categorical(y_train, num_class)

y_val = tf.keras.utils.to_categorical(y_val, num_class)



k_init = tf.initializers.TruncatedNormal(mean = 0.1, stddev = 0.05)

b_init = tf.initializers.constant(value = 1e-4)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rotation_range=15,

    zoom_range = 0.20,

    width_shift_range=0.20,

    height_shift_range=0.20

)




model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape = input_shape),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation='relu', strides=2, padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    

    tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation='relu', strides=2, padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(num_class, activation='softmax')

])
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.002,

                    rho=0.9,

                    momentum=0.1,

                    epsilon=1e-07,

                    centered=True,

                    name='RMSprop')
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',

                                          factor=0.25,

                                          patience=2,

                                          verbose=1,

                                          mode="auto",

                                          min_delta=0.0001,

                                          cooldown=0,

                                          min_lr=0.00001)

es = tf.keras.callbacks.EarlyStopping(monitor='loss', 

                   mode='min', verbose=1,

                   patience=300,

                   restore_best_weights=False )


model.compile(loss= "categorical_crossentropy",  optimizer = optimizer, metrics=['accuracy'])

model.summary()
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),

                              epochs = epochs,

                              steps_per_epoch = 100,

                              validation_data = (x_val, y_val),

                              validation_steps=50,

                              callbacks=[lr, es],

                              verbose=2)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
scores = model.evaluate(x_val, y_val)
test=pd.read_csv('../input/Kannada-MNIST/test.csv')



test_id=test.id



test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)
test.shape
y_pre=model.predict(test)     ##making prediction

y_pre=np.argmax(y_pre,axis=1)
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')



sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)

sample_sub.head()
os.chdir("..")

os.listdir()