import tensorflow as tf

import pandas as pd

import numpy as np

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.datasets import fashion_mnist

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt

import seaborn as sbn

%matplotlib inline
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train.shape, x_test.shape
## Adding color dimention to data

x_train = np.expand_dims(x_train, -1)

x_test = np.expand_dims(x_test, -1)

x_train.shape, x_test.shape


K = len(set(y_train))      ###### Number Of Labels

print(K)

sbn.countplot(y_train)

 
# Spliting X_train Set into training set and validation test

x_train, val_x, y_train, val_y = train_test_split(x_train, y_train, test_size=0.20)
es = EarlyStopping(monitor='loss', patience=12)

filepath="/kaggle/working/bestmodel.h5"

md = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# defininig ImageDataGeneratore to increase data

datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
# Important Variables

epochs = 20

batch_size = 128

input_shape = (28, 28, 1)

adam = tf.keras.optimizers.Adam(0.001)




i = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)

x = BatchNormalization()(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)



# x = GlobalMaxPooling2D()(x)

x = Flatten()(x)

x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(K, activation='softmax')(x)



model = Model(i, x)

model.summary()

# Compiling Model

model.compile(optimizer=adam,

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
# Fit Model

History = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),

          epochs = epochs,

          validation_data = (val_x, val_y),

          callbacks = [es,md],

          shuffle= True

        )
# Plot loss per iteration

import matplotlib.pyplot as plt

plt.plot(History.history['loss'], label='loss')

plt.plot(History.history['val_loss'], label='val_loss')

plt.legend()
# Plot accuracy per iteration

plt.plot(History.history['accuracy'], label='acc')

plt.plot(History.history['val_accuracy'], label='val_acc')

plt.legend()
model1 = load_model(filepath)

model1.summary()
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

  print(cm)



  plt.imshow(cm, interpolation='nearest', cmap=cmap)

  plt.title(title)

  plt.colorbar()

  tick_marks = np.arange(len(classes))

  plt.xticks(tick_marks, classes, rotation=45)

  plt.yticks(tick_marks, classes)



  fmt = 'd'

  thresh = cm.max() / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

      plt.text(j, i, format(cm[i, j], fmt),

               horizontalalignment="center",

               color="white" if cm[i, j] > thresh else "black")



  plt.tight_layout()

  plt.ylabel('True label')

  plt.xlabel('Predicted label')

  plt.show()





p_test = model1.predict(x_test).argmax(axis=1)

cm = confusion_matrix(y_test, p_test)

plot_confusion_matrix(cm, list(range(10)))