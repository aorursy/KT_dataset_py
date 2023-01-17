import tensorflow as tf

import pandas as pd

import numpy as np

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sbn

%matplotlib inline
X = pd.read_csv('../input/digit-recognizer/train.csv')

X_test = pd.read_csv('../input/digit-recognizer/test.csv')

X.shape, X_test.shape
X_train = X.drop(['label'],1)

Y_train = X['label']

X_train.shape
K = len(set(Y_train))      ###### Number Of Labels

print(K)

sbn.countplot(Y_train)
#  Reshaping Data

X_train = np.asarray(X_train)

X_test = np.asarray(X_test)

X_train.shape, X_test.shape
# Normalizing and Reshaping Data

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train = X_train/255

X_test = X_test/255

X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1,28 ,28, 1)

X_train.shape, X_test.shape

# one hot encoding y data

Y_train= tf.keras.utils.to_categorical(Y_train, 10)

Y_train.shape
# Spliting X_train Set into training set and validation test

x_train, val_x, y_train, val_y = train_test_split(X_train, Y_train, test_size=0.20)
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 15)
es = EarlyStopping(monitor='loss', patience=12)

filepath="/kaggle/working/bestmodel.h5"

md = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# Important Variables

epochs = 30

num_classes = 10

batch_size = 128

input_shape = (28, 28, 1)

adam = tf.keras.optimizers.Adam(0.001)
i = Input(shape=input_shape)

x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)

x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)

#x = Conv2D(32, (3, 3), strides=2, activation='relu')(x)

x = Flatten()(x)

x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(K, activation='softmax')(x)



model = Model(i, x)

model.summary()
# Compiling Model

model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# Fit Model

History = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

          epochs = epochs,

          validation_data = (val_x, val_y),

          callbacks = [es,md],

          shuffle = True

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
## Loading Model and Making Prediction





model1 = load_model("/kaggle/working/bestmodel.h5")

model1.summary()
pred = model1.predict(X_test)

pred_class = np.argmax(pred,axis=1)







submissions=pd.DataFrame({"ImageId": list(range(1,len(pred_class)+1)),

                         "Label": pred_class})

submissions.to_csv("submissions.csv", index=False, header=True)

submissions
