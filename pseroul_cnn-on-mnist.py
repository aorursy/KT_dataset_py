import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print('Train:', train.shape)

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print('Test:', test.shape)

X_train = (train.iloc[:,1:].values)

y_train = train.iloc[:,0].values

X_test = test.values

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

print('X_train:', X_train.shape)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

print('X_test:', X_test.shape)

from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)

print('y_train:', y_train.shape)
from sklearn.model_selection import train_test_split

from keras.preprocessing import image



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

print('train size: ', X_train.shape)

print('val size: ', X_val.shape)

gen =ImageDataGenerator(rescale=1/255., rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                               height_shift_range=0.08, zoom_range=0.08)



val_gen = image.ImageDataGenerator(rescale=1/255.)



batches = gen.flow(X_train, y_train, batch_size=64)

val_batches = val_gen.flow(X_val, y_val, batch_size=64)
model = Sequential([

    Convolution2D(32,(3,3), activation='relu', input_shape=(28,28,1)),

    BatchNormalization(axis=1),

    Convolution2D(32,(3,3), activation='relu'),

    MaxPooling2D(),

    BatchNormalization(axis=1),

    Convolution2D(64,(3,3), activation='relu'),

    BatchNormalization(axis=1),

    Convolution2D(64,(3,3), activation='relu'),

    MaxPooling2D(),

    Flatten(),

    BatchNormalization(),

    Dense(512, activation='relu'),

    BatchNormalization(),

    Dense(10, activation='softmax')

    ])

model.summary()

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=5, validation_data=val_batches, validation_steps=val_batches.n)
acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc)) # Get number of epochs



plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()



plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )
model.evaluate_generator(val_batches)
test_batches = val_gen.flow(X_test, shuffle=False, batch_size=1)

predictions = model.predict_generator(test_batches, steps=X_test.shape[0])
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": np.argmax(predictions, axis=1)})

print(submissions)

submissions.to_csv("submission.csv", index=False, header=True)