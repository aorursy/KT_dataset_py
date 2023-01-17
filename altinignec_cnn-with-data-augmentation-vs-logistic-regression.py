import pandas as pd

import numpy as np
training_set = pd.read_csv('../input/train.csv')
y_train = np.array(training_set['label'])
# Scale the input to be between 0 and 1

X_train = training_set.drop(['label'], axis=1).values/255
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
print(X_train.shape)

print(X_valid.shape)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver='lbfgs', n_jobs=-1, multi_class='multinomial').fit(X_train, y_train)

y_pred = log_reg.predict(X_valid)
from sklearn.metrics import accuracy_score



print(accuracy_score(y_pred, y_valid))
X_test = pd.read_csv('../input/test.csv').values/255

X_test.shape
y_test = log_reg.predict(X_test)

res = pd.concat([pd.DataFrame(list(range(1,X_test.shape[0]+1))), pd.DataFrame(y_test)], axis=1)

res.to_csv('output.csv', header=['ImageId', 'Label'], index=False)
import keras

from keras.utils import to_categorical
X_train = X_train.reshape(-1,28,28,1)

X_valid = X_valid.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)
y_train = to_categorical(y_train)

y_valid = to_categorical(y_valid)
print(X_train.shape)

print(y_train.shape)

print(X_valid.shape)

print(y_valid.shape)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D



model = Sequential()

model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



# w/o Data Augmentation



#model.fit(X_train, y_train,

#          batch_size=512,

#          epochs=3,

#          verbose=1,

#          validation_data=(X_valid, y_valid))
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import math
datagen = ImageDataGenerator(

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    zca_epsilon=1e-6,

    rotation_range=5.,

    width_shift_range=.08,

    height_shift_range=.08,

    shear_range=30*(math.pi/180),

    zoom_range=0.08,

    data_format="channels_last"

)



datagen.fit(X_train)



model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),

                    steps_per_epoch=len(X_train) / 64, 

                    epochs=5,

                    validation_data=(X_valid, y_valid))
plt.plot(model.history.epoch, model.history.history['loss'], label='Training Loss')

plt.plot(model.history.epoch, model.history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')
y_test = model.predict(X_test)
y_test = np.argmax(y_test,axis = 1)
res = pd.concat([pd.DataFrame(list(range(1,X_test.shape[0]+1))), pd.DataFrame(y_test)], axis=1)

res.to_csv('output.csv', header=['ImageId', 'Label'], index=False)