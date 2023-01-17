import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import os
print(os.listdir("../input"))
train_file =  "../input/train.csv"
test_file =  "../input/test.csv"
output_file = "digit-recognizer_submission.csv"
train = pd.read_csv(train_file).values
x_train, x_val, y_train, y_val = train_test_split(
    train[:,1:], train[:,0], test_size=0.1)
fig, ax = plt.subplots(2, 1, figsize=(12,6))
ax[0].plot(x_train[0])
ax[0].set_title('784x1 data')
ax[1].imshow(x_train[0].reshape(28,28), cmap='gray')
ax[1].set_title('28x28 data')
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(y_train[0])
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer = Adam(lr=1e-4), metrics=["accuracy"])


epochs = 30
batch_size = 64
validation_steps = 10000

datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)

annealer = ReduceLROnPlateau(monitor='val_acc', patience=1, verbose=2, 
                             factor=0.5, min_lr=0.0000001)
#early_stopping = EarlyStopping(monitor='val_loss', patience=10)

train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = datagen.flow(x_val, y_val, batch_size=batch_size)

hist = model.fit_generator(train_generator,
                           steps_per_epoch=x_train.shape[0]//batch_size,
                           epochs=epochs,
                           validation_data=val_generator,
                           validation_steps=validation_steps//batch_size,
                           callbacks=[annealer])
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()
y_hat = model.predict(x_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)
test = pd.read_csv(test_file).values
x_test = test.astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_hat,axis=1)
y_pred
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series(y_pred,name="Label")],axis = 1)
submission.to_csv("submission.csv",index=False)
submission