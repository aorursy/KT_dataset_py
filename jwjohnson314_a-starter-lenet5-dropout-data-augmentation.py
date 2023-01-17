# imports
import numpy as np
import pandas as pd  # for csv files
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

% matplotlib inline
# load the data
xtr = np.load('../input/xtr.npy')
xte = np.load('../input/xte.npy')
ytr = np.load('../input/ytr.npy')
# plot some images
fig, ax = plt.subplots(2, 5, figsize=(16, 6))
for i in range(5):
    for j in range(2):
        idx = np.random.randint(low=0, high=60000, size=1)[0]
        ax[j, i].imshow(xtr[idx, :, :], cmap='gray')
        ax[j, i].set_title(ytr[idx])
        ax[j, i].axis('off')
# basic preprocessing
xtr = np.expand_dims(xtr, -1).astype('float32') / 255
xte = np.expand_dims(xte, -1).astype('float32') / 255

ytr = np.array([[1 if ytr[i] == j else 0 for j in range(10)] for i in range(len(ytr))])
# LeNet5 style model with some dropout
model = Sequential()
model.add(Conv2D(16, 5, input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, 5, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2)
# create train set index
train_idx = np.random.choice(range(len(xtr)), size=int(np.floor(0.8 * len(xtr))), replace=False)

# create val set index
val_idx = np.array([i for i in range(len(xtr)) if i not in train_idx])
# create training and validation datasets
xtrain = xtr[train_idx, :, :]
ytrain = ytr[train_idx, :]
xval = xtr[val_idx, :, :]
yval = ytr[val_idx, :]
# early stopping
early_stop = EarlyStopping(patience=3, monitor='val_loss')

# training parameters
batch_size = 128
epochs=50

# note that I now fit_generator, rather than just fit
# batch_size goes to the datagen.flow method, while 
# epochs goes to fit_generator directly
# see docs for details
model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                    steps_per_epoch = len(xtr) // batch_size,
                    epochs=epochs,
                    validation_data=(xval, yval),
                    callbacks=[early_stop])
# get predictions (as probabilities)
predictions = model.predict(xte)

# turn probabilities into classes
preds = predictions.argmax(axis=1)

# write submission to dataframe then csv
submission = pd.DataFrame({'id': range(len(xte)), 'label': preds})
submission.to_csv('my_submission.csv', index=None)
