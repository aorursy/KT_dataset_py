import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.optimizers import RMSprop
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D

import os
print(os.listdir("../input"))

img_rows = 28
img_cols = 28
num_classes = 10

trainData = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = trainData.label
x = trainData.iloc[:,1:]

# normalize
x = x / 255.0
test = test / 255.0

# label encoding
y = to_categorical(y,num_classes)

# reshape imgs
x = x.values.reshape(-1,img_rows,img_cols,1)
test = test.values.reshape(-1,img_rows,img_cols,1)

model = Sequential()
model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer ='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x, y,
                    batch_size=300,
                    epochs=10,
                    validation_split = 0.2)
import matplotlib.pyplot as plt

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best')

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best')
# predict results
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)