# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
BATCH_SIZE = 86
EPOCH = 50
# read data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# pd.DataFram -> np.array
train = np.array(train)
test = np.array(test)

# splite data and label
train_x = train[:, 1:]
train_y = train[:, 0]

train_x = train_x.reshape((-1, 28, 28, 1))
train_y = to_categorical(train_y)
test = test.reshape((-1, 28, 28, 1)) / 255

# train val spilte
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)


INPUT_DIM = train_x[0].shape
OUTPUT_DIM = len(train_y[0])
# data augmentation
datagen = ImageDataGenerator(rotation_range=10,
                            shear_range=0.1,
                             zoom_range=0.1,
                            rescale=1/255)
train_it = datagen.flow(train_x, train_y, batch_size=BATCH_SIZE, shuffle=True)
val_it = datagen.flow(val_x, val_y, batch_size=BATCH_SIZE, shuffle=True)
# visualization
# def show(x):
#     plt.imshow(x.reshape(28,28))
    
# show(x[0])
def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=5, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(output_dim, activation='softmax'))
    
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
model = build_model(INPUT_DIM, OUTPUT_DIM)
model.summary()
history = model.fit_generator(train_it, validation_data=val_it, epochs=EPOCH, 
                              steps_per_epoch=len(train_x)//BATCH_SIZE, validation_steps=len(val_x)//BATCH_SIZE,
                              callbacks=[learning_rate_reduction])

print('final acc:', sum(history.history['acc'][-10:]) / 10)
print('final val acc:', sum(history.history['val_acc'][-10:])/ 10)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
predict = model.predict(test)
predict = np.argmax(predict, axis=1)
result = []
for i, label in enumerate(predict):
    result.append((i+1, label))
df = pd.DataFrame(result, columns=['ImageId', 'Label'])

df.to_csv('/kaggle/working/submission.csv', index=False)
