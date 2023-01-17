# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", dtype=int)

X_test = pd.read_csv("../input/test.csv", dtype=int)
X_train = train.drop('label', axis=1)

y_train = train['label']
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

plot0 = X_train.iloc[717].values.reshape(28,28)

plt.imshow(plot0)
X_train = X_train/255.0

X_test = X_test/255.0
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

from keras.preprocessing.image import ImageDataGenerator

X_train2 = np.array(X_train, copy=True) 

y_train2 = np.array(y_train, copy=True) 

X_train3 = np.array(X_train, copy=True) 

y_train3 = np.array(y_train, copy=True) 

X_train4 = np.array(X_train, copy=True) 

y_train4 = np.array(y_train, copy=True) 

X_train5 = np.array(X_train, copy=True) 

y_train5 = np.array(y_train, copy=True) 

data = ImageDataGenerator(

    featurewise_center=True,

    zoom_range = 0.15, 

    featurewise_std_normalization=True,

    rotation_range=45,

    height_shift_range=0.15,

    width_shift_range=0.15

    )



data.fit(X_train, augment=True, rounds=1, seed=5)



result_x  = np.concatenate((X_train3, X_train2, X_train), axis=0)

result_y  = np.concatenate((y_train3, y_train2, y_train), axis=0)

result_x2 = np.array(result_x, copy=True) 

result_y2 = np.array(result_y, copy=True)

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)

result_y = to_categorical(result_y, num_classes = 10)
from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=4, verbose=1, factor=0.5, min_lr=0.00001)

                                                                                      

from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
import keras

from keras import Sequential

from functools import partial



DefaultConv2D = partial(keras.layers.Conv2D,

                        kernel_size=3, activation='relu', padding ="SAME")

model1=Sequential([

    DefaultConv2D(filters=64, kernel_size=3, input_shape=[28, 28, 1]),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=128),

    DefaultConv2D(filters=128),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=256),

    DefaultConv2D(filters=256),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation='relu'),

    keras.layers.Dropout(0.6),

    keras.layers.Dense(units=64, activation='relu'),

    keras.layers.Dropout(0.6),

    keras.layers.Dense(units=10, activation='softmax')

    ])
from keras import optimizers

optimizer= optimizers.SGD(lr=0.1)

model1.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model1.fit_generator(data.flow(result_x, result_y, batch_size=35),

                    steps_per_epoch=len(result_x) / 32, epochs = 12)

history1 = model1.fit(X_train, y_train, epochs=30, validation_split=0.15, callbacks=[learning_rate_reduction])
pred1 = model1.predict(X_test)

voting = np.sum([pred1], axis=0)

pred_all = []

for i in range(0, len(voting)):

    pred_all.append(np.argmax(voting[i]))
result = pd.DataFrame(columns= ['ImageID', 'Label'])

for i in range(1, 28001):

    result.loc[i-1] = [i, pred_all[i-1]]

    print(i, pred_all[i])
my_submission = result

my_submission.to_csv('submission.csv', index=False)