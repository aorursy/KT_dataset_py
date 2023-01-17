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
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, UpSampling2D

from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split



dropout = True

dropout_val = 0.2
df_train = pd.read_csv('../input/train.csv')
y = df_train['label'].values

x = df_train.drop('label', axis=1).values
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.15)
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)

xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)
def modelConv2D():

    model = Sequential()

    model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    if dropout:

        model.add(Dropout(dropout_val))

    

    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    if dropout:

        model.add(Dropout(dropout_val))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    if dropout:

        model.add(Dropout(dropout_val))

    

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    if dropout:

        model.add(Dropout(dropout_val))

    

    model.add(UpSampling2D(size=(2, 2)))

    model.build()

    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    if dropout:

        model.add(Dropout(dropout_val))

    

    model.add(Flatten())

    model.add(Dense(300, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(0.001), metrics=['accuracy'])

    return model
model = modelConv2D()
history = model.fit(xTrain, yTrain, epochs=50, validation_data=(xTest, yTest))
test_df = pd.read_csv('../input/test.csv')

test_df.index += 1
testData = test_df.values

testData = testData.reshape(testData.shape[0], 28, 28, 1)
pred = model.predict(testData)
finalsub = []

for p in pred:

    finalsub.append(np.argmax(p))
submission = pd.DataFrame({

    'ImageId': test_df.index,

    'Label': finalsub

})
submission.to_csv('submission.csv', index=False)
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

%matplotlib inline

layer_outputs = [layer.output for layer in model.layers]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(xTest[12].reshape(1, 28, 28, 1))

# xTest[0].shape
def display_activations(activations, cols, rows, activation_index):

    activation = activations[activation_index]

    act_index = 0

    fig, ax = plt.subplots(rows, cols)

    for row in range(rows):

        for col in range(cols):

            ax[row][col].imshow(activation[0, :, :, act_index], cmap='gray')

            act_index += 1

            

    
plt.imshow(xTest[12][:, :, 0], cmap='gray')
display_activations(activations, 2, 2, 2)