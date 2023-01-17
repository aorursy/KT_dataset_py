import keras

from keras.models import Sequential

from keras.layers import Dense

import numpy as np



# fix random seed for reproducibility

np.random.seed(7)
import pandas as pd



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
x_df = pd.DataFrame(train_df.iloc[:,1:])

x_df
y_df = pd.get_dummies(train_df['label'])

y_df
# create model

model = Sequential()

model.add(Dense(512, input_dim=784, activation='relu'))

model.add(Dense(512, activation='sigmoid'))

model.add(Dense(10, activation='sigmoid'))
# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x = x_df.values

y = y_df.values



x

y



# Fit the model

model.fit(x, y, epochs=4, batch_size=128)
predictions = model.predict(test_df.values)
# testing!

num = 7015
firstToPredict = test_df.values[num].reshape((28,28))
from PIL import Image
im = Image.fromarray(firstToPredict.astype('uint8'))

im.save('test.png')

im
firstPrediction = predictions[num]

np.argmax(firstPrediction)
from PIL import Image
im = Image.fromarray(firstToPredict.astype('uint8'))

im.save('test.png')

im
firstPrediction = predictions[num]

np.argmax(firstPrediction)