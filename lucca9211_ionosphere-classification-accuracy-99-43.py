import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder

# load dataset

data = pd.read_csv("../input/ionosphere_data_kaggle.csv",delimiter=",")

data.head()
# Print Missing value

print(data.isnull().sum())
# split into input (X) and output (Y) variables



X = data.values[1:,0:34].astype(float)

Y = data.values[1:,34]

# encode class values as integers

encoder = LabelEncoder()

encoder.fit(Y)

Y = encoder.transform(Y)
# create model

model = Sequential()

model.add(Dense(34, input_dim=34 , activation= 'relu' ))

model.add(Dense(1,  activation= 'sigmoid' ))

# Compile model

epochs = 50

learning_rate = 0.1

decay_rate = learning_rate / epochs

momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(loss= 'binary_crossentropy' , optimizer=sgd, metrics=[ 'accuracy' ])

# Fit the model

model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=8, verbose=2)
# evaluate the model

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))