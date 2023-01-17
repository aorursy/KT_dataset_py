import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

print(train.shape)
print(train.head(n=5))
print(test.shape)
print(test.head(n=5))
train.info()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['Product_Info_2'] = encoder.fit_transform(train['Product_Info_2'])
test['Product_Info_2'] = encoder.fit_transform(test['Product_Info_2'])
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
print(train.columns)
x_cols = train.columns[1:-1]
print(x_cols)
n_input = len(x_cols)
n_output = 9
print (n_input)
print (n_output)
model = Sequential()
# MLP with 126 -> 200 -> 9
model.add(Dense(input_dim=n_input, output_dim=200, init="uniform", activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(input_dim=200, output_dim=200, init="uniform", activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(input_dim=200, output_dim=n_output, init="uniform", activation="softmax"))
rmsprop = RMSprop(lr=0.005, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
response_transformed = np_utils.to_categorical(train['Response'])
print(response_transformed.shape)
(train[x_cols].values).shape
model.fit(train[x_cols].values, response_transformed, nb_epoch=50, batch_size=1000)
test['Response'] = model.predict_classes(test[x_cols].values)
import time
PREDICTIONS_FILENAME_PREFIX = 'predictions_'
PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_PREFIX + time.strftime('%Y%m%d-%H%M%S') + '.csv'
print(test.columns)
print(PREDICTIONS_FILENAME)
test[['Id','Response']].to_csv(PREDICTIONS_FILENAME, index=False)