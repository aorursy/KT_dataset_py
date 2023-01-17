# Loading libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Multilayer perceptron Neural Network

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import np_utils
# importing data

traindata = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv')

testdata = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv')
# encoding string data columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

traindata['color'] = le.fit_transform(traindata['color'])

testdata['color'] = le.fit_transform(testdata['color'])

traindata['type'] = le.fit_transform(traindata['type'].astype(str))
# Getting required features

X = traindata.iloc[:,1:6].values

y = traindata.iloc[:,6].values

X_test = testdata.iloc[:,1:6].values
# Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X)

X_test = sc.fit_transform(X_test)
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
# Using one hot encoding for multi-class categories (Fast-ANN)

y = np_utils.to_categorical(y)
# Defining keras model

def ghost_model():

	# create model

	model = Sequential()

	model.add(Dense(5, input_dim=5, init='normal', activation='relu'))

	model.add(Dense(3, init='normal', activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
# build the model

model = ghost_model()

# Fit the model

model.fit(X_train, y, nb_epoch=200, batch_size=5, verbose=0)
# predicting results

pred = model.predict(X_test)

pred = pred.argmax(1)

pred = le.inverse_transform(pred)
# generating output

id_data = testdata[['id']].values

type_data = np.expand_dims(pred,axis=1)

f = np.hstack((id_data,type_data))

df = pd.DataFrame(f, columns = ['id', 'type'])

df.to_csv('sample_submission.csv', index=False)

df.head()