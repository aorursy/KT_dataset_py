import numpy as np

import pandas as pd

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split



#Load dataset

dataset = pd.read_csv('../input/heart.csv')



X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
#One Hot encoding cp

label_encoder = LabelEncoder()

X[:, 2] = label_encoder.fit_transform(X[:, 2])



one_hot_encoder = OneHotEncoder(categorical_features=[2])

X = one_hot_encoder.fit_transform(X).toarray()

X = X[:, 1:]
#One Hot encoding slope

label_encoder = LabelEncoder()

X[:, 12] = label_encoder.fit_transform(X[:, 12])



one_hot_encoder = OneHotEncoder(categorical_features=[12])

X = one_hot_encoder.fit_transform(X).toarray()

X = X[:, 1:]
#One Hot encoding thal

label_encoder = LabelEncoder()

X[:, 15] = label_encoder.fit_transform(X[:, 15])



one_hot_encoder = OneHotEncoder(categorical_features=[15])

X = one_hot_encoder.fit_transform(X).toarray()

X = X[:, 1:]
#Spliting dataset into train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



print(X_train.shape, X_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation



#MLP model

model = Sequential()



model.add(Dense(48, input_shape=(X_train.shape[1],)))

model.add(Activation('relu'))



model.add(Dense(8))

model.add(Activation('relu'))



model.add(Dense(1))

model.add(Activation('sigmoid'))
model.summary()

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100)

loss, accuracy = model.evaluate(X_test, y_test)



print('Model accuracy -->', accuracy)