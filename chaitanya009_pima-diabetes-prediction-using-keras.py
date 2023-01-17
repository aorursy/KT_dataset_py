# imports
import pandas as pd
import numpy as np
# read data from csv file
data = pd.read_csv('../input/diabetes.csv')
#data description
data.describe(include="all")
X = data.iloc[:,0:8]
y = data.iloc[:,8]
#Spliting our dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Data scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform((X_train))
X_test = scaler.fit_transform((X_test))
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
model = Sequential([
    Dense(8, input_shape=(8,), activation='relu'),
    Dense(6, activation='relu'),
    Dense(2, activation='softmax')
])
model.summary()
model.compile(Adam(lr=.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=20, shuffle=True, verbose=2)
y_pred = model.predict_classes(X_test, batch_size=10, verbose=0)
#Accuracy score
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_pred)
print("accuracy: ", acc)
