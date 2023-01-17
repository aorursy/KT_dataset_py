
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Flatten
import numpy as np
from sklearn.preprocessing import StandardScaler

#Loading the dataset
df = pd.read_csv('../input/nasa-asteroids-classification/nasa.csv')

df = pd.get_dummies(df)

# Separating the dataset into input (X) and output (y) variables
X = df.loc[:, df.columns != 'Harzardous']
y = df.loc[:, 'Hazardous']

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

#Scaling the dataset
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Building the Model
model = Sequential()
model.add(Dense(64, input_dim=(20), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#Compiling the Model 
compilation = model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Training the model
X_train = X_train[:,:20]
X_test = X_test[:,:20]
model.fit(X_train, y_train, epochs=10, verbose=False)

#Printing Results
scores = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))