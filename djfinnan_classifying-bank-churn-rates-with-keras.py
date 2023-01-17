import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing the dataset
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

df = pd.get_dummies(df)

# Splitting data into input (X) and output (y) variables
X = df.loc[:, df.columns!='Exited']
y = df.loc[:, 'Exited']

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling the Dataset
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Building the model
model = Sequential()
model.add(Dense(64, input_dim=(20), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#Compling the model
compilation = model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Training the Model
X_train = X_train[:,:20]
X_test = X_test[:,:20]
model.fit(X_train, y_train, epochs=20, verbose=False)

#Printing Results
scores = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))