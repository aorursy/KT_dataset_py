# first neural network with keras make predictions
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(16)

#Importing the dataset
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
pd.get_dummies(df)

#Checking for missing values 
print(df.isnull().any())
# Splitting data into input (X) and output (y) variables
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling the Dataset
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
model = Sequential()
model.add(Dense(24, input_dim=8, activation='relu'))
model.add(Dense(12, activation ='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
model.fit(X_train, y_train, epochs=50, verbose=False)

#Printing Results
scores = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))
