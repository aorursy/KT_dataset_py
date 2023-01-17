import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

 

# Importing the dataset

dataset = pd.read_csv('../input/churn-modellingcsv/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]

y = dataset.iloc[:, 13]



print(X.columns)

print(X)

print(y)

 

#Create dummy variables

geography=pd.get_dummies(X["Geography"],drop_first=True)

gender=pd.get_dummies(X['Gender'],drop_first=True)



print(geography)

print(gender)

 

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)



## Drop Unnecessary columns

X=X.drop(['Geography','Gender'],axis=1)

 

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



print(X_train)

print(X_test)
# Part 2 - Now let's make the ANN!

 

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU

from keras.layers import Dropout

 

 

# Initialising the ANN

classifier = Sequential()

 

# Adding the input layer and the first hidden layer

classifier.add(Dense(activation=LeakyReLU(alpha=0.1), input_dim=11, units=6, kernel_initializer="he_uniform"))

 

# Adding the second hidden layer

classifier.add(Dense(6, kernel_initializer='he_uniform',activation=LeakyReLU(alpha=0.1)))



# Adding the output layer

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))

 

# Compiling the ANN

classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

 

# Fitting the ANN to the Training set

model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)
# list all data in history

 

print(model_history.history.keys())

# summarize history for accuracy

print(model_history.history['accuracy'])

print(model_history.history['val_accuracy'])

 

# summarize history for loss

print(model_history.history['loss'])

print(model_history.history['val_loss'])
# Part 4 - Making the predictions and evaluating the model

 

# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

 

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

 

# Calculate the Accuracy

from sklearn.metrics import accuracy_score

score=accuracy_score(y_pred,y_test)

 

print(cm)

print(score)

print(classifier.summary())