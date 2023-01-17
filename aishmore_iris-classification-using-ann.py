# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing dataset
Iris = pd.read_csv('../input/iris/Iris.csv')
Iris.head()
Iris.info()
# determining all iris species included
Iris['Species'].value_counts(dropna=False)
# creating dummmy variable for the species type
Species = pd.get_dummies(Iris['Species'])
#Selecting required data
X = Iris.iloc[:, 1:5].values
y = Species.values
#Splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#Importing keras packages
from keras.models import Sequential
from keras.layers import Dense
#Initializing Artificial Neural Network (ANN)
classifier = Sequential()
#Adding input layer and first hidden layer
classifier.add(Dense(activation="relu", kernel_initializer="uniform", input_dim=4, units=3))
#Adding second hidden layer
classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=3))
#Adding output layer
classifier.add(Dense(activation="sigmoid", kernel_initializer="uniform", units=3))
#Compiling ANN
classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'] )
#Fitting ANN to training set
history = classifier.fit(X_train, y_train, batch_size = 25, epochs = 500, validation_split=0.25, verbose=0)

#Predicting test results
y_pred = classifier.predict(X_test)
plt.scatter(x=y_test,y=y_pred, color= 'green')
plt.xlabel('y_test values')
plt.ylabel('y_pred values')
plt.show()
import matplotlib.pyplot as plt



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()