import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
import theano
import tensorflow
import keras 
from keras.models import Sequential
from keras.layers import Dense
import os
print(os.listdir("../input"))

# Reading data 
data = pd.read_csv('../input/adult.csv', names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
                                         'marital-status', 'occupation', 'relationship','race', 'sex',
                                         'capital-gain', 'capital-loss','hours-per-week','native-country',
                                         'Output'] )
data.info()
data.describe().T
data.head().T
data.replace(to_replace=' ?',value= np.nan ,inplace=True)
sns.heatmap(data.isnull())
data.dropna(inplace=True)
X = data.iloc[:, 0:14].values
y = data.iloc[:, 14].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 3] = labelencoder_X2.fit_transform(X[:, 3])
labelencoder_X3 = LabelEncoder()
X[:, 5] = labelencoder_X3.fit_transform(X[:, 5])
labelencoder_X4 = LabelEncoder()
X[:, 6] = labelencoder_X4.fit_transform(X[:, 6])
labelencoder_X5 = LabelEncoder()
X[:, 7] = labelencoder_X5.fit_transform(X[:, 7])
labelencoder_X6 = LabelEncoder()
X[:, 8] = labelencoder_X6.fit_transform(X[:, 8])
labelencoder_X7 = LabelEncoder()
X[:, 9] = labelencoder_X7.fit_transform(X[:, 9])
labelencoder_X8 = LabelEncoder()
X[:, 13] = labelencoder_X8.fit_transform(X[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,
         33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49,50,51,53,54,55,56,58,59,60,61,62,63,64,65,67,68,69,
         70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,
         102,103]]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 96))

# Adding the second hidden layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 10)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
((4134+990)/(4131+990+546+366))*100