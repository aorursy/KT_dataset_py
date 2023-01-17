import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
print(X[1])

print(y)
X.shape
y.shape
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]
X[1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train[1:5]
y_train[1:5]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
classifier = Sequential()
initializers = keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)
classifier.add(Dense(units = 11, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense( units = 8, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))
classifier.add(Dense(units = 4, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'sigmoid'))
opti = keras.optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad=False)
classifier.compile(optimizer = opti, loss = 'binary_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, validation_split=0.10, batch_size = 32, epochs = 10000)
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

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
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

print(y_pred)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
total_test_sample = 2000

Accuracy = ((cm[0][0]+cm[1][1])/total_test_sample)*100

Recall = (cm[0][0]/(cm[0][0]+cm[1][0]))*100

Precision = (cm[0][0]/(cm[0][0]+cm[0][1]))*100
Recall_1 = (cm[0][0]/(cm[0][0]+cm[1][0]))

Precision_1 = (cm[0][0]/(cm[0][0]+cm[0][1]))
F = (Recall_1+Precision_1)/(2*Recall_1*Precision_1)
print("********** CONFUSION MATRIX MEASURES**********")

print("The accuracy is:::::",Accuracy,"%")

print("\n")

print("**********************************************")

print("The Recall is:::::", Recall,"%")

print("\n")

print("**********************************************")

print("The Precision is:::::",Precision,"%")

print("\n")

print("**********************************************")

print("The F-Measure is:::::",F)