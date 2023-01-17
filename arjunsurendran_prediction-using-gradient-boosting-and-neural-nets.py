#Importing Libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Importing dataset

voice = pd.read_csv('../input/voice.csv')
voice.head()
voice.info()
y = 1 * (voice['label'] == 'male')
X = voice.drop('label',axis = 1).values
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Features')

# Draw the heatmap using seaborn

sns.heatmap(voice.drop('label', axis = 1).astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
from sklearn.ensemble import GradientBoostingClassifier as GBC

gbc = GBC(learning_rate = 0.01)

gbc.fit(X_train,y_train)
y_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)

print('Accuracy = ', end = "")

print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = gbc, X = X, y = y, cv = 10, n_jobs = -1)

print('Accuracy =', accuracies.mean()*100, '%')

print('Standard Deviation =', accuracies.std()*100, '%')
#Importing Libraries for ANN

import keras

from keras.models import Sequential

from keras.layers import Dense
#Initialisng the ANN

classifier = Sequential()



#Input Layer and First Hidden Layer

classifier.add(Dense(units = 32, activation = 'relu', input_dim = X.shape[1]))



#Adding the Second hidden layer

classifier.add(Dense(units = 32, activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, activation = 'sigmoid'))



#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting ANN to the training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 0)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test,y_pred)

print(cm)

print('Accuracy = ', end = "")

print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))