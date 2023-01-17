# import libraries

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from keras.models import Sequential

from sklearn.metrics import confusion_matrix

from sklearn.impute import SimpleImputer 

from keras.layers import Dense,Activation,Dropout

from sklearn.model_selection import train_test_split
#filter warnings

warnings.filterwarnings("ignore")

# import data 

data = pd.read_csv('../input/diabetes/diabetes.csv',delimiter=',')

data.head()
# Visualize Malign and Benign Patient Count

count = data.Outcome.value_counts()

count.plot(kind='bar')

plt.legend()

plt.show()
# First 8 column is input parameters. 

X = data.iloc[:,0:8]

# Last Column is output data. So (1 is normal, 0 is diabet) 

y = data.iloc[:,-1]

X
scaler = preprocessing.MinMaxScaler()

X = scaler.fit_transform(X)

X
# train test split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# Create Model with KERAS library

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(8,))) # FC Fully Connected Layer, input_dimension is dataset input parameters

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))               

model.add(Dense(1, activation='sigmoid'))
# Compile Model

model.compile(

    optimizer="rmsprop",

    loss="binary_crossentropy",

    metrics=["accuracy"])
# Fit Model

history =  model.fit(X_train,y_train, 

                     epochs=200,

                     verbose=1,

                     validation_data=(X_test, y_test))
# model save

model.save_weights("example.h5")
# Visualize Loss and Accuracy Rates

plt.plot(history.history["loss"],label="train_loss")

plt.plot(history.history["val_loss"],label="test_loss")

plt.legend()

plt.show()



plt.figure()

plt.plot(history.history["accuracy"],label="train_acc")

plt.plot(history.history["val_accuracy"],label="test_acc")

plt.legend()

plt.show()
Y_pred = model.predict(X_test)

Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred]
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, Y_pred)

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

print('Confusion Matrix :')

print(cm) 

print('Accuracy Score :',accuracy_score(y_test, Y_pred))

print('Report : ')

print(classification_report(y_test, Y_pred))