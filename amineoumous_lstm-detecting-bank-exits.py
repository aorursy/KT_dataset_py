import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import os

print(os.listdir("../input"))
# Multiple Columns Label Encoder

from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns 



    def fit(self,X,y=None):

        return self



    def transform(self,X):

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)
# Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

Churn_Modelling = pd.read_csv("../input/bank-customer-churn-modeling/Churn_Modelling.csv")

X = Churn_Modelling.iloc[:,3:-1]

y = Churn_Modelling.iloc[:,-1]

X = MultiColumnLabelEncoder(columns = ['Geography','Gender']).fit_transform(pd.DataFrame(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = MinMaxScaler(feature_range=(0,1))

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

X
# LSTM Implementation

import keras

from subprocess import check_output

from keras.layers.core import Dense, Activation, Dropout

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

import time

trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
from numpy import newaxis

model = Sequential()



model.add(LSTM(input_shape=(1,10),units=6,return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(32,return_sequences=True))

model.add(LSTM(32))

model.add(Dropout(0.1))

model.add(Dense(activation="sigmoid", units=1))



start = time.time()

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print ('compilation time : ', time.time() - start)
history=model.fit(trainX,y_train,batch_size=500,epochs=1000,validation_split=0.1)
trainPredict = model.predict(trainX)

print(trainPredict)

print(model.summary())
plt.plot(np.array(history.history['accuracy']) * 100)

plt.plot(np.array(history.history['val_accuracy']) * 100)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['train', 'validation'])

plt.title('Accuracy over epochs')

plt.show()
y_pred = model.predict(testX)

print(y_pred[:5])
y_pred = (y_pred > 0.5).astype(int)

print(y_pred[:5])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print (((cm[0][0]+cm[1][1])*100)/(len(y_test)), '% of testing data was classified correctly')