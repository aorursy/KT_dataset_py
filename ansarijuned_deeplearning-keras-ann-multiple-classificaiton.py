import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
dataset = pd.read_csv("../input/iris.data",header=None)
dataset
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=101)
y1
Y
model = Sequential()

model.add(Dense(4,input_shape=(4,), activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=150)
y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
