import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,BatchNormalization,Dropout
data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
data.head()
classes = data['class']
features = data.drop('class',axis=1)
features = pd.get_dummies(features)
features
#Change the categorical variables in the Classs column to numbers
classes.replace('p',0,inplace=True)
classes.replace('e',1,inplace=True)
classes
from sklearn.model_selection import train_test_split

y=classes
x=features

#Splitting training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
model = Sequential()
model.add(Dense(32,input_shape=(117,)))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])
prediction=model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
predictions = model.predict_classes(x_test)
predictions
