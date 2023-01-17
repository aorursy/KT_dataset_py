import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pil
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
example = pd.read_csv("../input/gender_submission.csv")
keep_columns = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Fare", "Embarked"]

x_train = df_train[keep_columns]
x_train['Sex'] = x_train['Sex'].map({'male' : 1, 'female' : 0})
x_train['Embarked'] = x_train['Embarked'].map({'Q' : 2, 'S' : 0, 'C' : 1})
x_train.fillna(x_train.Age.mean(), inplace = True)
y = df_train.Survived
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y, test_size = 0.2)


X_val = df_test[keep_columns]
X_val['Embarked'] = X_val['Embarked'].map({'Q' : 2, 'S' : 0, 'C' : 1})
X_val['Sex'] = X_val['Sex'].map({'male' : 1, 'female' : 0})
X_val["Age"].fillna(X_val.Age.mean(), inplace  = True)
X_val["Fare"].fillna(X_val.Fare.mean(), inplace = True)
mini_batches = 256
epoch = 200 


#model

model = Sequential()
model.add(Dense(128, input_shape = (7,), activation = 'sigmoid'))
model.add(Dense(128))
model.add(LeakyReLU(0.1))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

#Compiling
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Training

history = model.fit(X_train, Y_train, batch_size = mini_batches, epochs = epoch, verbose = 1, validation_data = (X_test, Y_test) )

loss, accuracy = model.evaluate(X_test, Y_test)
print('accuracy = ', accuracy)
predictions = model.predict(X_val, batch_size  = mini_batches)
i = 0
prediction = np.zeros(predictions.shape)
for i in range(0,np.size(predictions,0)):
    prediction[i] = np.argmax(predictions[i])
prediction = pd.DataFrame(prediction,columns = ['PassengerId', 'Survived'], dtype = int)
prediction['PassengerId'] = X_val['PassengerId']
print(prediction)
#Would invite any suggestions to improve. Do leave a comment.