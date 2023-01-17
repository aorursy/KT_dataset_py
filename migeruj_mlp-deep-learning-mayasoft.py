import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_csv("../input/credit_data.csv")

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout


from sklearn.preprocessing import StandardScaler, LabelBinarizer
X = df[['edad', 'sexo','trabajo','casa','ingresos','monto','duracion','proposito']]
y = df['riesgo']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=3000,
          batch_size=128)


score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
print("Precision Average")
print((score[0]+score[1])/2)
#Ejemplo de uso | Predict one instance
test_data = np.array([[35,1,2,0,3,23000,12,0]]) #Parametros del cliente a evaluar | Client Parameters to predict
prediction_res = model.predict(test_data)
prediction_res = prediction_res[0]
print("0: Riesgo Bajo | Good Risk")
print("1: Riesgo Alto | Bad Risk")
print(prediction_res.argmax(axis=-1))