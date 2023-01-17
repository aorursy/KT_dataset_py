import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as plt
import keras

dataset = pd.read_csv('../input/FirstDraft3.csv')
X = dataset.iloc[:, 0:559].values
y = dataset.iloc[:, 560].values
y=y-1
y_train = keras.utils.to_categorical(y)

dataset = pd.read_csv('../input/TestDataSet.csv')
XT = dataset.iloc[:, 0:559].values
yT = dataset.iloc[:, 560].values
yT=yT-1
y_test = keras.utils.to_categorical(yT)




#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X)
x_test = sc.fit_transform(XT)




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

#classifier = Sequential()
#classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 559))
#classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
#classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#classifier.fit(X_train, y_train, batch_size = 25, epochs = 100)
#
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=559))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)



y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

