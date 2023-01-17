import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()

type(housing)
print(housing['DESCR'])
X_train_full,X_test,y_train_full,y_test = train_test_split(housing.data,housing.target)

X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_valid = scaler.transform(X_valid)

X_test = scaler.transform(X_test)
from tensorflow import keras



input_ = keras.layers.Input(shape=X_train.shape[1:])

hidden1 = keras.layers.Dense(30, activation="relu")(input_)

hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.concatenate([input_, hidden2])

output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input_], outputs=[output])
model.summary()
keras.utils.plot_model(model,'wide_alpha.png',show_shapes=True)
model.compile(loss = 'mean_squared_error',optimizer=keras.optimizers.SGD(lr = 1e-3))
history = model.fit(X_train,y_train,

          epochs = 50,

          validation_data = (X_valid,y_valid))
fig = plt.figure(dpi = 100,figsize = (5,3))

ax = fig.add_axes([1,1,1,1])

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

ax.plot(epochs,loss,lw = 2,color = 'green',label = 'Loss')

ax.plot(epochs,val_loss,lw = 2,color = 'blue',label = 'Val Loss')

plt.grid(True)

plt.legend()

ax.set(xlabel = 'Number of Epochs',ylabel = 'Loss',title = 'Loss Curve')

plt.show()
X_new = X_test[:3]

y_pred = model.predict(X_new)

y_pred
input_A = keras.layers.Input(shape = [5],name = 'wide_input')

input_B = keras.layers.Input(shape = [6],name = 'deep_input')



hidden1 = keras.layers.Dense(30,activation = 'relu')(input_B)

hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)



concat = keras.layers.concatenate([input_A,hidden2])



output = keras.layers.Dense(1)(concat)



model = keras.Model(inputs = [input_A,input_B],outputs = [output])
model.summary()
keras.utils.plot_model(model,'multi.png',show_shapes=True)
model.compile(loss = 'mean_squared_error',optimizer=keras.optimizers.SGD(lr = 1e-3))
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]

X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]

X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]

X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
history = model.fit((X_train_A,X_train_B),y_train,

          epochs = 50,

          validation_data = ((X_valid_A,X_valid_B),y_valid))
fig = plt.figure(dpi = 100,figsize = (5,3))

ax = fig.add_axes([1,1,1,1])

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

ax.plot(epochs,loss,lw = 2,color = 'green',label = 'Loss')

ax.plot(epochs,val_loss,lw = 2,color = 'blue',label = 'Val Loss')

plt.grid(True)

plt.legend()

ax.set(xlabel = 'Number of Epochs',ylabel = 'Loss',title = 'Loss Curve')

plt.show()
y_pred = model.predict((X_new_A,X_new_B))

y_pred
input_A = keras.layers.Input(shape = [5],name = 'wide_input')

input_B = keras.layers.Input(shape = [6],name = 'deep_input')



hidden1 = keras.layers.Dense(30,activation='relu')(input_B)

hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)



concat = keras.layers.concatenate([input_A,hidden2])



output = keras.layers.Dense(1,name = 'main_output')(concat)

aux_output = keras.layers.Dense(1,name = 'aux_output')(hidden2)



model = keras.Model(inputs = [input_A,input_B],outputs = [output,aux_output])



model.summary()
keras.utils.plot_model(model,'complex.png',show_shapes=True)
model.compile(loss = ['mse','mse'],loss_weights = [0.9,0.1],optimizer='sgd')
history = model.fit((X_train_A,X_train_B),(y_train,y_train),

          epochs = 50,

          validation_data = ((X_valid_A,X_valid_B),(y_valid,y_valid)))
pd.DataFrame(history.history).head()
fig = plt.figure(dpi = 100,figsize = (5,3))

ax = fig.add_axes([1,1,1,1])

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(loss)+1)

ax.plot(epochs,loss,lw = 2,color = 'green',label = 'Loss')

ax.plot(epochs,val_loss,lw = 2,color = 'blue',label = 'Val Loss')

plt.grid(True)

plt.legend()

ax.set(xlabel = 'Number of Epochs',ylabel = 'Loss',title = 'Loss Curve')

plt.show()
total_loss,main_loss,aux_loss = model.evaluate((X_test_A,X_test_B),

                                               (y_test,y_test))
y_pred_main,y_pred_aux = model.predict([X_new_A,X_new_B])
y_pred_main
y_pred_aux