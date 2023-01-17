from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
housing=fetch_california_housing()
print(housing.DESCR)
df=pd.DataFrame(data=housing.data,columns=housing.feature_names)

target=pd.DataFrame(data=housing.target,columns=['target'])

df=pd.concat([df,target],sort=True,axis=1

            )

df.head(15)
X_train_full,X_test, y_train_full,y_test=train_test_split(

    housing.data, housing.target)



X_train,X_valid, y_train,y_valid=train_test_split(

    X_train_full,y_train_full)
scaler1=StandardScaler()

X_train=scaler1.fit_transform(X_train)

X_valid=scaler1.transform(X_valid)

x_test=scaler1.transform(X_test)
model=tf.keras.models.Sequential([

    tf.keras.layers.Dense(30,activation='relu',input_shape=X_train.shape[1:]),

    tf.keras.layers.Dense(10,activation='relu'),

    tf.keras.layers.Dense(1)

])
model.compile(loss=tf.keras.losses.mean_squared_error,optimizer='sgd'

             )

history=model.fit(X_train,y_train,epochs=20,

                  validation_data=(X_valid,y_valid))
#mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3] # pretend these are new instances

y_pred = model.predict(X_new)

y_pred
final_df=pd.DataFrame(history.history).plot(figsize=(13,5))

plt.grid(True)

plt.gca().set_ylim(0.25, 0.8) # set the vertical range to [0-1]

plt.show()
input_=tf.keras.layers.Input(shape=X_train.shape[1:])

hidden1=tf.keras.layers.Dense(30,activation='relu')(input_)

hidden2=tf.keras.layers.Dense(10,activation='relu')(hidden1)

concat=tf.keras.layers.concatenate(inputs=[input_ , hidden2])

output=tf.keras.layers.Dense(1)(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-3))
history1=model.fit(X_train,y_train,epochs=20,

         validation_data=(X_valid,y_valid))
final_df1=pd.DataFrame(history1.history).plot(figsize=(13,5))

plt.grid(True)

plt.gca().set_ylim(0, 2.2) # set the vertical range to [0-1]

plt.show()
input_A = tf.keras.layers.Input(shape=[5], name="wide_input")

input_B = tf.keras.layers.Input(shape=[6], name="deep_input")

hidden1 = tf.keras.layers.Dense(30, activation="relu")(input_B)

hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)

concat = tf.keras.layers.concatenate([input_A, hidden2])

output = tf.keras.layers.Dense(1, name="output")(concat)

model = tf.keras.Model(inputs=[input_A, input_B], outputs=[output])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-3))
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]

X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]

X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]

X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
history2=model.fit((X_train_A, X_train_B), y_train, epochs=20,

 validation_data=((X_valid_A, X_valid_B), y_valid))
final_df2=pd.DataFrame(history2.history).plot(figsize=(13,5))

plt.grid(True)

plt.gca().set_ylim(0, 3.5) # set the vertical range to [0-1]

plt.show()
input_A=tf.keras.layers.Input(shape=[5,], name='wide_input')

input_B=tf.keras.layers.Input(shape=[6,], name='deep_input')

hidden1=tf.keras.layers.Dense(30,activation='relu')(input_B)

hidden2=tf.keras.layers.Dense(30,activation='relu')(hidden1)

concat=tf.keras.layers.concatenate(inputs=[input_A,hidden2])

output =tf.keras.layers.Dense(1, name="output")(concat)

aux_output=tf.keras.layers.Dense(1,name='aux_output')(hidden2)

model=tf.keras.Model(inputs=[input_A,input_B],outputs=[output,aux_output])
model.compile(loss=['mse','mse'],optimizer='sgd',loss_weights=[0.9,0.1])
history3=model.fit([X_train_A,X_train_B],[y_train,y_train],epochs=20,

          validation_data=([X_valid_A,X_valid_B],[y_valid,y_valid]))
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

print('y_pred_main:',y_pred_main)

print('y_pred_aux:',y_pred_aux)
final_df3=pd.DataFrame(history3.history).plot(figsize=(13,5))

plt.grid(True)

plt.gca().set_ylim(0.3, 1.99) # set the vertical range to [0-1]

plt.show()
class WideAndDeepModel(tf.keras.Model):

    def __init__(self,units=30,activation='relu',**kwargs ):

        super().__init__(**kwargs)

        self.hidden1=tf.keras.layers.Dense(units=units,activation=activation)

        self.hidden2=tf.keras.layers.Dense(units=(units/2),activation=activation)

        self.main_output = tf.keras.layers.Dense(1)

        self.aux_output = tf.keras.layers.Dense(1)

        

    def call(self,inputs):

        input_A, input_B = inputs

        hidden1 = self.hidden1(input_B)

        hidden2 = self.hidden2(hidden1)

        concat = tf.keras.layers.concatenate([input_A, hidden2])

        main_output = self.main_output(concat)

        aux_output = self.aux_output(hidden2)

        return main_output, aux_output

    

    

model = WideAndDeepModel()
model.compile(loss=['mse','mse'],optimizer='sgd',loss_weights=[0.9,0.1])
history4=model.fit([X_train_A,X_train_B],[y_train,y_train],epochs=20,

          validation_data=([X_valid_A,X_valid_B],[y_valid,y_valid]))
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

print('y_pred_main:',y_pred_main)

print('y_pred_aux:',y_pred_aux)
final_df4=pd.DataFrame(history4.history).plot(figsize=(13,7))

plt.grid(True)

plt.gca().set_ylim(0.25, 2.3) # set the vertical range to [0-1]

plt.show()