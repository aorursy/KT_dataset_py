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
final_df=pd.DataFrame(history.history).plot(figsize=(10,5))
plt.grid(True)
plt.gca().set_ylim(0, 2.2) # set the vertical range to [0-1]
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
final_df1=pd.DataFrame(history1.history).plot(figsize=(10,5))
plt.grid(True)
plt.gca().set_ylim(0, 2.2) # set the vertical range to [0-1]
plt.show()
# The End