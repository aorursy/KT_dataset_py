import pandas as pd
import numpy as np
import tensorflow as tf
tf.__version__
dataset=pd.read_excel('../input/combined-cycle-power-plant-data-set/Folds5x2_pp.xlsx')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam',loss='mean_squared_error')
ann.fit(x_train,y_train,batch_size=32,epochs=100)
y_pred=ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))