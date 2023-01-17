

import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("../input/real-estate-price-prediction/Real estate.csv")

print(data)
data.drop('X1 transaction date',axis=1)
target=data['Y house price of unit area'].to_numpy()
data=data.drop('Y house price of unit area',axis=1)

inputs=data.to_numpy()
s_input=StandardScaler()
inputs=s_input.fit_transform(inputs)
inputs.shape

np.savez('tf_data',inputs=inputs,targets=target)
datatf=np.load('tf_data.npz')
inputss=datatf['inputs']
targets=datatf['targets']

print(targets.shape)
output_size=1
input_size=inputss.shape[1]
print(input_size)
model=tf.keras.Sequential([tf.keras.layers.Dense(8,input_shape=(7,),activation='relu'),tf.keras.layers.Dense(8,activation='relu'),tf.keras.layers.Dense(output_size)])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_squared_error')
model.fit(inputss,targets,epochs=1500,verbose=1)
weights=model.layers[0].get_weights()[0]
bias=model.layers[0].get_weights()[1]
prediction=model.predict(inputss)
weights
prediction
targets


