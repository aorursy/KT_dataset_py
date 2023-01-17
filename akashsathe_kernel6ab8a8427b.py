import pandas as pd
import numpy as np
import os
pwd

raw_data=pd.read_csv("../input/deep-learning-az-ann/Churn_Modelling.csv")
raw_data

data=raw_data.copy()
data
data.describe()
data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)
data

data
nas=data.isnull().sum()
nas

data2=data.drop(['Exited'],axis=1)
data2
data_with_dummies=pd.get_dummies(data2,drop_first=True)
data_with_dummies=data_with_dummies.values
num_counters=int(np.sum(data['Exited']))
num_counters
indices_to_remove=[]
zeros_counter=0
for i in range (data_with_dummies.shape[0]):
    if data['Exited'][i]==0:
        zeros_counter=zeros_counter+1
        if zeros_counter>num_counters:
            indices_to_remove.append(i)   
indices_to_remove
x=np.delete(data_with_dummies,indices_to_remove, axis=0)
x

y_data=data['Exited']
y_data=y_data.values
y=np.delete(y_data,indices_to_remove, axis=0)
y
from sklearn.preprocessing import StandardScaler
shuffled_indices=np.arange(x.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_indices

x_shuffled=x[shuffled_indices]
x_shuffled

y_shuffled=y[shuffled_indices]
y_shuffled

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x_shuffled)
x_scaled
train_counts=int(0.8*shuffled_indices.shape[0])
validation_counts=int(0.1*shuffled_indices.shape[0])
train_counts
x_train=x_scaled[:train_counts]
x_train
x_validation=x_scaled[train_counts:train_counts+validation_counts]
x_validation
x_test=x_scaled[train_counts+validation_counts:]
x_test
y_train=y_shuffled[:train_counts]
y_train.shape

y_validation=y_shuffled[train_counts:train_counts+validation_counts]
y_validation.shape
y_test=y_shuffled[train_counts+validation_counts:]
y_test.shape
import tensorflow as tf
input_size=11
output_size=2
hidden_size=50
model=tf.keras.Sequential([
    
    tf.keras.layers.Dense(hidden_size,activation='relu'),
    tf.keras.layers.Dense(hidden_size,activation='relu'),
    tf.keras.layers.Dense(output_size,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
batch_size=100
max_epochs=100
model.fit(x_train,y_train,batch_size=batch_size,epochs=max_epochs,validation_data=(x_validation,y_validation),verbose=2)
test_loss,test_accuracy=model.evaluate(x_test,y_test)