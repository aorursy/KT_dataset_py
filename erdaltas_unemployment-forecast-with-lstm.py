# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#for visualization
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import training set
training_set=pd.read_csv("../input/number-of-unemployed-train.csv")
print(training_set.head())
training_set=training_set.iloc[:,0].values
plt.plot(training_set)
plt.show()
#normalization
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=training_set.reshape(-1,1)
training_set=sc.fit_transform(training_set)
training_set[:5]

#İnput and output.timestep=1
x_train=training_set[0:158]
y_train=training_set[1:159]
print("x-train:\n",x_train[:5],"\ny-train:\n",y_train[:5])
#Reshape.
x_train=np.reshape(x_train,(158,1,1))
#Importing Keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#Initialising the RNN
regressor=Sequential()
#Add input layer and LSTM layer
regressor.add(LSTM(70,activation="sigmoid",input_shape=(None,1)))
#output layer
regressor.add(Dense(units=1))
#Compile the RNN
regressor.compile(optimizer="adam",loss="mean_squared_error")
#fitting training set
regressor.fit(x_train,y_train,batch_size=32,epochs=200)
#Making prediction
test_set=pd.read_csv("../input/number-of-unemployed-test.csv")
real=test_set.iloc[:,0].values.reshape(-1,1)
print(test_set.head())
#prediction
inputs=real
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(14,1,1))
predicted=regressor.predict(inputs)
predicted=sc.inverse_transform(predicted)
#train graphic
train_predict=regressor.predict(x_train)
train_predict=sc.inverse_transform(train_predict)
plt.plot(train_predict,color="red",label="train_predict")
plt.plot(sc.inverse_transform(y_train),color="blue",label="train_real")
plt.xlabel("Date")
plt.ylabel("Unemployed Number")
plt.title("Train Prediction")
plt.legend()
plt.show()

#test graphic
plt.plot(real,color="red",label="real")
plt.plot(predicted,color="blue",label="predicted")
plt.xlabel("Date")
plt.ylabel("Unemployed Number")
plt.title("Test Prediction")
plt.legend()
plt.show()