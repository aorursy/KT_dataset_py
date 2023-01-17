import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing training set
dataset_train=pd.read_csv('../input/tesla-stock-dataset/tesla_train.csv')
traininng_set=dataset_train.iloc[:,1:2].values
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(traininng_set)

#Sclaed values of TSLA stock pricest
training_set_scaled
x_train=[]
y_train=[]
for i in range(50,1235):
    x_train.append(training_set_scaled[i-50:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

training_set_scaled.shape
x_train
x_train.shape
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# Building a function Recurrent Neural Network with Keras
# Dropout is being used to prevent overfitting

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Initialisation 
# Adding layers and definig the model
# this model has 5 hidden layers 

# Initialisation
reg=Sequential()

#Layer 1
reg.add(LSTM(units=100,return_sequences=True,input_shape=(x_train.shape[1],1)))
reg.add(Dropout(0.2))
#Layer 2
reg=Sequential()
reg.add(LSTM(units=100,return_sequences=True))
reg.add(Dropout(0.2))
#Layer 3
reg=Sequential()
reg.add(LSTM(units=100,return_sequences=True))
reg.add(Dropout(0.2))
#Layer 4
reg=Sequential()
reg.add(LSTM(units=100,return_sequences=True))
reg.add(Dropout(0.2))
#Layer 5
reg=Sequential()
reg.add(LSTM(units=100))
reg.add(Dropout(0.2))

# Final Output layer
reg.add(Dense(units=1))
# Compiling our neural network by choosing our loss functions and optimizer 
# Adam optimizer 
reg.compile(optimizer='adam',loss='mean_squared_error')

# Training the model on the training data
reg.fit(x_train,y_train,epochs=100,batch_size=32)
# Predictions
# Import the real values of 2020 July 14 -August 14
dataset_test=pd.read_csv('../input/tesla-stock-dataset/tesla_test.csv')
test_set=dataset_test.iloc[:,1:2].values

test_set
# We cannot directly scale the test values. Hence, we will first concanteneate witht the original train data and then scacle
# this way the scaling factor for both the test and train data remains the same
total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=total[len(total)-len(dataset_test)-50:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
# Creating a datastructure for predicting the on the test data which contains 24 days
x_test=[]
for i in range(50,74):
    x_test.append(inputs[i-50:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stockprices=reg.predict(x_test)

# using the inverse transform to get our real values back from the scacled values
predicted_stockprices=sc.inverse_transform(predicted_stockprices)



predicted_stockprices
# Visualising the results
plt.plot(test_set,color='red',label='Real Tesla Stock price')
plt.plot(predicted_stockprices,color='blue',label='Predicted Tesla Stock price')
plt.title('Tesla stock price prediction using LSTM')
plt.xlabel('Timeline (13th July- 14th August 2020)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



# Root mean squared error

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_set, predicted_stockprices))
print(rmse)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(test_set,predicted_stockprices)
mae


# In Stock price prediction RMSE is not the right metric to judge the accuracy of the model as RMSE is dependent 
# on the values of the output . A small difference is heavily penalised.
# Hence, after doing some research of my own I have found that relative error would be a better option.
# Even MAPE ( Mean absolute percentage error is a good measure I have not used MAPE as it is still in develeopment stage 
#           on sklearn and the beta version is unstable . It divides by zero . Please read the documenetation)
# # After going through a few sources I have learnt that RMSE/ Average value is a good metric to judge these kind of models. 
# 88/1517= 0.058 an dpercentage error is 5.8 %


#  upvote if you liked the project. This is just a basic model and there is much more that can be done.


    

