import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
# reading the data and calculating the computation time



%time training_data = pd.read_csv('../input/Google_Stock_Price_Train.csv')



# checking the shape of the data

training_data.shape
training_data.head()

# let's describe the data



training_data.describe()
# Lets work on the open stock price only and take out the " open " stock column.



training_data = training_data.iloc[:, 1:2]



# The training_data is in the form of dataframe.

training_data.shape
# checking the head of the data



training_data.head()
import matplotlib.pylab as plt

plt.figure(figsize=(10,5))

plt.plot(training_data, color ='green');

plt.ylabel('Stock Price')

plt.title('Google Stock Price')

plt.xlabel('Time')

plt.show()


# Normalize the training data between [0,1]

from sklearn.preprocessing import MinMaxScaler

#the fit method, when applied to the training dataset, learns the model parameters (for example, mean and standard deviation). 

#We then need to apply the transform method on the training dataset to get the transformed (scaled) training dataset.

#We could also perform both of this step in one step by applying fit_transform on the training dataset.

mm = MinMaxScaler(feature_range = (0, 1))

training_data_scaled = mm.fit_transform(training_data)

training_data_scaled.shape
plt.figure(figsize=(10,5))

plt.plot(training_data_scaled);

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.show()


# Getting the inputs and outputs directly if you know how to consider the past data for the number of time stamps needed for RNN.

x_train = training_data_scaled[59:1257]

y_train = training_data_scaled[60:1258]

print(x_train.shape)

print(y_train.shape)
'''for i in range(60,1258):

    #first 59 values of x will be the training data and the 60th value will be output

    #2nd set of values start from 60th (inculding the output of the  first set) will be the next 60 input values of the 2nd set and it continues.

    #append is to add the values to the values to x and y

    #as the values are in the form of dataframe, they has to be stored in the form of a numpy array.

    x_train.append(training_data_scaled[i-60:i, 0])

    y_train.append(training_data_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

print(x_train.shape)

print(y_train.shape)'''
# reshaping

x_train = np.reshape(x_train, (1198,1,1))

print(x_train.shape)
import keras 

from keras.models import Sequential #helps to create model, layer by layer.

from keras.layers import Dense, LSTM, Dropout

#The dense layer is fully connected layer, so all the neurons in a layer are connected to those in a next layer.

#The dropout drops connections of neurons from the dense layer to prevent overfitting. the neurons whose value falls under 0, will be removed.

#LSTM gates to control the memorizing process. For detailed information on LSTM, go through the link below.

''' https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47 '''
# Create model using LSTM, Dropout and Dense layer as an output layer. 

#Initializing the RNN

regressor = Sequential()

regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

regressor.add(Dropout(0.2))
# Adding second hidden layer

regressor.add(LSTM(units = 50,return_sequences = True))

regressor.add(Dropout(0.2))
# Adding third hidden layer

regressor.add(LSTM(units = 50,return_sequences = True))

regressor.add(Dropout(0.2))
#Adding fourth hidden layer

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))
# Adding dense layer to get the final output. The input of n-1 layer, will be the output for n layer.

regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
# Train the model

regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)
test_data = pd.read_csv('../input/Google_Stock_Price_Test.csv')

test_stock = test_data.iloc[:,1:2]

len(test_stock)
input_value = test_stock

input_value = mm.transform(input_value)

# perfor the same process, converting a 2D array to 3D

input_value = np.reshape(input_value, (20, 1, 1))


prediction = regressor.predict(input_value)

prediction = mm.inverse_transform(prediction)
# visualizing the results



plt.rcParams['figure.figsize'] = (15, 8)



plt.plot(test_stock, color = 'red', label = 'Real  Stock ')

plt.plot(prediction, color = 'green', label = 'Predicted  Stock ')

plt.title('Final Stock Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()