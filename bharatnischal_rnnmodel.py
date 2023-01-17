# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
dataset = pd.read_csv("../input/cottonprices/cotton-prices-historical-chart-data.csv")
dataset.info()
dataset.head(10)
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))
plt.plot(dataset)
training_set = dataset.iloc[:9898].values
test_set = dataset.iloc[9898:].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
X_test = []
dataset = dataset.iloc[:].values
for i in range(len(training_set), len(dataset)):
    X_test.append(dataset[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_test_values = regressor.predict(X_test)
predicted_test_values = sc.inverse_transform(predicted_test_values)
predicted_test_values = predicted_test_values
# Visualising the results
import matplotlib.pyplot as plt

plt.plot(test_set, color = 'red', label = 'Real Cotton Price')
plt.plot(predicted_test_values, color = 'blue', label = 'Predicted Cotton Price')
plt.title('Cotton Price Prediction')
plt.xlabel('Time')
plt.ylabel('Cotton Price')
plt.legend()
plt.savefig('prices.jpg')
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
regressor.save_weights("model.h5")
# from keras.models import model_from_json
# # load json and create model
# json_file = open('../input/trained-values/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# regressor = model_from_json(loaded_model_json)
# # load weights into new model
# regressor.load_weights("../input/trained-values/model.h5")
# print("Loaded model from disk")
regressor2 = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor2.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50, return_sequences = True))
regressor2.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50, return_sequences = True))
regressor2.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50))
regressor2.add(Dropout(0.2))
# Adding the output layer
regressor2.add(Dense(units = 1))

# Compiling the RNN
regressor2.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor2.fit(X_train, y_train, epochs = 100, batch_size = 32)
X_test2 = []
dataset = dataset.iloc[:].values
for i in range(len(training_set), len(dataset)):
    X_test2.append(dataset[i-60:i, 0])
X_test2 = np.array(X_test2)
X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))
predicted_test_values2 = regressor2.predict(X_test2)
predicted_test_values2 = sc.inverse_transform(predicted_test_values2)
predicted_test_values2 = predicted_test_values2
# Visualising the results
import matplotlib.pyplot as plt

plt.plot(test_set, color = 'red', label = 'Real Cotton Price')
plt.plot(predicted_test_values2, color = 'blue', label = 'Predicted Cotton Price')
plt.title('Cotton Price Prediction')
plt.xlabel('Time')
plt.ylabel('Cotton Price')
plt.legend()
plt.savefig('prices.jpg')
model_json = regressor2.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
regressor2.save_weights("model2.h5")
# from keras.models import model_from_json
# # load json and create model
# json_file = open('../input/model2/model2.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# regressor2 = model_from_json(loaded_model_json)
# # load weights into new model
# regressor2.load_weights("../input/model2/model2.h5")
# print("Loaded model from disk")
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.layers import Dropout
plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/cottonprices/cotton-prices-historical-chart-data.csv")
df.info()
df.shape
plt.figure(figsize=(16,8))
plt.title('Cotton Price History')
plt.plot(df)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()
data = df
dataset = data.values#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 
#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)
#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model
model.fit(x_train, y_train, batch_size=4, epochs=10)
#Test data set
test_data = scaled_data[training_data_len - 60: , : ]#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test[0]
#Convert x_test to a numpy array 
x_test = np.array(x_test[0:2])
#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Convert x_test to a numpy array 
x_test = np.array(x_test)
#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling
#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse
# Visualising the results
import matplotlib.pyplot as plt

plt.plot(data[training_data_len:], color = 'red', label = 'Real Cotton Price')
plt.plot(predictions, color = 'blue', label = 'Predicted Cotton Price')
plt.title('Cotton Price Prediction')
plt.xlabel('Time')
plt.ylabel('Cotton Price')
plt.legend()
plt.savefig('prices.jpg')
#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train)
plt.plot(valid)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
#Show the valid and predicted prices
valid
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model2.h5")
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.layers import Dropout
plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/cottonprices/cotton-prices-historical-chart-data.csv")
df.info()
plt.figure(figsize=(16,8))
plt.title('Cotton Price History')
plt.plot(df)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()
def modelTraining(time,neurons,optimizer,batch,epochs,train_data):
    
    x_train=[]
    y_train = []
    for i in range(time,len(train_data)):
        x_train.append(train_data[i-time:i,0])
        y_train.append(train_data[i,0])
    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=neurons, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=neurons, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    #Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    #Train the model
    model.fit(x_train, y_train, batch_size=batch, epochs=epochs)
    
    #Test data set
    test_data = scaled_data[training_data_len - time: , : ]#Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(time,len(test_data)):
        x_test.append(test_data[i-time:i,0])
        
    #Convert x_test to a numpy array 
    x_test = np.array(x_test)
    
    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling
    
    #Plot/Create the data for the graph
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid['Predictions'] = predictions#Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train)
    plt.plot(valid)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('img'+str(time)+str(neurons)+str(optimizer)+str(batch)+str(epochs)+'.png')
    plt.show()
    
    #save the model
    model_json = model.to_json()
    with open("model"+str(time)+str(neurons)+str(optimizer)+str(batch)+str(epochs)+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model"+str(time)+str(neurons)+str(optimizer)+str(batch)+str(epochs)+".h5")
    
    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print(rmse)
    return rmse
result_error = []
result_parameters = []

time_values = [30,45,60]
neurons_values = [40,50]
optimizer_values = ['adam','rmsprop']
batch_values = [4,16,32]
epochs_values = [10,15]

dataset = df.values#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets


for t in time_values:
    for n in neurons_values:
        for o in optimizer_values:
            for b in batch_values:
                for e in epochs_values:
                    result_parameters.append([t,n,o,b,e])
                    result_error.append(modelTraining(t,n,o,b,e,train_data))

print(result_parameters)
print(result_error)
min_index = result_error.index(min(result_error))
result_parameters[min_index]
plt.plot(result_error)
for i in range(len(result_error)):
    result_parameters[i].append(result_error[i])
conclusion = pd.DataFrame(result_parameters,columns=['learning_time','neurons','optimizer','batch_size','epochs','RMS'])
conclusion
conclusion.sort_values('RMS')
# Training on 45 days
result_error_45 = []
result_parameter_45 = [[45,50,'rmsprop',4,10],[45,50,'adam',16,15]]

result_error_45.append(modelTraining(45,50,'rmsprop',4,10,train_data))
result_error_45.append(modelTraining(45,50,'adam',16,15,train_data))
print(result_parameter_45)
print(result_error_45)
for i in range(len(result_error_45)):
    result_parameter_45[i].append(result_error_45[i])
conclusion_45 = pd.DataFrame(result_parameter_45,columns=['learning_time','neurons','optimizer','batch_size','epochs','RMS'])
conclusion_45
# Training on 60 days
result_error_60 = []
result_parameter_60 = [[60,50,'rmsprop',4,10],[60,50,'adam',16,15]]

result_error_60.append(modelTraining(60,50,'rmsprop',4,10,train_data))
result_error_60.append(modelTraining(60,50,'adam',16,15,train_data))
print(result_parameter_60)
print(result_error_60)
for i in range(len(result_error_60)):
    result_parameter_60[i].append(result_error_60[i])
conclusion_60 = pd.DataFrame(result_parameter_60,columns=['learning_time','neurons','optimizer','batch_size','epochs','RMS'])
conclusion_60
results = pd.concat([conclusion,conclusion_45,conclusion_60], axis=0)
results
results.sort_values('RMS')
from keras.models import model_from_json
# load json and create model
json_file = open('../input/bestrnnparameters/model3040rmsprop410.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor2 = model_from_json(loaded_model_json)
# load weights into new model
regressor2.load_weights("../input/bestrnnparameters/model3040rmsprop410.h5")
print("Loaded model from disk")
#Getting the models predicted price values
predictions = regressor2.predict(x_test) 
print(predictions)
predictions = scaler.inverse_transform(predictions)#Undo scaling
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.layers import Dropout
plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/dataset3/Price.csv")
df.info()
df.head(5)
df = df[df['Total Value (Lacs)']!=0]
df['price'] = df['Total Value (Lacs)']/df["Quantity (000's)"]
plt.figure(figsize=(16,8))
plt.title('Cotton Price History')
plt.plot(df['price'])
# plt.xticks(list(range(0,len(df))),df['Date'])
plt.ylabel('Price',fontsize=18)
plt.show()
def modelTraining(time,neurons,optimizer,batch,epochs,train_data):
    
    x_train=[]
    y_train = []
    for i in range(time,len(train_data)):
        x_train.append(train_data[i-time:i,0])
        y_train.append(train_data[i,0])
    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=neurons, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=neurons, return_sequences=False))
    model.add(Dropout(0.2))
#     model.add(LSTM(units=neurons, return_sequences=False))
#     model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    #Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print("Reached here")
    
    #Train the model
    model.fit(x_train, y_train, batch_size=batch, epochs=epochs)
    
    #Test data set
    test_data = scaled_data[training_data_len - time: , : ]#Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(time,len(test_data)):
        x_test.append(test_data[i-time:i,0])
        
    #Convert x_test to a numpy array 
    x_test = np.array(x_test)
    
    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling
    
    #Plot/Create the data for the graph
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid['Predictions'] = predictions#Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('img2-'+str(time)+str(neurons)+str(optimizer)+str(batch)+str(epochs)+'.png')
    plt.show()
    
    #save the model
    model_json = model.to_json()
    with open("model2-"+str(time)+str(neurons)+str(optimizer)+str(batch)+str(epochs)+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model2-"+str(time)+str(neurons)+str(optimizer)+str(batch)+str(epochs)+".h5")
    
    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print(rmse)
    return rmse
df = pd.DataFrame(df['price'])
dataset = df.values#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets
print("Root mean Square error =",modelTraining(60,50,'rmsprop',4,10,train_data))

result_error = []
result_parameters = []

time_values = [60]
neurons_values = [40,50]
optimizer_values = ['adam','rmsprop']
batch_values = [4,16,32]
epochs_values = [10,15]

# dataset = df.values#Get /Compute the number of rows to train the model on
# training_data_len = math.ceil( len(dataset) *.8) 

# #Scale the all of the data to be values between 0 and 1 
# scaler = MinMaxScaler(feature_range=(0, 1)) 
# scaled_data = scaler.fit_transform(dataset)

# #Create the scaled training data set 
# train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets


for t in time_values:
    for n in neurons_values:
        for o in optimizer_values:
            for b in batch_values:
                for e in epochs_values:
                    result_parameters.append([t,n,o,b,e])
                    result_error.append(modelTraining(t,n,o,b,e,train_data))

print(result_parameters)
print(result_error)
min_index = result_error.index(min(result_error))
result_parameters[min_index]
plt.plot(result_error)
for i in range(len(result_error)):
    result_parameters[i].append(result_error[i])
result_parameters
conclusion = pd.DataFrame(result_parameters,columns=['learning_time','neurons','optimizer','batch_size','epochs','RMS'])
conclusion
conclusion.sort_values('RMS')
# Training on 45 days
result_error_45 = []
result_parameter_45 = [[45,40,'rmsprop',4,15],[45,50,'adam',4,15]]

result_error_45.append(modelTraining(45,40,'rmsprop',4,15,train_data))
result_error_45.append(modelTraining(45,50,'adam',4,15,train_data))
print(result_parameter_45)
print(result_error_45)
for i in range(len(result_error_45)):
    result_parameter_45[i].append(result_error_45[i])
conclusion_45 = pd.DataFrame(result_parameter_45,columns=['learning_time','neurons','optimizer','batch_size','epochs','RMS'])
conclusion_45
# Training on 30 days
result_error_30 = []
result_parameter_30 = [[30,40,'rmsprop',4,15],[30,50,'adam',4,15]]

result_error_30.append(modelTraining(30,40,'rmsprop',4,15,train_data))
result_error_30.append(modelTraining(30,50,'adam',4,15,train_data))
print(result_parameter_30)
print(result_error_30)
for i in range(len(result_error_30)):
    result_parameter_30[i].append(result_error_30[i])
conclusion_30 = pd.DataFrame(result_parameter_30,columns=['learning_time','neurons','optimizer','batch_size','epochs','RMS'])
conclusion_30
results = pd.concat([conclusion,conclusion_45,conclusion_30], axis=0)
results
results.sort_values('RMS') 
