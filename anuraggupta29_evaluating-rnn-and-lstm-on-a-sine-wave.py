#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
#creating a sine wave
x = np.linspace(0,50, 501)
y = np.sin(x)
print('length of x : ',len(x))
print('length of y : ',len(y))
print(x[:10])
print(y[:10])
#plot the sine wave
plt.figure(figsize=(15,6))
plt.plot(x,y)
plt.show()
#create a dataframe with these values
df = pd.DataFrame(data=y, index=x, columns=['sine'])
display(df.head())
#the train df will be used for training the model and the test df will be used for comparing with the output
all_points = len(df)
test_ratio = 0.1
train_end_point = int(all_points*(1-test_ratio))
test_begin_point = train_end_point+1

print(train_end_point)
print(test_begin_point)
#define train and test dataframe
train = df.iloc[:train_end_point+1]
test = df.iloc[test_begin_point:]

#view the dataframes
display(train.head())
display(train.tail())

display(test.head())
display(test.tail())
scaler = MinMaxScaler()
#scale the values(which are between -1 and 1) to between 0 and 1
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
print(scaled_train.max(), scaled_train.min())
print(scaled_test.max(), scaled_test.min())
#TimeseriesGenerator(inputs, targets, length, batch_size)

#input => our train series of values

#target => our train series of values
#beacuse we are not predicting an output for each input
#the output will also be a part of the series based on previous inputs

#length => The total number of points we will use to predict the next point(s) in each run of generator
#batch_size => the total number of points we will predict in each run of  generator

len_input_series = 50 #use 50 points to predict next point(s)
batch_size = 1 #predict next 1 point based on input point(s)
n_features = 1 #we have only 1 feature (sine of x)

generator_train = TimeseriesGenerator(scaled_train, 
                                scaled_train, 
                                length=len_input_series, 
                                batch_size=batch_size)
#create the function for rnn model
def rnnmodel():
    model = Sequential()
    
    model.add(SimpleRNN(units = len_input_series, input_shape = (len_input_series, n_features)))
    #Still doubtful what units is, but is often kept equal to len of input series
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model
#view model summary
modelrnn = rnnmodel()
modelrnn.summary()
#fit the generator to model
modelrnn.fit_generator(generator_train,epochs=5)
#get the losses and plot them for each epoch
losses = pd.DataFrame(modelrnn.history.history)
losses.plot()
#predict the output for the entire range of test values (50 points)
test_predictions = []
first_eval_batch = scaled_train[-len_input_series:]
current_batch = first_eval_batch.reshape((1, len_input_series, n_features))

for i in range(len(test)):
    current_pred = modelrnn.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)
#transform the output back to its original range of values (i.e between -1 and 1)
true_predictions = scaler.inverse_transform(test_predictions)

#add predictions to test dataframe
test['Predictions'] = true_predictions

#display predictions and plot
display(test.head())
test.plot()
#scale the data of entire df
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)
#create a generator for entire df, taking 50 values to predict next 1 value
generator_all = TimeseriesGenerator(scaled_full_data, 
                                scaled_full_data, 
                                length=len_input_series, 
                                batch_size=batch_size)
#define function for entire model
def finalmodel():
    model = Sequential()
    
    model.add(SimpleRNN(units=len_input_series, input_shape=(len_input_series, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model
#initialize model and fit generator
modelfinal = finalmodel()
modelfinal.fit_generator(generator_all, epochs=10)
#predict the next 501 points
forecast = []

first_eval_batch = scaled_full_data[-len_input_series:]
current_batch = first_eval_batch.reshape((1, len_input_series, n_features))

for i in range(len(df)):
    current_pred = modelfinal.predict(current_batch)[0]
    forecast.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#rescale the predicted values bw -1 and 1
forecast = scaler.inverse_transform(forecast)

#index for predicted values
forecast_index = np.arange(50.1,100.2,step=0.1)

#plot the praph
plt.plot(df.index,df['sine'])
plt.plot(forecast_index,forecast)
plt.show()
#stop training if validation loss increases for continuously 3 times
early_stop = EarlyStopping(monitor='val_loss',patience=3)

#define generator and validation generator
generator_lstm = TimeseriesGenerator(scaled_train,scaled_train,
                               length=len_input_series-1,batch_size=1)

validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                          length=len_input_series-1,batch_size=1)
#define the lstm model function
def lstmmodel():
    model = Sequential()
    
    model.add(LSTM(units=len_input_series, input_shape=(len_input_series, n_features)))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model
#create the lstm model
modellstm = lstmmodel()

#fit the generator to the model
modellstm.fit_generator(generator,
                    epochs=15,
                    validation_data = validation_generator,
                    callbacks = [early_stop])
#make prediction over the x values in test df
#note : we r not actually using the values in test df
test_predictions = []

first_eval_batch = scaled_train[-len_input_series:]
current_batch = first_eval_batch.reshape((1, len_input_series, n_features))

for i in range(len(test)):
    current_pred = modellstm.predict(current_batch)[0]
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#make predictions and add to test df and plot the graph
true_predictions = scaler.inverse_transform(test_predictions)
test['LSTM Predictions'] = true_predictions
test.plot(figsize=(12,8))