#import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load data and plot data
df = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=MRTSSM448USN&scale=left&cosd=1992-01-01&coed=2020-03-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date=2020-05-16&revision_date=2020-05-16&nd=1992-01-01')
df.rename(columns={'MRTSSM448USN':'Sales'}, inplace=True)
df['DATE'] = df['DATE'].astype('datetime64[ns]') 
df.set_index('DATE', drop=True, inplace=True)

print(df.info())
display(df.head())
df.plot(figsize=(12,6))
#split into training and testing data
print('total entries = ', len(df))

test_size = 18
test_ind = len(df)-18

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

display(train.head())
print('Train shape : ',train.shape)
display(test.head())
print('Train shape : ',test.shape)
from sklearn.preprocessing import MinMaxScaler
#scale the input values to between 0 and 1
scaler = MinMaxScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#create a time series generator taking 12 values and predicting next 1 value
length = 12
generator = TimeseriesGenerator(scaled_train, 
                                scaled_train, 
                                length=length, 
                                batch_size=1)

X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
#create the test model
n_features = 1

def testmodel():
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', input_shape=(length, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model

modeltest = testmodel()
print(modeltest.summary())
from tensorflow.keras.callbacks import EarlyStopping
#add an early stop => stop training if validation loss decreases for continuously 2 times
early_stop = EarlyStopping(monitor='val_loss',patience=2)

validation_generator = TimeseriesGenerator(scaled_test,
                                           scaled_test, 
                                           length=length, 
                                           batch_size=1)
#fit the generator to model
modeltest.fit_generator(generator,epochs=20,
                    validation_data=validation_generator,
                   callbacks=[early_stop])
#create a losses dataframe and plot it
losses = pd.DataFrame(modeltest.history.history)
losses.plot()
plt.show()
#make prediction on the timestamps of test dataframe
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = modeltest.predict(current_batch)[0]
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#rescale the predictions and plot along with true values
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
display(test.head())
test.plot()
plt.show()
#scale the data
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)
#create a generator
length = 12
generator_final = TimeseriesGenerator(scaled_full_data, 
                                scaled_full_data, 
                                length=length, 
                                batch_size=1)
#create the final model
def finalmodel():
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model

modelfinal = finalmodel()
#fit generator to model
modelfinal.fit_generator(generator_final, epochs = 10)
#predict the future values
forecast = []
periods = 36

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):
    current_pred = modelfinal.predict(current_batch)[0]
    forecast.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#rescale the predictions and assign them the correct date range
forecast = scaler.inverse_transform(forecast)

forecast_index = pd.date_range(start='2020-04-01',periods=periods,freq='MS')
forecast_df = pd.DataFrame(data=forecast,index=forecast_index,columns=['Forecast'])

display(forecast_df.head())
#plot the entire dataset and predictions
ax = df.plot()
forecast_df.plot(ax=ax)
#plot a range of 6 years only for better visualisation
ax = df.plot()
forecast_df.plot(ax=ax)
plt.xlim('2017-01-01','2023-4-01')