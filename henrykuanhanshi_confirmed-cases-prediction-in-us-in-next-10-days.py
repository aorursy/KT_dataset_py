import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
%matplotlib inline
dataset = pd.read_csv("../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")
us_data = dataset[dataset['Country/Region'].isin(['US'])]
us_data = us_data.drop(['Country/Region','Province/State','Lat','Long'], axis=1)
date=us_data.columns.tolist()
us_data_nd=us_data.values
us_data_sum=us_data_nd.sum(axis=0)
us_data_sum=us_data_sum.reshape(1,87)
print(us_data_sum)
X,Y = [], []
for i in range(77):
    X.append(us_data_sum[0,i:i+10])
    Y.append(us_data_sum[0,i+10])
X=np.array(X)
Y=np.array(Y)

X_Train = np.reshape(X, (X.shape[0],X.shape[1],1))
def create_model():
  model = Sequential()
  model.add(LSTM(50,input_shape=(10,1),return_sequences=True,activation='relu'))
  model.add(LSTM(100,activation='relu'))
  model.add(Dense(1))
  return model
model=create_model( )
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_Train,Y, epochs=200, verbose=1)
predictions=us_data_sum
output=[]
for i in range(10):
  prediction=model.predict(np.array(predictions[0][-10:]).reshape(1,10,1))
  output.append(prediction)
  predictions=np.append(predictions,prediction,axis=1)
us_data_sum_list=[]
for i in range(us_data_sum.shape[1]):
  us_data_sum_list.append(int(us_data_sum[0][i]))
for i in range(len(output)):
  output[i]=float(output[i])
plt.figure(figsize=(30,15)) 
plt.plot(date,us_data_sum_list,color='forestgreen')
plt.xlabel("Date",fontsize = 20)
plt.ylabel("Number of Confirmed Cases",fontsize = 20)
plt.plot(['4/21/20','4/22/20','4/23/20','4/24/20','4/25/20','4/26/20','4/27/20','4/28/20','4/29/20','4/30/20'],output,color='red')
plt.legend(['Confirmed', 'Predicted Confirmed'], loc='upper left', fontsize='xx-large')
plt.title('Prediction Model of Covid-19 in US in next 10 days',fontsize=30)
plt.xticks(rotation=60,fontsize=10)
plt.yticks(fontsize=20)
plt.grid()
plt.show()
