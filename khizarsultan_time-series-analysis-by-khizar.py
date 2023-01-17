import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# data = https://drive.google.com/open?id=1Qm1L8izAJ-8NAt2ZROmtAVSf1CNEPyGH

id = "1Qm1L8izAJ-8NAt2ZROmtAVSf1CNEPyGH"
from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials 
auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)
download = drive.CreateFile({'id':id})

download.GetContentFile('household_power_consumption.txt')

print(f"data has been download to google colab")
df = pd.read_csv('household_power_consumption.txt', sep = ';', 

                 parse_dates={'datetime':['Date','Time']},

                 na_values=['nan','?'],

                 index_col = 'datetime'

                 )
df.head(100)
df.shape
df.describe(include='all')
df.info()
# remove null values

df.isnull().sum()
# bearable outliers

df.Global_active_power.plot(kind='box') 
# df.fillna({

#     'Global_active_power':np.mean()

# })

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline
cat_pipe = Pipeline([

       ('imputer', Imputer(strategy='median'))              

])

cleaned_data = cat_pipe.fit_transform(df)
clean_df = pd.DataFrame(cleaned_data,columns=df.columns)

clean_df.isnull().sum()
clean_df.set_index(df.index, inplace = True)
# now explore the monthly wise gloabl active power

monthly_resampled_data_mean = clean_df.Global_active_power.resample('M').mean()

monthly_resampled_data_sum = clean_df.Global_active_power.resample('M').sum()



monthly_resampled_data_mean.plot(title = 'Global_active_power resampled over month for mean')

plt.tight_layout()

plt.show() 



monthly_resampled_data_sum.plot(title = 'Global_active_power resampled over month for sum', color = 'red')

plt.tight_layout()

plt.show() 

r2 = clean_df.Global_reactive_power.resample('M').agg(['mean', 'std'])

r2.plot(subplots = True, title='Global_reactive_power resampled over day', color='red')

plt.show()
r2 = clean_df.Voltage.resample('M').agg(['mean', 'std'])

r2.plot(subplots = True, title='Voltage resampled over month', color='red')

plt.show()
# sns.pairplot(clean_df, kind = 'reg')

sns.pairplot(clean_df)

plt.show()

# KDE Plot described as Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable. 

# It depicts the probability density at different values in a continuous variable. 

# We can also plot a single graph for multiple samples which helps in more efficient data visualization
# global active power and gloabl density are directly proportional to each other

clean_df.Global_reactive_power.resample('W').mean().plot(color='y', legend=True)

clean_df.Global_active_power.resample('W').mean().plot(color='r', legend=True)

clean_df.Sub_metering_1.resample('W').mean().plot(color='b', legend=True)

clean_df.Global_intensity.resample('W').mean().plot(color='g', legend=True)

plt.show()
clean_df.Global_reactive_power.resample('W').mean().plot(kind = 'hist', color='y', legend=True)

clean_df.Global_active_power.resample('W').mean().plot(kind = 'hist', color='r', legend=True)

clean_df.Sub_metering_1.resample('W').mean().plot(kind = 'hist', color='b', legend=True)

clean_df.Global_intensity.resample('W').mean().plot(kind = 'hist',color='g', legend=True)

plt.show()
# find the percentage change with the previous row 

data_returns = clean_df.pct_change()

data_returns
sns.jointplot(x='Voltage', y='Global_active_power', data=data_returns)  

plt.show()




def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]

	dff = pd.DataFrame(data)

	cols, names = list(), list()

	# input sequence (t-n, ... t-1)

	for i in range(n_in, 0, -1):

		cols.append(dff.shift(i))

		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)

	for i in range(0, n_out):

		cols.append(dff.shift(-i))

		if i == 0:

			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

		else:

			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together

	agg = pd.concat(cols, axis=1)

	agg.columns = names

	# drop rows with NaN values

	if dropnan:

		agg.dropna(inplace=True)

	return agg

 
resamble_data_hours = clean_df.resample('h').mean() 
resamble_data_hours.shape
# its time to normalize the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

scaled = scaler.fit_transform(resamble_data_hours)
scaled # normalized data
# frame as supervised learning

reframed = series_to_supervised(scaled, 1, 1)
reframed
# we only need var1(t) (Global Active Power) Output variabel so,

# we should delete other(var2(t)	var3(t)	var4(t)	var5(t)	var6(t)	var7(t)) output varibles



reframed.drop(reframed.columns[[8,9,10,11,12,13]], inplace=True, axis=1)
reframed.head(10)
# from sklearn.model_selection import train_test_split

# split into train and test sets

values = reframed.values



n_train_time = 365*24

train = values[:n_train_time, :]

test = values[n_train_time:, :]

##test = values[n_train_time:n_test_time, :]

# split into input and outputs



train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]



# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 

# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,LSTM,Conv1D,MaxPool1D

from tensorflow.keras.optimizers import SGD,Adam
X_train.shape[2]
model = Sequential()

model.add(LSTM(100, input_shape = (X_train.shape[1],X_train.shape[2])))



model.add(Dropout(0.2))

# model.add(LSTM(80))

# model.add(Dropout(0.3))



model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
# now fit the model

history = None

history = model.fit(x=X_train,y=Y_train,batch_size=70,epochs=50,verbose=2,validation_data=(X_test,Y_test),shuffle=False)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Loss of Training and Validation")

plt.xlabel("Epoches")

plt.ylabel("Loss")

plt.legend(['Train','Test'], loc = 'upper right')

plt.show()
from sklearn.metrics import mean_squared_error
# invert predictions

# make a prediction

yhat = model.predict(X_test)



test_X = X_test.reshape((X_test.shape[0], 7))



# # invert scaling for forecast

# test_X[:,-6:] mean only features



inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0] #out put variable



# # invert scaling for actual



test_y = Y_test.values.reshape((len(Y_test), 1))

inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

# calculate RMSE



rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
# without inverse, its is normlize

y_predict = model.predict(X_test)

mse = np.sqrt(mean_squared_error(Y_test,y_predict))

print(f"The Mean Squarred error is: {mse}")
sample = list(range(200))

plt.figure(figsize=(10,5))

plt.plot(sample,inv_y[:200],marker = '.', label = 'Actual')



plt.plot(sample,inv_yhat[:200],marker = '.', label = 'Prediction')

plt.ylabel('Global_active_power', size=15)

plt.xlabel('Time step', size=20)

plt.legend(fontsize=15)

plt.title("This is not Overfitting the data")

plt.show()