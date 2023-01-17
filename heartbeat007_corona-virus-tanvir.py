import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df.head()
df
df = df.drop('Sno',axis=1)
df.head()
df=df.replace(to_replace ="China", 

                 value ="Mainland China")

df['Country'].value_counts().plot.bar()
df['Country'].value_counts().plot.pie()
print('Total Confirmed Cases:',df['Confirmed'].sum())

print('Total Deaths: ',df['Deaths'].sum())

print('Total Recovered Cases: ',df['Recovered'].sum())
affected = df['Confirmed'].sum()

died = df['Deaths'].sum()

recovered = df['Recovered'].sum()
import matplotlib.pyplot as plt
labels = ['affected','died','recovered']

sizes = [15, 30, 45, 10]

explode = (.1, .1, .1)  # only "explode" the 2nd slice (i.e. 'Hogs')



plt.pie([affected,died,recovered],explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()
df1 = df.groupby(["Country"]).sum()
df1.plot.bar()
china_df = df.loc[df['Country'] == 'Mainland China']
china_df['Province/State'].value_counts().plot.pie()
china_df['Province/State'].value_counts().plot.bar()
print('Total Confirmed Cases in china:',china_df['Confirmed'].sum())

print('Total Deaths in china: ',china_df['Deaths'].sum())

print('Total Recovered Cases i china: ',china_df['Recovered'].sum())
affected = china_df['Confirmed'].sum()

died = china_df['Deaths'].sum()

recovered = china_df['Recovered'].sum()
labels = ['affected','died','recovered']

sizes = [15, 30, 45, 10]

explode = (.1, .1, .1)  # only "explode" the 2nd slice (i.e. 'Hogs')



plt.pie([affected,died,recovered],explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()
## top ten countried

# top10_countries=df[['Country']].nlargest(10,['Confirmed']).reset_index(drop=True)



highest_affected = df[['Province/State','Confirmed','Last Update']].nlargest(10, 'Confirmed')

highest_affected.set_index('Last Update')[['Confirmed']].plot.bar()
new_df = df.set_index('Country')
highest_death = new_df[['Deaths','Last Update']].nlargest(10, 'Deaths').plot.bar()
df.groupby(["Province/State"]).sum().plot()
highest_affected = df[['Province/State','Confirmed','Last Update']]
china_df.groupby(["Province/State"]).sum()[['Confirmed']].plot.bar()
china_df.groupby(["Province/State"]).sum()[['Deaths']].plot.bar()
china_df.groupby(["Province/State"]).sum()[['Recovered']].plot.bar()
df.groupby(["Date"]).sum()['Confirmed'].plot()
df.groupby(["Date"]).sum()['Deaths'].plot()
df.groupby(["Date"]).sum()['Recovered'].plot()
df.isnull().sum()
df3 = df[['Last Update','Deaths']]
df3.set_index('Last Update').plot()
df.isnull().sum()
df.corr()
import seaborn as sns
sns.heatmap(df.corr())
df3
df3.columns = ['ds', 'y']
df3
from fbprophet import Prophet
m = Prophet()

m.fit(df3)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast
m.plot_components(forecast)
m.plot(forecast,uncertainty=True)
df4 = df[['Last Update','Confirmed']]
df4.columns = ['ds', 'y']
m = Prophet()

m.fit(df4)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast
m.plot_components(forecast)
m.plot(forecast,uncertainty=True)
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

df.head()
from sklearn.model_selection import train_test_split
df.Date = pd.to_datetime(df.Date)

#df = df.set_index("Month")

df.head()
df2 = df[['Date','Confirmed','Deaths','Recovered']]
df2.head()
X = df2[['Date']]

Y = df2[['Confirmed']]
x_train,x_test,y_train,y_test = train_test_split(X,Y)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

scaled_xtrain_data = scaler.transform(x_train)

scaled_xtest_data = scaler.transform(x_test)

scaled_xtest_data
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = scaled_xtrain_data.shape[1]

n_features= 1

generator = TimeseriesGenerator(scaled_xtrain_data, scaled_xtrain_data, length=n_input, batch_size=1)
generator
scaled_xtrain_data.shape
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM



lstm_model = Sequential()

lstm_model.add(LSTM(200, activation='relu', input_shape=(1,1)))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(200,activation='relu'))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])



lstm_model.summary()
lstm_model.fit_generator(generator,epochs=20)

import matplotlib.pyplot as plt

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.xticks(np.arange(0,21,1))

plt.plot(range(len(losses_lstm)),losses_lstm);