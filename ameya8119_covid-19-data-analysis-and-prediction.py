# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

#LOADING THE DATASET

pd.options.display.max_columns = 500

pd.options.display.max_rows = 500

X = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
print(X.shape)
print(X.dtypes)
X.head(10)
#converting the columns into correct form 

X['ObservationDate'] = pd.to_datetime(X['ObservationDate'])

X['Last Update'] = pd.to_datetime(X['Last Update'])

print(X.dtypes)
#made a new dataset by extracting the obeservation date

#by selecting the observation date as the end of each month the entire data for the month is selected 

#grouped them according to country of region 

#only the confirmed cases was extracted 

#sum of each month using .sum()

#took only the first 10 countries and sorted the total no of confirmed cases in each country in non-ascending order 



First_Month = X[X['ObservationDate']=='01/31/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
Second_Month = X[X['ObservationDate']=='02/29/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
Third_Month = X[X['ObservationDate']=='03/31/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
Fourth_Month = X[X['ObservationDate']=='04/30/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
Fifth_Month = X[X['ObservationDate']=='05/30/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
Sixth_Month = X[X['ObservationDate']=='06/30/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
Seventh_Month = X[X['ObservationDate']=='07/30/2020'].groupby('Country/Region')['Confirmed'].sum().head(10).sort_values(ascending = False)
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (20,10))

#fig.tight_layout uses to seperate each of the given subplots so that there is no overlapment 

fig.tight_layout(pad=5.0)



ax[0,0].bar(First_Month.index,First_Month)

ax[0,0].set_xticklabels(First_Month.index,rotation = 45)

ax[0,0].title.set_text('January')



ax[0,1].bar(Second_Month.index,Second_Month, color = 'g')

ax[0,1].set_xticklabels(Second_Month.index,rotation = 45)

ax[0,1].title.set_text('February')



ax[0,2].bar(Third_Month.index,Third_Month, color = 'c')

ax[0,2].set_xticklabels(Third_Month.index,rotation = 45)

ax[0,2].title.set_text('March')



ax[0,3].bar(Fourth_Month.index,Fourth_Month, color = 'r')

ax[0,3].set_xticklabels(Fourth_Month.index,rotation = 45)

ax[0,3].title.set_text('April')



ax[1,0].bar(Fifth_Month.index,Fifth_Month)

ax[1,0].set_xticklabels(Fifth_Month.index,rotation = 45)

ax[1,0].title.set_text('May')



ax[1,1].bar(Sixth_Month.index,Sixth_Month, color = 'g')

ax[1,1].set_xticklabels(Sixth_Month.index,rotation = 45)

ax[1,1].title.set_text('June')



ax[1,2].bar(Seventh_Month.index,Seventh_Month, color = 'c')

ax[1,2].set_xticklabels(Third_Month.index,rotation = 45)

ax[1,2].title.set_text('July')



plt.show()

Confirmed_cases = X[X['Country/Region']=='India'].groupby('ObservationDate')['Confirmed'].sum()

Deaths_cases = X[X['Country/Region']=='India'].groupby('ObservationDate')['Deaths'].sum()

Recovered_cases = X[X['Country/Region']=='India'].groupby('ObservationDate')['Recovered'].sum()
plt.figure(figsize = (20,10))

plt.plot(Confirmed_cases,color = 'c',marker = 'v',label = 'Confirmed_cases')

plt.plot(Deaths_cases,color = 'g',marker = 'x',label = 'Deaths')

plt.plot(Recovered_cases,color = 'b',marker = 'o',label = 'Recovered')

plt.xlabel('Covid-19 Cases count')

plt.ylabel('total No of Affected people')

plt.legend

plt.show()
#sorting out the data

X_India = X[X['Country/Region']=='India']

arranged_dataset = X_India.groupby(['ObservationDate']).agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})

arranged_dataset.tail(15)

arranged_dataset.shape
training_set = arranged_dataset.iloc[:,0:1].values



#Date Preprocessing

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)



#Creating data structure with 45 timesteps 

X_train = []

y_train = []

for i in range(45,180):

    X_train.append(training_set_scaled[i-45:i, 0])

    y_train.append(training_set_scaled[i, 0])

    

X_train, y_train = np.array(X_train) , np.array(y_train)   



#Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



#Initialize the RNN

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



regressor = Sequential()



#Add first LSTM layer and Dropout regularisation

regressor.add(LSTM(units =50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))



#Adding second layer

regressor.add(LSTM(units =50, return_sequences = True))

regressor.add(Dropout(0.2))



#Adding third layer

regressor.add(LSTM(units =50, return_sequences = True))

regressor.add(Dropout(0.2))



#Adding fourth layer

regressor.add(LSTM(units =50))

regressor.add(Dropout(0.2))



#Output layer

regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mse')



#Training the model

#Taking a small batch size because the number of data points to train on is limited

regressor.fit(X_train, y_train, epochs = 50, batch_size = 5)
#Prediction and visualization

real_confirmed_cases = arranged_dataset.iloc[170:213,0:1].values



X_test = []



for i in range(170,213):

    X_test.append(training_set_scaled[i-45:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_confirmed_cases = regressor.predict(X_test)

predicted_confirmed_cases = sc.inverse_transform(predicted_confirmed_cases)
plt.figure(figsize = (12,8))

plt.plot(real_confirmed_cases, color='c',marker = 'o', label = 'Real Confirmed Cases')

plt.plot(predicted_confirmed_cases, color='g',marker = 'o', label = 'Predicted Number of Cases')

plt.title('Coronavirus Forecasting Trend in Cases')

plt.xlabel('Days')

plt.ylabel('Number of Cases')

plt.legend()

plt.show()