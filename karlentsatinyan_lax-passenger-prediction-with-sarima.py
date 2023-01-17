import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

#print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import statsmodels.api as sm

from statsmodels.tsa.statespace.sarimax import SARIMAX
file = pd.read_csv("../input/los-angeles-international-airport-passenger-traffic-by-terminal.csv")

file2 = file.copy()

file2.head()
print("Number of null values: \n")

print(file2.isnull().sum())
fig, (ax1,ax2,ax3)=plt.subplots(1,3, figsize=(20,10))

ax1.pie(file2['Terminal'].value_counts(), labels=file2.Terminal.unique(), autopct='%1.1f%%', startangle=180)

ax2.pie(file2['Arrival_Departure'].value_counts(), labels=file2.Arrival_Departure.unique(), autopct='%1.1f%%', startangle=180)

ax3.pie(file2['Domestic_International'].value_counts(), labels=file2.Domestic_International.unique(), autopct='%1.1f%%', startangle=180)

ax1.set_title("TERMINALS", fontsize=14), ax2.set_title("DEP./ARRIV.", fontsize=14), ax3.set_title("INT./DOM.", fontsize=14)

plt.draw()
plt.figure(figsize=(14,5))

sns.countplot(y="Terminal", hue="Arrival_Departure", data=file2)

plt.title("arrival/departures of passengers for each terminal", fontsize=16)

plt.show()

plt.figure(figsize=(14,5))

sns.countplot(y="Terminal", hue="Domestic_International", data=file2)

plt.title("domestic vs international flights for each terminal", fontsize=16)

plt.legend(loc="best")

plt.show()
file2.groupby("ReportPeriod").sum().head()
x=file2.groupby("ReportPeriod").sum()

file3=pd.DataFrame(x,columns=["Passenger_Count"])



file3.index=pd.to_datetime(file3.index, format='%Y-%m-%d').strftime('%Y-%m')

file3.index.name ="Date"

file3.columns=["Number"]

file3.head()
fig,ax=plt.subplots(figsize=(15,5))

ax.plot(file3)

ax.set(xlabel="Date", ylabel="Number")

ax.set_title("Total number of passengers per period", fontsize=16)

plt.show()
#Changing the order of the indices

df=pd.DataFrame(file3)

df_ascending=df.sort_index(axis=0, ascending=True, inplace=False)

df=df_ascending
#splitting into training and testing data

percent_training=0.8

split_point=round(len(df)*percent_training)

training, testing=df[0:split_point], df[split_point:]

print("The shape of training set is: ",training.shape)

print("The shape of testing set is: ",testing.shape)
#To check the stationarity of data

'''from statsmodels.tsa.stattools import adfuller

def check_adfuller(ts):

    result=adfuller(ts, autolag="AIC")

    print("test statistics: ", result[0])

    print("p-value: ", result[1])

    print("Critical value: ", result[4])

check_adfuller(training["Number"])'''
training_diff=training.diff().dropna()

plt.figure(figsize=(10,5))

plt.plot(training_diff)

plt.title("After taking first difference", fontsize=14)

plt.draw()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig,ax = plt.subplots(2,1,figsize=(15,8))

plt.suptitle("ACF/PACF of the first differencing", fontsize=16)

fig = sm.graphics.tsa.plot_acf(training_diff, lags=20, ax=ax[0])

fig = sm.graphics.tsa.plot_pacf(training_diff, lags=20, ax=ax[1])

plt.show()
'''

Rules for SARIMA model selection from ACF/PACF plots

These are all rule of thumbs, not an exact science for picking the number of each parameters in SARIMA(p,d,q)(P,D,Q)[S]. It is an art in picking good parameters from the ACF/PACF plots. The following rules also apply to ARMA and ARIMA models.



Identifying the order of differencing:



d=0 if the series has no visible trend or ACF at all lags is low.



d≥1 if the series has visible trend or positive ACF values out to a high number of lags.



Note: if after applying differencing to the series and the ACF at lag 1 is -0.5 or more negative the series may be overdifferenced.



Note: If you find the best d to be d=1 then the original series has a constant trend. A model with d=2 assumes that the original series has a time-varying trend.



Identifying the number of AR and MA terms

p is equal to the first lag where the PACF value is above the significance level.



q is equal to the first lag where the ACF value is above the significance level.



Identifying the seasonal part of the model:



S is equal to the ACF lag with the highest value (typically at a high lag).



D=1 if the series has a stable seasonal pattern over time.



D=0 if the series has an unstable seasonal pattern over time.



Rule of thumb: d+D≤2



P≥1 if the ACF is positive at lag S, else P=0.



Q≥1 if the ACF is negative at lag S, else Q=0.



Rule of thumb: P+Q≤2

'''
model_fit=SARIMAX(training, order=(6, 1, 6), seasonal_order=(1,0,0, 6), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
K=len(testing)

forecast=model_fit.forecast(K)

#forecast=np.exp(forecast)

forecast.index=testing.index

plt.figure(figsize=(25,8))

plt.plot(df,"b")

plt.plot(forecast, "r")

plt.xlabel("Date")

plt.ylabel("Number")

plt.xticks(rotation=90)

plt.autoscale(enable=True, axis='x', tight=True)

plt.axvline(x=testing.index[0], linestyle='--', color="gray")

plt.draw()



from sklearn.metrics import mean_squared_error

print("RMSE is: ", round(np.sqrt(mean_squared_error(testing,forecast))))
'''my_list=[]

for p in [6,8,10,12]:

    for q in [4,6,12]:

        for P in [0,1,2]:

            for Q in [0,1]:

                for S in [6, 12]:

                    for D in [0,1]:

                        for d in [1,2]:

                            model_fit=SARIMAX(training, order=(p,1,q), seasonal_order=(P,1,Q,S), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

                            forecast=model_fit.forecast(32)

                            my_list.append(np.sqrt(mean_squared_error(testing,forecast)))

                            print(p,d,q,P,D,Q,S,np.sqrt(mean_squared_error(testing,forecast)))

print(np.min(my_list))'''

model_fit=SARIMAX(training, order=(10, 1, 6), seasonal_order=(2, 1, 0, 6), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
forecast=model_fit.forecast(K)

forecast.index=testing.index

plt.figure(figsize=(25,8))

plt.plot(df,"b")

plt.plot(forecast, "r")

plt.xlabel("years")

plt.ylabel("number")

plt.xticks(rotation=90)

plt.autoscale(enable=True, axis='x', tight=True)

plt.axvline(x=testing.index[0], linestyle='--', color="gray")

plt.draw()

print("New RMSE is: ", round(np.sqrt(mean_squared_error(testing,forecast))))
t=pd.date_range(start="2019-04", end="2020-04", freq="M")

future=pd.DataFrame(index=t, columns=file3.columns)

future.index=pd.to_datetime(future.index, format='%Y-%m-%d').strftime('%Y-%m')

df=pd.concat([file3, future], axis=0)
model_fit.predict(start=160, end=171)
df.Number.iloc[-12:]=model_fit.predict(start=160, end=171).values
plt.figure(figsize=(25,5))

#b=df["Number"].iloc[-24:-12]

year_ahead_forecast=df["Number"].iloc[-12:]



plt.plot(training, label="training")

plt.plot(testing, "k", label="testing")

plt.plot(forecast, "r", label="validation")

plt.plot(year_ahead_forecast,"g", label="year ahead forecast")



plt.legend()

plt.axvline(x=testing.index[0], linestyle='--', color="gray")

plt.axvline(x=year_ahead_forecast.index[0], linestyle='--', color="gray")

plt.xlabel("years")

plt.ylabel("number")

plt.xticks(rotation=90)

plt.show()