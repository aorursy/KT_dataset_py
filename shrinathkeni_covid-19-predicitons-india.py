#Hello everyone,



#As i have started my journey towards becoming an expert Data Scientist, i am very glad to present you  all my first piece of work.



#Covid-19 Cases in India has been on a rise since past two months and it has become very necessary to analyze as well as predict the rise of Covid-19 Cases for the forthcoming months.



#As the State of Maharashtra has seen a rapid increase of Covid-19 cases , i have focused on Prediction of Covid cases in Maharashtra for next two months.   



#So here is my small piece of work where i have Predicted  Daily Covid-19 Newcases in Maharashtra  for the month of August and September.





#I have also plotted some graphs showing: 

#1. First 10 states where the Total number of Confirmed Cases till date are the highest  

#2. Total Number of Confirmed Cases  in Maharashtra for Every Month  (Bar Plot)

#3.  Number of Newcases  Everyday  in Maharashtra (Time Series Plot)

#4. Auto Correlation plot and Partial Auto Correlation plot

#5. Actual vs Predicted Covid Cases Graph for the current time frame (Timeseries plot)

#6. Actual vs Predicted Covid Cases Graph for future time frame (TimeSeries plot)

#7. Daily Covid Expected New Cases in Maharashtra for the month August and September (Bar plot)

#8. Total Number of Newases In Maharashtra for Month of August and September (Bar Plot)



#Language Used : Python

#Libraries Used : Pandas, Seaborn, Matplotlib, Numpy , Statsmodel, Sklearn

#Model Used : Timeseries SARIMAX



#As it is my first work with Timeseries Model i request you all to go through it.

#Guidance on any modifications as well as new ideas for the model is welcomed.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as snd

%matplotlib inline

import numpy as np

import calendar

import datetime

import matplotlib.dates as mdates

import statsmodels.api as sm

import statsmodels.tsa.api as smt

import statsmodels.formula.api as smf

from io import StringIO

from sklearn.metrics import mean_squared_error

from pandas.tseries.offsets import DateOffset

import warnings as warn

warn.filterwarnings("ignore")

from matplotlib.dates import DateFormatter

pd.set_option('precision', 0)
dataindia=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

dataindia.head()
dataindiag=dataindia["State/UnionTerritory"]

dataindiag.value_counts()
total_case_india=dataindia[["State/UnionTerritory","Confirmed"]]

total_case_india=total_case_india.groupby(["State/UnionTerritory"]).agg("max").reset_index()
total_case_india=total_case_india.sort_values("Confirmed",ascending=False)
plt.figure(figsize=(12,8))

snd.barplot(x='State/UnionTerritory',y='Confirmed',data=total_case_india[:10])
dataindia=dataindia.drop(["Time","ConfirmedIndianNational","ConfirmedForeignNational","Sno","Cured","Deaths"],axis=1)
dataindia.head()
indexNames = dataindia[ dataindia['State/UnionTerritory'] != "Maharashtra" ].index

datamaha=dataindia.drop(indexNames)
datamaha.loc[97, "Confirmed"]=8

datamaha.head()
dates=pd.date_range(start="2020-03-09",freq="D",periods=len(datamaha))

datamaha["Month"]=dates.month_name()

datamaha.head()
datamaha.set_index(dates,inplace=True)

datamaha=datamaha.drop(["Date"],axis=1)

datamaha.tail()
Monthly_data=datamaha[["Month","Confirmed"]]

Monthly_data=Monthly_data.groupby(["Month"]).agg("max").reset_index()

Monthly_data=Monthly_data.sort_values("Confirmed",ascending=True)
Monthly_data
plt.figure(figsize=(7,5))

snd.barplot(x="Month", y="Confirmed",data=Monthly_data)
datamaha["Confirmed1"]=datamaha[["Confirmed"]]

datamaha["Confirmed1"]=datamaha["Confirmed1"].shift(1)

datamaha["Newcases"]=datamaha["Confirmed"] - datamaha["Confirmed1"] 

datamaha.head()
datamaha=datamaha.drop(["Confirmed","Confirmed1"],axis=1)

datamaha.head()
datamaha=datamaha.fillna(2)

datamaha["Newcases"][datamaha["Newcases"] < 0 ]= 1

datamaha.head(3)
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.figure(figsize=(10,5))

plt.grid(axis='both', alpha=.3)

#datamaha["Newcases"].plot()

plt.plot(datamaha["Newcases"])

plt.xlabel("Months")

plt.ylabel("Newcases")
rolmean=datamaha["Newcases"].rolling(window=12).mean()

rolstd=datamaha["Newcases"].rolling(window=12).std()
plt.figure(figsize=(12,8))

plt.plot(datamaha["Newcases"],label='Newcases')

plt.plot(rolmean,label="Rolling mean")

plt.plot(rolstd,label="Rolling Std")

plt.legend(loc="best")

plt.show(block=False)
decompostion=sm.tsa.seasonal_decompose(datamaha["Newcases"])

fig=decompostion.plot()
from statsmodels.tsa.stattools import adfuller 
stationary_result=adfuller(datamaha["Newcases"])

#print(stationary_result)
def stationfnct(stationary_result): 

    labels=["ADF test stat","p-value","Lags used","no of obs"]

    for value,label in zip(stationary_result,labels):

            print(label+" : "+str(value)) 

            

stationfnct(stationary_result)            
#adf_test=ADFTest(alpha=0.05)

#adf_test.should_diff(datamaha["Newcases"])

datamaha["Differencing"]=datamaha["Newcases"]-datamaha["Newcases"].shift(2)

datamaha=datamaha.dropna()

datamaha.head()
stationfnct(adfuller(datamaha["Differencing"].dropna()))

stationary_result=adfuller(datamaha["Differencing"].dropna())

print(stationary_result)
rolmean=datamaha["Differencing"].rolling(window=12).mean()

rolstd=datamaha["Differencing"].rolling(window=12).std()

plt.figure(figsize=(10,5))

plt.grid(axis='both', alpha=.3)

plt.plot(datamaha["Differencing"],label="Newcases")

plt.plot(rolmean,label="Rolling mean")

plt.plot(rolstd,label="Rolling Std")

plt.xlabel("Months")

plt.ylabel("Newcases")

plt.legend(loc="best")

plt.show(block=False)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig=plt.figure(figsize=(12,8))

ax1=fig.add_subplot(211)

fig=sm.graphics.tsa.plot_acf(datamaha["Differencing"].dropna(),lags=30,ax=ax1)

ax2=fig.add_subplot(212)

fig=sm.graphics.tsa.plot_pacf(datamaha["Differencing"].dropna(),lags=30,ax=ax2)
#from pmdarima.arima import auto_arima



#model = auto_arima(datamaha_train["Newcases"], trace=True, error_action='ignore', suppress_warnings=True)

#model.fit(datamaha_train["Newcases"])



#forecast = model.predict(n_periods=len(datamaha_test["log"]))

#forecast = pd.DataFrame(forecast,index = datamaha_test["log"].index,columns=['Prediction'])



#plot the predictions for validation set

#plt.plot(datamaha_train["log"], label='Train')

#plt.plot(datamaha_test["log"], label='Valid')

#plt.plot(forecast, label='Prediction')

#plt.show()
model=sm.tsa.statespace.SARIMAX(datamaha["Newcases"],order=(1,2,3),seasonal_order=(1,2,3,4))

resultss=model.fit()
datamaha["Expected_NewCases"]=resultss.predict(start=125,end=143,dynamic=True)

#datamaha[["Expected_Cases","Newcases"]].plot(figsize=(12,8))

plt.figure(figsize=(10,5))

plt.grid(axis='both', alpha=.3)

#plt.plot(datamaha[["Expected_Cases","Newcases"]])

plt.plot(datamaha["Newcases"],label="Newcases")

plt.plot(datamaha["Expected_NewCases"],label="Predicted Covid Cases")

plt.legend(loc="best")

plt.xlabel("Months")

plt.ylabel("Newcases")
future_dates=[datamaha.index[-1] + DateOffset(days=x) for x in range(0,60)]

future_dateset=pd.DataFrame(index=(future_dates[1:]),columns=datamaha.columns)
futureisreal=pd.concat([datamaha,future_dateset])
plt.figure(figsize=(10,5))

plt.grid(axis='both', alpha=.3)

#datamaha["Newcases"].plot()

#plt.plot(datamaha["Newcases"])

futureisreal["Expected_NewCases"]=resultss.predict(start=144,end=207)

plt.plot(futureisreal["Newcases"],label="Present Cases")

plt.plot(futureisreal["Expected_NewCases"],label="Expected Covid Cases")

plt.legend(loc="best")

#plt.plot(futureisreal[["Expected_Cases","Present"]])

plt.xlabel("Months")

plt.ylabel("Newcases")
futureisreal=futureisreal.drop(["Differencing","Newcases"],axis=1)
futureisreal=futureisreal["2020-08-03":]
datess=pd.date_range(start="2020-08-03",freq="D",periods=len(futureisreal))
futureisreal["State/UnionTerritory"]="Maharashtra"

futureisreal["Month"]=datess.month_name()

futureisreal["Day"]=datess.day

futureisreal['date'] = futureisreal['Day'].map(str)+'-'+futureisreal['Month'].map(str)
futureisreal[["date","Expected_NewCases"]].head(10)
futureisreal[["date","Expected_NewCases"]].tail()
f, axes = plt.subplots(2,1, figsize=(30, 10))



#futureisreal[['Expected_NewCases',"date"]].sort_index()[:30].plot.bar()

snd.barplot(x='date',y='Expected_NewCases',orient='v',color="red",data=futureisreal[0:29],ax=axes[0])

plt.tight_layout()

snd.barplot(x='date',y='Expected_NewCases',orient='v',color="blue",data=futureisreal[29:60],ax=axes[1])

plt.tight_layout()
Monthly_future_data=futureisreal[["Month","Expected_NewCases"]]

Monthly_future_data=Monthly_future_data.groupby(["Month"]).agg(sum).reset_index()

#Monthly_data=Monthly_data.sort_values("Confirmed",ascending=True)
Monthly_future_data.head()
plt.figure(figsize=(7,5))

snd.barplot(x="Month", y="Expected_NewCases",data=Monthly_future_data)
