import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


beer_data=pd.read_csv("../input/time-series-datasets/monthly-beer-production-in-austr.csv")
beer_data
fig,ax=plt.subplots()
ax.plot(beer_data["Month"][0:20],beer_data["Monthly beer production"][0:20])
ax.set_xticklabels(beer_data["Month"],rotation=90)
plt.title(" First 20 record plots of Monthly Beer Production")
plt.show()
#print(type(beer_data))
#print(beer_data.info())
beer_data["Month"]=pd.to_datetime(beer_data["Month"]).dt.date

beer_data.set_index("Month")
beer_data1 = beer_data


#Plotting a 3 day moving average

rolling= beer_data1["Monthly beer production"].rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
# plot original and transformed dataset
fig,ax=plt.subplots()
fig.set_size_inches(14,12, forward=True)
ax.plot(beer_data1["Month"],rolling_mean,color="red")
ax.plot(beer_data1["Month"],beer_data1["Monthly beer production"],color="Blue")
ax.set_xticklabels(beer_data["Month"],rotation=90)
plt.title(" 3 day moving average for Monthly Berr Production")
plt.show()



#Using beer_data to check for stationarity usinf adfuller test
test_result=adfuller(beer_data["Monthly beer production"])
print(" Printing AD-Fuller Test results for Monthly Berr Production: \n")
for name,rez in zip(["ADF Test Stat","p-value","#Lags used","# of obs. used"],test_result):
    print(name," :" +str(rez))
    
#Since p-value is greater than 0.05, we reject accept the null hypothesis that the TS is non-stationary

#Making non stationary into stationary:
#Differencing:

beer_data["First diff"]=beer_data["Monthly beer production"]-beer_data["Monthly beer production"].shift(12)
print(" Beer Data after First Difference: \n")
print(beer_data.head(15))

#Adfuller retest
test_result1=adfuller(beer_data["First diff"].dropna())
print("\n\n")
for name,rez in zip(["ADF Test Stat","p-value","#Lags used","# of obs. used"],test_result1):
    print(name," :" +str(rez))

#P value<0.05 hence Ho rejected. TS is NOW stationary
#Plotting autocorrelation 



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.subplot(211)
plot_acf(beer_data["First diff"].iloc[12:], ax=plt.gca())
plt.subplot(212)
plot_pacf(beer_data["First diff"].iloc[12:], ax=plt.gca())
plt.show()

##p=1,d=1,q=1

##ARIMA model
from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(beer_data["Monthly beer production"].iloc[12:],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()
fig,ax=plt.subplots()
fig.set_size_inches(14,12, forward=True)
##
ax.plot(beer_data["Month"][12:],beer_data["First diff"][12:],color="blue")
ax.plot(beer_data["Month"][13:],model_fit.fittedvalues, color= 'red')
ax.set_xticklabels(beer_data["Month"],rotation=90)
plt.title(" Predicted VS Actual Values for ARIMA")
plt.show()


#Actual prediction:
'''from pandas.tseries.offsets import DateOffset
future_dates=[beer_data["Month"][-1:0] +DateOffset(months=x) for x in range(0,20)]
future_dates'''

## Appending Dates for predictions first
beer_data["Month"].tail()
date=pd.date_range(start='09/01/1995',periods=15, freq='MS')

beer_data["Month"].tail()
prod=np.zeros(15)
data=pd.DataFrame({"Month":date,
                   "Monthly beer production":prod})

future_data=pd.concat([beer_data,data],ignore_index=True)

#print(future_data)
future_data.set_index('Month')
print(future_data.head())
#uture_data["Predict"]=model_fit.predict(start=475,end=490,dynamic=True)
#uture_data
#rint(dir(model_fit))
start = 2
end = len(future_data)- 1
future_data["Predict"]=model_fit.predict(start,end,dynamic=True)
future_data
future_data.set_index('Month')
fig,ax=plt.subplots()
ax.plot(future_data["Month"],future_data["Predict"],color="Red")
ax.plot(future_data["Month"],future_data["Monthly beer production"],color="Blue")
plt.show()

print("date index")
future_data.set_index('Month')
print(future_data.index)

#SARIMA

import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(beer_data["Monthly beer production"],order=(1,1,1),seasonal_order=(1,1,1,12))
results=model.fit()
beer_data["fore"]=results.predict(start=350,end=475,dynamic=True)
#eer_data[["Monthly beer production","fore"]].plot(figsize=(12,8))
print(results.summary())

fig,ax=plt.subplots()
ax.plot(beer_data["Month"],beer_data["Monthly beer production"],color="blue",label="original")
ax.plot(beer_data["Month"],beer_data["fore"],color="red",label="ori=pred")
ax.set_xticklabels(beer_data["Month"],rotation=90)
plt.title(" Actual VS Predicted plot for SARIMAX")
plt.show()


#Actual pred
start=2
end = len(future_data)- 1
future_data["Predict"]=results.predict(start=476,end=490,dynamic=True,exog=None)
future_data
print("prediction")
fig,ax=plt.subplots()
ax.plot(future_data["Month"],future_data["Predict"],color="Red")
ax.plot(future_data["Month"][0:476],future_data["Monthly beer production"][0:476],color="Blue")
ax.set_xticklabels(future_data["Month"],rotation=90)
plt.title(" Actual AND Predicted plot for SARIMAX")
plt.show()
#print(future_data.index)