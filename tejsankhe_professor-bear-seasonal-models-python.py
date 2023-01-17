import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from plotnine import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import date
data = pd.read_csv("../input/colorado_river.csv")
print(data.columns)
data.head(10)

plt.plot(data.index,data["flow"], 'black')
plt.scatter(data.index,data["flow"],c='black')
plt.show()
#  it is difficult to identify seasonality trends here
# So,aggregate the data by month to better understand this trend
ggplot(data, aes(x='month', y='flow/1000'))+geom_line()+theme_bw()+facet_wrap('~year')
# add a little style
# visualize year by year
# modifying data to timeseries for creating forecast models
def date_str(month, year):
    return str(int(year))+"-"+str(int(month))
mydata = data[0:72]
for index, row in mydata.iterrows():
    mydata.loc[index, 'date'] = date_str(row['month'], row['year'])
mydata['date'] = pd.to_datetime(mydata['date'], infer_datetime_format=True)
mydata = mydata.drop(['month', 'year'], axis=1)
mydata = mydata.set_index('date')
#Since we hypothesize that there is seasonality,
#we can take the seasonal difference (create a variable that gives the 12TH differences), then look at the ACF and PACF.
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
diff_12 = difference(mydata['flow'],12)
plot_pacf(diff_12,lags = 48)
plot_acf(diff_12, lags =48)
print(acf(diff_12, nlags = 48))
print(pacf(diff_12, nlags = 48))
#we see that for both the ACF and PACF we have significant autocorrelation at seasonal (12, 24, 36) lags. The ACF has a cluster around 12, 
#and not much else besides a tapering pattern throughout. Further, the PACF also has spikes on two multiples of S, AR(2)
# Try, ARIMA (1,0, 0) x (2, 1, 0)12
my_order = (1, 0, 0)
my_seasonal_order = (2, 1, 0, 12)
mod1 = sm.tsa.statespace.SARIMAX(mydata, order=my_order, seasonal_order=my_seasonal_order)
result = mod1.fit()
plt.plot(result.resid)
plt.title("Standardized Residuals")
plt.xlabel("Time")
plt.show()
plt.clf()
plot_acf(result.resid)
print("Normal Q-Q Plot of Std Residual")
qqplot(result.resid)
lags_= range(5,36)
a = acorr_ljungbox(result.fittedvalues, lags = lags_)

plt.scatter(lags_,a[1]*1000)
plt.title("p values for Ljung-Box statistic")
plt.xlabel("lag")
plt.ylabel("p value")
plt.show()
plt.clf()
result.summary()
mod = ARIMA(mydata,order=(1, 0, 0))
# Fit the Model
mod_fit = mod.fit(disp=0)
plt.plot(mod_fit.fittedvalues,mod_fit.resid)
plt.scatter(mod_fit.fittedvalues,mod_fit.resid)
plt.xlabel("fitted(mod)")
plt.ylabel("mod$residuals")
plt.plot(mydata, c='red')
plt.plot(mod_fit.fittedvalues, c='blue')
plt.ylabel("mod$x")
plt.xlabel("Time")
# Now, we have a reasonable prediction, we can forecast the model, say 24 months into the future.
forecast = result.predict(start = 72, end = 95, dynamic= True) 
plt.plot(mydata[0:72])
plt.plot(forecast, c= "red")
forecast