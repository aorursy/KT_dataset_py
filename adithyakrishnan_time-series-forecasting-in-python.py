import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input/"))

sales=pd.read_excel("../input/forecasting_case_study.xlsx")
sales['Post-Shipment Invoice Date']=pd.to_datetime(sales['Post-Shipment Invoice Date'])
sales.info()
#missing values therefore 
s=sales.loc[sales['Post-Shipment Invoice Date'].isnull()].index
sales=sales.drop(sales.index[s],axis=0)
print (sales.info())
# All quantity zero therefore now delete all these missing values.


sales["d_1"]=sales.apply(lambda x: (x['Post-Shipment Invoice Date']-x['SO Date']).days,axis=1)
#Print all the  unique column elements. 
print (sales['End Customer Code'].nunique())
sales.groupby ("region",as_index="False")[['Qty(Net)']].mean()
plt.figure(figsize=(18,8))
sns.barplot(data=sales,x="region",y="Qty(Net)")
#Qty has more shipments from region 3,10,12,23,24
sales.groupby ("plant",as_index="False")[['Qty(Net)']].mean()
plt.figure(figsize=(10,8))
sns.barplot(data=sales,x="plant",y="Qty(Net)")
#plant 3 has the major shipments
sales.groupby ("Technology",as_index="False")[['Qty(Net)']].mean()
plt.figure(figsize=(10,8))
sns.barplot(data=sales,x="Technology",y="Qty(Net)")
#Technology 8 is leading in sales. t1 is also good 
#As we need to show the aggregate result
sales['dow']=sales['Post-Shipment Invoice Date'].dt.dayofweek
sales['year']=sales['Post-Shipment Invoice Date'].dt.year
sales['month']=sales['Post-Shipment Invoice Date'].dt.month
sales['date']=sales['Post-Shipment Invoice Date'].dt.date

sales.info()
#Now plotting the trends for dates,month,year,dow. 
s=sales.groupby(['date'])[['Qty(Net)']].sum()
s1=sales.groupby(['year'])[['Qty(Net)']].sum()
s2=sales.groupby(['month'])[['Qty(Net)']].sum()
s3=sales.groupby(['dow'])[['Qty(Net)']].sum()

fig, ax = plt.subplots(2, 2,figsize=(15,5))


ax[0][0].plot(s)
ax[0][1].plot(s1)
ax[1][0].plot(s2)
ax[1][1].plot(s3)

#Year has an increasing trend 
#date of week and month also shows a interesting patterns
#date also shows some kind of  timeseries. 

sales['date_int'] = sales['date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

sales.drop(['Post-Shipment Invoice Date','SO Date','date'],inplace=True,axis=1)
sales_final=sales.groupby(['year','month'])[['Qty(Net)','year','month']].sum()
sales_final1= sales.groupby(['year','month'], as_index=False)['Qty(Net)'].sum()

#sales_final1['final_date']=sales_final1.apply(lambda x : x['month']+'/'+x['year'])
sales_final1['final_date']="28-"+sales_final1["month"].map(str) + '-'+ sales_final1["year"].map(str)
sales_final1=sales_final1[['final_date','Qty(Net)']]
sales_final1['final_date']=pd.to_datetime(sales_final1['final_date'])
sales_final1
sns.lineplot(data=sales_final1,x='final_date',y='Qty(Net)')

#trend of past 36 months.
from pandas.tools.plotting import lag_plot
plot_lags=25
rows=int(plot_lags/5)
cols=int(plot_lags/5)
fig,axes=plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(10,10))
fig.set_figwidth(plot_lags)
fig.set_figwidth(plot_lags)
count=1
for i in range (rows):
    for j in range(cols):
        lag_plot(sales_final1["Qty(Net)"],lag=count,ax=axes[i,j])
        count+=1
        
        
# The date is very distorted as it does not have some major relationship . Hence it would not give us 
# a good accuracy 
decompose=sales_final1
decompose.index=decompose["final_date"]
decompose=decompose[["Qty(Net)"]]
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(decompose["Qty(Net)"],freq=10)
sales_final1

decomposition.plot()
#The data for 2015 was very less. Hence we not see very well for 2015.  
# Hence we can eliminate 2015 as well

#We could see a good Trend from 2016-03
#We could also see a great seasonality captured as it has a cyclicity . 
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pandas import Series
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
X = np.array(sales_final1['Qty(Net)'])
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
#We could see the value of ADF Statistic is less than 1%. and p value is less 
#than 0.05 hence we cannot reject the null hypothesis.
#Hence the time series is stationary 
sales_final2=sales_final1[['Qty(Net)']]

lag_acf = acf(sales3, nlags=20)
lag_pacf = pacf(sales3, nlags=20)

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(sales_final2)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(sales_final2)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(sales_final2)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(sales_final2)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(sales_final3,lags=30,alpha=0.05)
plot_pacf(sales_final3,lags=30,alpha=0.05)

sales_final2
#First Two Values are very less and could be outliers because they are not telling us any significant things

sales_final3=sales_final2[2:]


aic=[]
pdq=[]
for p in range(6):
    for d in range(2):
        for q in range(4):
            try:
                arima_mod=ARIMA(sales_final3.values,order=(p,d,q)).fit(transparams=True)

                x=arima_mod.aic

                x1= p,d,q
                print (x1,x)

                aic.append(x)
                pdq.append(x1)
            except:
                pass
print (min(aic))


#Minimum AIC value will give us the best results for (p,d,q)
#Did

import statsmodels.api as sm

X = sales_final3.values
history = [x for x in X]
predictions_out = list()
for t in range(len(X)):
    model1 = ARIMA(history, order=(1,1,0))   #0,1,2
    model1_fit = model1.fit(disp=0)
    output = model1_fit.forecast()
    yhat = output[0]
    predictions_out.append(yhat)
    obs = X[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(X, predictions_out)
print('Test MSE: %.3f' % error)



plt.plot(X)
plt.plot(predictions_out, color='red')
plt.show()



#Finding out the forcasted values of Jan 19,Feb 19,March 19

X = sales_final3.values
predictions_out=[]
history = [x for x in X]
for t in range(3):
    model1 = ARIMA(history, order=(1,1,0))   #0,1,2
    model1_fit = model1.fit(disp=0)
    output = model1_fit.forecast()
    yhat = output[0]
    predictions_out.append(yhat)
    history.append(yhat)
predictions=sales_final3.values
predictions=np.concatenate((predictions,predictions_out)) #Concatinating the data of 3 more months that we are forecasting
plt.plot(sales_final3.values)
plt.show()
plt.plot(predictions, color='red')
plt.show()

'''While plotting the values are not showing any significant change. It is keeping the Chages 
very less . Hence going for SARIMA. Earlier we also saw the cyclicity of seasonal trend using 
decomposition'''


import itertools
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
warnings.filterwarnings("ignore") # specify to ignore warning messages
import statsmodels.api as sm 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(sales_final3,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
#Above we found out the best SARIMAX parameters therefore we are using those configurations
import statsmodels.api as sm 
mod = sm.tsa.statespace.SARIMAX(sales_final3,
                                order=(0, 2, 2),
                                seasonal_order=(0, 2, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()


yhat = results.predict(start=1, end=40)
print (yhat)
#yhat=yhat.reshape(-1,1)
#SARIMA is showing negative values and i am stuck.
y=sales_final3.values
#yy=np.concatenate((y,yhat))
plt.plot(sales_final3.values)
plt.show()
plt.plot(yhat.values, color='red')
plt.show()
# Therefore the predicted values are. 
print (yhat)
