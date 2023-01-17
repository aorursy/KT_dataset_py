# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

%matplotlib inline



# Load specific forecasting tools

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARMA

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders

from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots

# from pmdarima import auto_arima                              # for determining ARIMA orders

# from pyramid.arima import auto_arima

print("hello")

# Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller

import datetime as dt

import matplotlib.pyplot as plt
df=pd.read_csv('../input/Metro_Interstate_Traffic_Volume.csv')

df_comp=df.copy()
df_comp["year"]=df['date_time'].str.slice(0,4).astype(int)

df_comp["month"]=df['date_time'].str.slice(5,7).astype(int)

df_comp["day"]=df['date_time'].str.slice(8,10).astype(int)

df_comp["year_month_day"]=df_comp["date_time"].str.slice(0,10)
# df_comp[((df_comp["holiday"]!="None") & (df_comp["year"]==2017))].drop_duplicates(["holiday"])
df_comp.date_time = pd.to_datetime(df_comp.date_time, dayfirst = True)

df_comp.drop_duplicates(["date_time"],inplace=True)

df_comp.set_index("date_time", inplace=True)
df_comp=df_comp.loc[:,["holiday","temp","rain_1h","snow_1h","traffic_volume","year","month","day","year_month_day"]]

df_comp=df_comp[(df_comp["year"]==2018) & df_comp["month"].isin([3,4,5,6,7])]



df_comp.head()
df_comp.loc[:,'holiday']=df_comp['holiday'].str.replace("None","1")

df_comp.loc[df_comp["holiday"]!='1','holiday']='0'

df_comp.loc[:,"holiday"]=df_comp['holiday'].astype(int)
pg=df_comp[((df_comp["holiday"]==0))]

pg=pg.loc[:,["month","day"]]

for i,j in pg.iterrows():

    #print(str(i))

    for m,n in df_comp.iterrows():

        sj=str(i)

        sj1=sj[:10]

        if(sj1==str(n['year_month_day'])):

            df_comp.loc[m,'holiday']=0

            





def getday(x):

    return int(pd.to_datetime(x).weekday())

df_comp["week_day"]=df_comp["year_month_day"].map(lambda x: getday(x))
type(df_comp.iloc[0]["week_day"])
for i,j in df_comp.iterrows():

    if(int(df_comp.loc[i,"week_day"])>4 or df_comp.loc[i,"holiday"]==0 ):

        df_comp.at[i,"weekday_encoded"]=0

    else:

        df_comp.at[i,"weekday_encoded"]=1

df_comp[(df_comp["year_month_day"]=="2018-07-05")]


df_comp["weekday_encoded"]=df_comp["weekday_encoded"].astype(int)



# df_comp["temp"]=df_comp["temp"]/100
df_comp.head(300).traffic_volume.plot(figsize=(20,5))
df_comp.tail()
df_comp=df_comp.asfreq('h')

print(df_comp.isna().sum())

df_comp=df_comp.fillna(method='ffill')

print(df_comp.isna().sum())

# df_comp.index.freq = 'h'

print(df_comp.index.freq)
df_comp=df_comp.loc[:,["traffic_volume","weekday_encoded","month","year"]]
df_comp
df_comp_train=df_comp[df_comp["month"].isin([3,4,5,6])]

df_comp_test=df_comp[df_comp["month"].isin([7])]
len(df_comp_train)

len(df_comp_test)
df_comp_train.iloc[1056:1104,0]=df_comp_train.iloc[888:936,0].values
df_comp_train["traffic_volume"].plot(figsize=(20,5))


from statsmodels.tsa.stattools import adfuller



def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("AAAAAAAAAAAAAAAa",result[1])

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")

        

        

adf_test(df_comp_train["traffic_volume"])
result = seasonal_decompose(df_comp_train['traffic_volume'],model="multiplicative")
result.seasonal.plot(figsize=(20,5));
result.trend.plot(figsize=(20,5),legend=True,color="purple")
# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler()

# y=StandardScaler()

# x=df_comp_train[["traffic_volume"]]

# y=sc.fit_transform(x)

# df_comp_train["traffic_scaled"]=y

df_comp_train.tail()
# auto_arima(df_comp_train["traffic_volume"],seasonal=True,m=24).summary()
# model_sarima_no_exog=ARIMA(df_comp_train["traffic_volume"],order=(168,0,1))

# result_sarima_no_exog=model_sarima_no_exog.fit()

# result_sarima_no_exog.summary()
# df_comp_train["traffic_volume"]=df_comp.traffic_volume.diff()
# adf_test(df_comp_train["traffic_volume"])
model_sarima_no_exog=SARIMAX(df_comp_train["traffic_volume"],order=(5,1,6),seasonal_order=(1,1,1,24),enforce_invertibility=False)

result_sarima_no_exog=model_sarima_no_exog.fit()

result_sarima_no_exog.summary()
start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_no_exog=result_sarima_no_exog.predict(start=start,end=end,dynamic=False)
df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_no_exog.plot()
from sklearn.metrics import mean_squared_error

from statsmodels.tools.eval_measures import rmse



error_mse_noex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_no_exog)

error_rmse_noex = rmse(df_comp_test['traffic_volume'], predictions_sarima_no_exog)



print("MSE",error_mse_noex)

print("RMSE",error_rmse_noex)
model_sarima_exog_weekend = SARIMAX(df_comp_train['traffic_volume'],order=(1,0,1),seasonal_order=(1,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend = model_sarima_exog_weekend.fit()

results_sarima_exog_weekend.summary()
start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend = results_sarima_exog_weekend.predict(start=start, end=end, dynamic=False,exog=df_comp_test[["weekday_encoded"]]).rename('SARIMA(1,0,1)(1,0,1,24) Predictions')
df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend.plot()
error_mse_ex_weekend = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend)

error_rmse_ex_weekend= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend)



print("MSE",error_mse_ex_weekend)

print("RMSE",error_rmse_ex_weekend)
model_sarima_exog_weekend_2 = SARIMAX(df_comp_train['traffic_volume'],order=(2,1,1),seasonal_order=(1,1,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend_2 = model_sarima_exog_weekend_2.fit()

results_sarima_exog_weekend_2.summary()
start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend_2 = results_sarima_exog_weekend_2.predict(start=start, end=end, dynamic=False,exog=df_comp_test[["weekday_encoded"]]).rename('SARIMAX(1,0,1)(1,0,1,24) Predictions')
df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend_2.plot()
error_mse_ex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_2)

error_rmse_ex= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_2)



print("MSE",error_mse_ex)

print("RMSE",error_rmse_ex)
model_sarima_exog_weekend_3 = SARIMAX(df_comp_train['traffic_volume'],order=(1,1,7),seasonal_order=(1,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend_3 = model_sarima_exog_weekend_3.fit()

results_sarima_exog_weekend_3.summary()
start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend_3=results_sarima_exog_weekend_3.predict(start=start,end=end,dynamic=False,exog=df_comp_test[["weekday_encoded"]])
df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend_3.plot()
error_mse_ex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_3)

error_rmse_ex= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_3)



print("MSE",error_mse_ex)

print("RMSE",error_rmse_ex)
model_sarima_exog_weekend_4 = SARIMAX(df_comp_train['traffic_volume'],order=(2,1,2),seasonal_order=(1,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend_4 = model_sarima_exog_weekend_4.fit()

results_sarima_exog_weekend_4.summary()
start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend_4=results_sarima_exog_weekend_4.predict(start=start,end=end,dynamic=False,exog=df_comp_test[["weekday_encoded"]])



df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend_4.plot()
error_mse_ex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_4)

error_rmse_ex= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_4)



print("MSE",error_mse_ex)

print("RMSE",error_rmse_ex)
model_sarima_exog_weekend_5 = SARIMAX(df_comp_train['traffic_volume'],order=(5,1,5),seasonal_order=(2,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend_5 = model_sarima_exog_weekend_5.fit()

results_sarima_exog_weekend_5.summary()



start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend_5=results_sarima_exog_weekend_5.predict(start=start,end=end,dynamic=False,exog=df_comp_test[["weekday_encoded"]])



df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend_5.plot()
error_mse_ex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_5)

error_rmse_ex= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_5)



print("MSE",error_mse_ex)

print("RMSE",error_rmse_ex)
model_sarima_exog_weekend_6_1 = SARIMAX(df_comp_train['traffic_volume'],order=(1,1,3),seasonal_order=(1,1,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend_6_1 = model_sarima_exog_weekend_6_1.fit()

results_sarima_exog_weekend_6_1.summary()



start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend_6_1=results_sarima_exog_weekend_6_1.predict(start=start,end=end,dynamic=False,exog=df_comp_test[["weekday_encoded"]])



df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend_6_1.plot()
error_mse_ex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_6_1)

error_rmse_ex= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_6_1)



print("MSE",error_mse_ex)

print("RMSE",error_rmse_ex)
model_sarima_exog_weekend_6 = SARIMAX(df_comp_train['traffic_volume'],order=(6,1,6),seasonal_order=(1,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])

results_sarima_exog_weekend_6 = model_sarima_exog_weekend_6.fit()

results_sarima_exog_weekend_6.summary()



start=len(df_comp_train)

end=len(df_comp_train)+len(df_comp_test)-1

predictions_sarima_exog_weekend_6=results_sarima_exog_weekend_6.predict(start=start,end=end,dynamic=False,exog=df_comp_test[["weekday_encoded"]])



df_comp_test["traffic_volume"].plot(figsize=(20,5))

predictions_sarima_exog_weekend_6.plot()
error_mse_ex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_6)

error_rmse_ex= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend_6)



print("MSE",error_mse_ex)

print("RMSE",error_rmse_ex)