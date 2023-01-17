# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv")

df.head()
df.describe()
df.info()
print('Number of rows:', df.shape[0])

print('Number of Airports:', df['Airport name'].nunique())

print('First Year:', df['Year'].min())

print('Last Year:', df['Year'].max())
months = df.columns[~df.columns.isin([

    'Airport name',

    'Airport coordinates',

    'Whole year', 'Year'

])]

mapping= {v: k for k,v in enumerate(months, start=1)} 
time_series = df.melt(

    id_vars=['Airport name', 'Year'],

    value_vars=months,

    var_name='Month',

    value_name="cargo_&_parcel"

)

time_series.head()
time_series['date'] = time_series.apply(lambda x: f"{x['Year']}-{mapping[x['Month']]:02d}", axis=1)



time_series['date'] = pd.to_datetime(time_series['date']) 


# Covert type

time_series = (

    time_series

    .rename(columns={'Airport name': 'airport', 'value': 'cargo_&_parcel'})

    .drop(columns=['Year', 'Month'])

)

time_series.head()
rpass=time_series.groupby(["date"])["cargo_&_parcel"].sum().loc[:'2020-01-01'] 

rpass.head()
import plotly.express as px

fig = px.line(rpass.reset_index(),x="date",y="cargo_&_parcel",title="TREND OF AIRLINES IN RUSSIA")

fig.update_xaxes(rangeslider_visible=True)

fig.show()
plt.figure(figsize=(20,7))

rpassrolling=time_series.groupby(["date"])["cargo_&_parcel"].sum().loc[:'2019-12-01'] 

rolmean = rpassrolling.rolling(window=13).mean()

rolstd = rpassrolling.rolling(window=13).std()

original=plt.plot(rpassrolling,color="blue",label="Original")

mean=plt.plot(rolmean,color="red",label="Mean")

std=plt.plot(rolstd,color="black",label="Std")

plt.legend()

plt.title("ROLLING TEST")

plt.show()
from statsmodels.tsa.stattools import adfuller

def adfuller_test(passenger):

    res=adfuller(passenger)

    labels=["ADF TEST STATISTICS","P-VALUE","LAGS USED","NUMBER OF OBSERVATION USED"]

    for value,label in zip(res,labels):

        print(label+' : '+str(value))

    if(res[1]<=0.05):

        print("Stationary")

    else:

        print("Not Stationary")

adfuller_test(rpass.reset_index()["cargo_&_parcel"].dropna())
rpass=rpass.reset_index()

rpass["seasonal difference"]=rpass["cargo_&_parcel"]-rpass["cargo_&_parcel"].shift(13)

rpass.tail()
plt.figure(figsize=(20,5))

adfuller_test(rpass.reset_index()["seasonal difference"].dropna())

px.line(rpass,y="seasonal difference",x="date",title="TREND AFTER SHIFTING")
plt.figure(figsize=(10,5))

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(rpass["cargo_&_parcel"])

plt.show()
import statsmodels as sm

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig=plt.figure(figsize=(12,8))

ax1=fig.add_subplot(211)

fig=plot_acf(rpass["seasonal difference"].dropna(),lags=40,ax=ax1)

ax2=fig.add_subplot(212)

fig=plot_pacf(rpass["seasonal difference"].dropna(),lags=40,ax=ax2)
#p=0,q=10,d=1

from statsmodels.tsa.arima_model import ARIMA
#rpass.set_index("date",inplace=True)

import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(rpass["cargo_&_parcel"],order=(0,1,10),seasonal_order=(0,1,10,12))

results=model.fit()
rpass["Future Prediction"]=results.predict(start=130,end=154,dynamic=True)

rpass[["cargo_&_parcel","Future Prediction"]].plot(figsize=(20,8))

plt.title("TESTING PREDICTED VALUE WITH DATASET")
from pandas.tseries.offsets import DateOffset

future_dates=[rpass.index[-1]+ DateOffset(months=x) for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=rpass.columns)

future_datest_df.tail()
future_df=pd.concat([rpass,future_datest_df])

future_df['Future Prediction']=results.predict(start=156,end=200,dynamic=True)

future_df[["cargo_&_parcel","Future Prediction"]].plot(figsize=(20,8))