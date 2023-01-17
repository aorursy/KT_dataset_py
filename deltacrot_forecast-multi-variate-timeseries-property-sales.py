import os # accessing directory structure

print(os.listdir('../input/property-sales'))
# We will clean, explore and visualise the raw data first

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

df=pd.read_csv('../input/property-sales/raw_sales.csv')

df=df[df.propertyType=='house'] #Let's limit the scope of this problem to houses only

df['datesold'] = pd.to_datetime(df['datesold'])

df=df.drop(columns=['postcode', 'propertyType'])

df = df[np.abs(df.price - df.price.mean()) <= (5.0 * df.price.std())] # Clean the outliers

%matplotlib inline 

plt.figure(figsize=(15,5))

plt.plot_date(df['datesold'], df['price'], xdate=True, markersize=1)
# We will group and visualise the data by the number of bedrooms

import seaborn

from  matplotlib import pyplot

df=df[df.bedrooms>1] # 0 and 1 bedrooms are not relevant to houses and should be discarded

_bedrooms=df['bedrooms'].unique().sort()

fg = seaborn.FacetGrid(data=df, hue='bedrooms', hue_order=_bedrooms, aspect=2, height=8)

fg.map(pyplot.scatter, 'datesold', 'price', alpha=.7, s=5).add_legend()

#This data may be useful for forecasts via a neural network.
#The data in ma_lga_12345.csv has been resampled to quartely intervals with a median aggregator outside of this notebook

#We will load it and visualise it first

df=pd.read_csv('../input/property-sales/ma_lga_12345.csv')

df=df[df.type=='house'] #Let's limit the scope of this problem to houses only

df['saledate'] = pd.to_datetime(df['saledate'])

df.tail()
#Pivot the data so we can feed it into the model

df=df.pivot(index='saledate', columns='bedrooms', values='MA').interpolate(method='linear', limit_direction='both')

df.tail()
#Plot the data

%matplotlib inline 

df.plot(figsize=(15,5))

#It is evident that 2 bedroom curve before 2009 is not an accurate representation of the actual median price.

#It is not possible for a 2 bedroom median price to be above that of 3 bedroom median price.

#This is due to low number of sales in that timeframe, which skews the calculated median price.
# Let's see what we can do to correct the overlapping lines for 2 and 3 br data

def separate_series(df):

    columns = list(df) 

    for col in columns: 

        if col== columns[-1]:

            break

        #Calculate average difference between 2 and 3 bedrooms for the recent  1/3 of the dataframe

        diff_mean= (df[col+1][:-int(len(df)/3)]-df[col][:-int(len(df)/3)]).mean()

        #Where 2 br price is higher than that of 3 br, replace it with 3 br price minus the diff

        #do_they_intersect = False if df[col].loc[df[col] >= df[col+1], ].empty else True

        #if do_they_intersect:

        df.loc[df[col] > df[col+1]-diff_mean, col] = df[col+1]-diff_mean

    return df

df=separate_series(df)

df.plot(figsize=(15,5))
#if you are after monthly frequency we can resample the quarterly data to monthly by

df_monthly = df.resample('M').interpolate(method='linear', limit_direction='both').astype(int)

#if you are after weekly or daily just replace 'M' with 'W' or 'D' respectively

df_monthly.tail()

df_monthly.plot(figsize=(15,5))
#Coint Johansen test for all # bedrooms columns

from statsmodels.tsa.vector_ar.vecm import coint_johansen

coint_johansen(df,-1,1).eig
#Split the data into train and test

train = df[:int(0.9*(len(df)))]

test = df[int(0.9*(len(df))):]



#Fit the model

from statsmodels.tsa.vector_ar.var_model import VAR



model = VAR(endog=train, freq='Q-DEC')

model_fit = model.fit()



#Forecast based on train data

forecast = model_fit.forecast(model_fit.endog, steps=len(test))
#Convert forecast data to a dataframe we can use

cols = df.columns

pred = pd.DataFrame(index=test.index, data=forecast,columns=[cols])

pred=pred.astype(int)

pred.tail()
#Plot actuals (df) and forecast (pred) on the same chart

ax = df.plot()

pred.plot(ax=ax,figsize=(15,5))
#Show percentage difference between the last period of the forecast and actual series. Less is better.

((pred.iloc[-1].values-pred.iloc[0].values)/df.iloc[-1].values)*100
#Here is the mean absolute percentage error

import numpy as np

for col in df.columns:

    print (str(col) +' bedrooms ' + str(np.mean(np.abs((df[col].iloc[-len(pred):].values - pred[[col]].values) / df[col].iloc[-len(pred):].values)) * 100))
#Forecast actuals

model = VAR(endog=df, freq='Q-DEC')

model_fit = model.fit()

forecast_period=8

prediction = model_fit.forecast(model_fit.endog, steps=forecast_period)

cols = df.columns

forecast_index = pd.DatetimeIndex(start ='2019-09-30', freq ='Q', periods=forecast_period) 

pred = pd.DataFrame(index=forecast_index, data=prediction,columns=[cols])

ax = df.plot()

pred.plot(ax=ax,figsize=(15,5))