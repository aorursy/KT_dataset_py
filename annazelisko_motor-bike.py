# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats



# Visulaisation

from matplotlib.pyplot import xticks

import seaborn as sns

%matplotlib inline
# Accessing our original data 

data = pd.DataFrame(pd.read_csv('../input/motorbike_ambulance_calls.csv')) 
data['date'] = pd.to_datetime(data['date'])

# we convert the seasons, so that we can see the effects better

data['season'] = data['season'].replace({'spring':1, 'summer':2, 'autumn':3, 'winter':4})

# adding a datetime coluumn

data['datetime'] =  pd.to_datetime(pd.to_datetime(data['date']) + pd.to_timedelta(data['hr'], unit='h'))
# we have some missing hours

list_dates = pd.date_range('2011-01-01', '2012-12-31', freq='1H', closed='left')



missing_dates = np.setdiff1d(list_dates.values, data['datetime'].values)
# this will give us some general descriptive statistics that we can use for ploting and  

data.describe()

# basic take aways cnt and windspeed are pretty skewed - max is far away from 75% this may causes issues with normality
sns.distplot(data['cnt']);

# since we know that we have skewness and kurtosis, let's what how high they are

print("Skewness: %f" % data['cnt'].skew())

print("Kurtosis: %f" % data['cnt'].kurt())

# from the values we have high, positive skewness since, our value is > 1.00

# From the chat we can see a high devation from the normal curve, positive kurtosis

# a positive kurtosis typically implies a log or square root data transformation
#correlation matrix of all elements, but not index since that is our primary key

corrmat = data[data.columns[1:]].corr()

f, ax = plt.subplots(figsize=(15, 9))

sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
# we will also want some key aggregations such as sum and means

cols = ['cnt','weathersit', 'temp', 'hum', 'windspeed']

# per month

monthly_mean = pd.DataFrame(data.groupby(['mnth', 'yr'])[cols].agg(['mean']))

monthly_mean.columns = ["mnth".join(x) for x in monthly_mean.columns.ravel()]

data_ana = pd.merge(data, monthly_mean, on='mnth', how='inner')

# per season

season_mean = pd.DataFrame(data.groupby(['season', 'yr'])[cols].agg(['mean']))

season_mean.columns = ["season".join(x) for x in season_mean.columns.ravel()]

data_ana = pd.merge(data_ana, season_mean, on='season', how='inner')

# date time value so we can plot cnts better

data_ana['mnthday'] = data_ana['datetime'].dt.strftime('%m-%d')

data_ana['yrmnth'] = data_ana['datetime'].dt.strftime('%y-%m')

data_ana['yrseason'] = data_ana[['yr', 'season']].apply(lambda x: str(x.yr)+'-'+str(x.season), axis=1) 
# box plot for all the hours in 2011

f, ax = plt.subplots(figsize=(8, 6))

data_2011 = data_ana[data_ana['yr']==0]

fig1 = sns.boxplot(x="hr", y="cnt", data=data_2011)

fig1.axis(ymin=0, ymax=1000);
# box plot for all the hours in 2012

f, ax = plt.subplots(figsize=(8, 6))

data_2012 = data_ana[data_ana['yr']==1]

fig2 = sns.boxplot(x="hr", y="cnt", data=data_2012)

fig2.axis(ymin=0, ymax=1000);
# hourly weekday plot

fig, ax = plt.subplots(figsize=(20,10))

data_weekday = data_ana[data_ana['weekday'].isin([1,2,3,4,5])]

sns.lineplot(x="hr", y="cnt", hue="weekday", estimator=None, data=data_weekday, ax=ax, palette='viridis')
# hourly weekends plot

fig, ax = plt.subplots(figsize=(20,10))

data_weekend = data_ana[(data_ana['weekday']==0) | (data_ana['weekday']==6)]

sns.lineplot(x="hr", y="cnt", hue="weekday", estimator=None, data=data_weekend, ax=ax, palette='viridis')
# holiday plot

f, ax = plt.subplots(figsize=(8, 6))

data_holiday = data_ana[data_ana['holiday']==1]

fig = sns.boxplot(x="hr", y="cnt", data=data_holiday)

fig.axis(ymin=0, ymax=1000);
# average daily plot 

fig, ax = plt.subplots(figsize=(20,10))

# daily per year will be easier to view

# daily data for 2011

sns.lineplot(x='mnthday', y='cnt', data=data_2011, ax=ax)
fig, ax = plt.subplots(figsize=(20,10))

# daily per year will be easier to view

# daily data for 2012

sns.lineplot(x='mnthday', y='cnt', data=data_2012, ax=ax)
# box plot - to compare discreate values of both years

yr_data = pd.concat([data['cnt'], data['yr']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="yr", y="cnt", data=yr_data)

fig.axis(ymin=0, ymax=1000);
# monthly plot

fig, ax = plt.subplots(figsize=(20,10))

sns.lineplot(x='yrmnth', y='cntmnthmean', data=data_ana, ax=ax)
# box plot - for month

mon_data = pd.concat([data['cnt'], data['mnth']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="mnth", y="cnt", data=mon_data)

fig.axis(ymin=0, ymax=1000); 
# features such as 'holiday','weekday','workingday' will have a more spread binary axis were the feature is or isnt that value

# features that may be interesting to view data trends against dates

cols = ['season', 'weathersit', 'temp', 'hum', 'windspeed']



fig, axes = plt.subplots(len(cols), 1, figsize=(18, 20), sharex=True)

for name, ax in zip(cols, axes):

    # will have a similar view to datetime view

    sns.scatterplot(data=data, x='date', y=name, ax=ax)

    ax.set_ylabel(name)

    ax.set_xlim(pd.to_datetime('2010-12-01'), pd.to_datetime('2013-01-31'))
"""

Python function for determing outliars in our data frames.

We are gonna be calling this on a few columns so it makes sens to have a function.

Input:

data - dataframe we wish to analyse

col - column we are doing our analysis of

Output:

all_outliars - list of all the indices of our flagged outliars

"""

def outliar_detection(data, col):

    # IQR detection outliar method

    Q1 = data[col].quantile(0.25)

    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    

    # IQR method outliars

    IQR_out = data[((data[col] < (Q1 - 1.5 * IQR)) |(data[col] > (Q3 + 1.5 * IQR)))]



    # zscore detection outliar methods

    z = np.abs(stats.zscore(data[col]))

    # zscore method outliars index will be in the first part of the array

    threshold = np.where(z > 3)

    

    # looking into indexs that have been flagged by both

    all_outliars = set(IQR_out['index'].tolist()) & set(threshold[0])

    return all_outliars





def flag(curr_index, all_outliars):

    if curr_index in all_outliars:

        return 1

    return 0



# columns we are gonna check for potential outliars

# makes sense to not check anything date related: 'index', 'date', 'season', 'yr', 'mnth', 'hr', 'weekday'

# holiday and workingday is greatly skewwed to 0, so the 1s will flag as outliars

columns = ['weathersit', 'temp', 'hum', 'windspeed', 'cnt']

for col in columns:

    all_outliars = outliar_detection(data, col)

    # if we find any outliars

    if len(all_outliars) > 0:

        data['outliar'+'_'+col] = data.apply(lambda x: flag(x['index'], all_outliars), axis=1)
fig, ax = plt.subplots(figsize=(20,10))

# Create data frame focusing on comparing hum and cnt, also so we can add a color column

hum_data = data[['cnt', 'hum', 'outliar_cnt', 'outliar_hum']]

 

# Add a colour column for outliar values

hum_data['color']= np.where(hum_data['outliar_hum']==1 , "#fc0000", 

         (np.where(hum_data['outliar_cnt']==1, "#fcbb00", "#3498db")) ) 



# ploting the hum outliars

sns.regplot(data=hum_data, x='hum', y='cnt', scatter_kws={'facecolors':hum_data['color']}, ax=ax)
# taking top 3 most correlated features with hum: weathersit, season and mnth to identify the general expected averages under these conditions

replace_hum_outliar = pd.DataFrame(data[(data['weathersit'].isin([2,3])) & (data['season']==1) & (data['mnth']==3) & ((data['index']<1552) | (data['index']>1572))].groupby('hr')['hum'].mean())

# identify the general expected averages, now lets replace all 0.0 for that outliar in hum

for x in replace_hum_outliar.index.values:

    data.loc[(data['date'] == pd.to_datetime('2011-03-10')) & (data['hr'] == x), 'hum'] = replace_hum_outliar['hum'][x]
fig, ax = plt.subplots(figsize=(20,10))

# Create data frame focusing on comparing windspeed and cnt, also so we can add a color column

wind_data = data[['cnt', 'windspeed', 'outliar_cnt', 'outliar_windspeed']]

 

# Add a colour column for outliar values

wind_data['color']= np.where(wind_data['outliar_windspeed']==1 , "#fc0000", 

         (np.where(wind_data['outliar_cnt']==1, "#fcbb00", "#3498db")) ) 



# plot

sns.regplot(data=wind_data, x='windspeed', y='cnt', scatter_kws={'facecolors':wind_data['color']}, ax=ax)
fig, ax = plt.subplots(figsize=(20,10))

# plotting outliars for cnt

sns.scatterplot(data=data, x='date', y='cnt', hue='outliar_cnt', ax=ax)

ax.set_xlim(pd.to_datetime('2010-12-01'), pd.to_datetime('2013-01-31'))
# replacing extreme outliar

data[(data['date'] >= pd.to_datetime('2012-02-01')) &  (data['date'] <= pd.to_datetime('2012-05-01')) & (data['cnt'] >= 850)]
# taking top most correlated features with cnt: temp, season, yr and hr to identify the general expected averages under these conditions

data.loc[(data['index'] == 10623), 'cnt'] = data[(data['temp'] == 0.72) & (data['season']==2) & (data['yr']==1)]['cnt'].mean() 
# Addressing the skewness in 'cnt', identified earlier

# Box-Cox transformation method to cnt, 0.5 is a square root transformation and 0 is log

from sklearn.preprocessing import power_transform

data['cnt'] = power_transform(data['cnt'].values.reshape(-1, 1), method='box-cox')
# Our distplot has change significantly since the first one

sns.distplot(data['cnt'])
# turn into sine and cosine components, to exsist between 0 to 1 still have cyclical characteristics

data['hr_sin'] = np.sin(data['hr']*(2.*np.pi/24))

data['hr_cos'] = np.cos(data['hr']*(2.*np.pi/24))
# categorical given as summer, spring, fall and winter, weathersit and weekday follow similar idea so we will one-hot encoded them

dummy_season = pd.get_dummies(data['season'].replace({1:'spring', 2:'summer', 3:'autumn', 4:'winter'})).astype(int)

dummy_weather = pd.get_dummies(data['weathersit'].replace({1:'clear', 2:'cloudy', 3:'lightrain', 4:'heavyrain'})).astype(int)

dummy_weekday = pd.get_dummies(data['weekday'].replace({0:'sunday', 1:'monday', 2:'tuesday', 3:'wednesday', 4:'thursday', 5:'friday', 6:'saturday'})).astype(int)



# data

data = pd.concat([data, dummy_season, dummy_weather, dummy_weekday], axis=1)
# we looked at outliars so we will drop 'outliar_hum','outliar_windspeed'. Reformatted: 'hr', 'season', 'weathersit', 'weekday'

# We will remove mnth and atemp, strongly correlated to season and atemp. Same idea keeping datetime and dropping date

# drop a column from each encoded column based on which one has the weakest correlation 'autumn', 'cloudy', 'tuesday'



data = data.drop(['outliar_hum','outliar_windspeed', 'atemp', 'mnth', 'date','index', 'hr', 'season', 'weathersit', 'weekday', 'autumn', 'cloudy', 'tuesday'], axis=1)

# change index from int to datetime we created earlier

data = data.set_index('datetime')
import statsmodels.api as sm



x_vars=data.drop(["cnt"], axis=1)

xvar_names=x_vars.columns

for i in range(0,xvar_names.shape[0]):

    y=x_vars[xvar_names[i]]

    x=x_vars[xvar_names.drop(xvar_names[i])]

    rsq=sm.OLS(formula="y~x", endog=y, exog=x).fit().rsquared  

    vif=round(1/(1-rsq),2)

    if int(vif) >= 5:

        print (xvar_names[i], " VIF = " , vif)
# based on VIF being > 5.0 which is the standard cut off

data = data.drop(['workingday','saturday', 'sunday'], axis=1)
cutoff = int(1737 * 0.80)



# Handling the training data

train_data = data[cutoff:]



# Handling the forecasting/prediction variable ~ 1389

test_data = data[-cutoff:]

# cutoff dates for model predication as a validation step

test_st = '2012-11-03 23:00:00'

test_end = '2012-12-31 23:00:00'
from statsmodels.tsa.seasonal import seasonal_decompose

# additive data based on the data we saw in our split linear graph

result = seasonal_decompose(train_data['cnt'], model='additive', freq=365)



fig = plt.figure()  

fig = result.plot()  

fig.set_size_inches(15, 12)
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



"""

Python unit root test function for determing stationarity: hourly and seasonal (shift of 12)

The function will perform ADFuller unit root test, there are many others test as well

We are gonna be calling this a few times after we differentiate so it makes sense to have a function.

Input:

series - dataframe column to be analyzed for stationarity

name - name of column

Output: printing test report

"""



def adfuller_test(series, name=''):

    # run the test

    r = adfuller(series, autolag='AIC')

    

    # hashing out important values to their respective 

    p_value = round(r[1], 4)

    

    # Print Summary

    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)



    if (p_value <= 0.05) and name == 'cnt':

        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis: Series is Stationary.")

    else:

        print(' Autocorrelation       = '+ str(series.autocorr()))

        print(f" => P-Value = {p_value}. Accept the Null Hypothesis: Series is Non-Stationary.")



# ADF Test on each column

for name, column in train_data.iteritems():

    adfuller_test(column, name=column.name)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



"""

In timeseries analysis ACF and PACF play a very important role in identifying

Input: column - name of column we wanna plot

Output: plots of ACF and PACF

"""

def acf_plot(column):        

    fig = plt.figure(figsize=(12,8))

    ax1 = fig.add_subplot(211)

    fig = plot_acf(column, lags=100, ax=ax1) 

    ax2 = fig.add_subplot(212)

    fig = plot_pacf(column, lags=100, ax=ax2)

    ax1.grid()

    ax2.grid()
# original plots

acf_plot(train_data['cnt'])  
# seasonal differentiated series with period 24

acf_plot(train_data['cnt'].diff(periods=24).dropna()) 
# seasonal differentiated series with period 24

acf_plot(train_data['cnt'].diff(periods=24).diff().dropna()) 
sarima_mod = sm.tsa.statespace.SARIMAX(train_data['cnt'], exog=train_data.drop(['cnt'], axis=1), order=(2,1,2), seasonal_order=(1,1,1,24)).fit()

print(sarima_mod.summary())
# plotting ACF and PACF residuals 

acf_plot(sarima_mod.resid) 
from statsmodels.stats.stattools import durbin_watson



# serial correlation implies we still have some unexplained trends in the model

print('cnt durbin_watson: {}'.format(durbin_watson(sarima_mod.resid)))
# checking our 4 assumptions of modeling

sarima_mod.plot_diagnostics(figsize=(15, 12))
from sklearn.metrics import mean_squared_error



# predicting based on pre-defined times above, these are to validate that the model is working

pred = sarima_mod.predict(test_st, test_end)

print('SARIMA model MSE: {}'.format(mean_squared_error(test_data['cnt'],pred)))
# comparing the predicted and test values

pd.DataFrame({'test':test_data['cnt'],'pred':pred}).plot(figsize=(20,10))

plt.show()