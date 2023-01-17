# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import chain

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()

#del env
# increase the number of displayed columns in dataframes

pd.set_option('display.max_columns', 100)

# Check if the run is for testing the code or final results.

# if checking the code then reduce the size of datasets for memory saving.

test_code = True
if test_code:

    market_train_df = market_train_df.head(100000)

    news_train_df = news_train_df.head(300000)
market_train_df.head()
# Categorizing and encoding the object type columns

def categorize(df):

    obj_df = df.select_dtypes(exclude=[np.int8, np.int16, np.int32, np.float32]).drop(['assetCode', 'time'], axis=1).astype('category')

    for col in obj_df.columns:

        df[col] = obj_df[col].cat.codes

    return df
# Scaler for numerical numbers

from sklearn.preprocessing import MinMaxScaler

std = MinMaxScaler()

def scaler(df):

    arr = df.select_dtypes(exclude=[object])

    temp = pd.DataFrame(std.fit_transform(arr), columns=arr.columns)

    temp = temp.astype(np.float16)

    df[arr.columns] = temp

    return df
## Functions for preparing the data. The code should run in 1-2 minutes for the full dataset.



def prepare_news(news_df):   

    agg_dict = {

        'sourceId' : ['count'],

        'urgency': ['min'],

        'takeSequence': ['max'],

        'provider' : ['count'],

        'bodySize': ['mean'],

        'wordCount': ['mean'],

        'sentenceCount': ['mean'],

        'companyCount': ['mean'],

        'relevance': ['mean'],

        'sentimentClass' : ['mean'],

        'sentimentNegative': ['mean'],

        'sentimentNeutral': ['mean'],

        'sentimentPositive': ['mean'],

        'sentimentWordCount': ['mean'],

        'noveltyCount12H': ['mean'],

        'noveltyCount24H': ['mean'],

        'noveltyCount3D': ['mean'],

        'noveltyCount5D': ['mean'],

        'noveltyCount7D': ['mean'],

        'volumeCounts12H': ['mean'],

        'volumeCounts24H': ['mean'],

        'volumeCounts3D': ['mean'],

        'volumeCounts5D': ['mean'],

        'volumeCounts7D': ['mean']

    }



    print('Preparing "news_train_df".')

    #code_list = news_df['assetCodes'].apply(lambda x: x.strip('{}').replace(',', ' ').replace('\'','').split())

    code_list = news_df['assetCodes'].str.findall(f"'([\w\./]+)'")

    code_list_ext = list(chain(*code_list))

    code_name_ext = news_df['assetName'].repeat(code_list.apply(len))

    code_time_ext = news_df['time'].repeat(code_list.apply(len))

    code_list_ext = pd.Series(code_list_ext, name='assetCode')

    code_df_ext = pd.DataFrame(code_list_ext).reset_index(drop=True)

    name_df_ext = pd.DataFrame(code_name_ext).reset_index(drop=True)

    time_df_ext = pd.DataFrame(code_time_ext).reset_index(drop=True)

    code_df_ext = code_df_ext.join(name_df_ext).join(time_df_ext)

    

    del time_df_ext, name_df_ext, code_list, code_name_ext, code_time_ext, code_list_ext 

    gc.collect()



    print('Extending the "news_train_df".')

    

    news_df = news_df.drop('assetCodes', axis=1)  

    news_df = news_df.merge(code_df_ext, on=['assetName', 'time'], how='inner')

    

    del code_df_ext

    print('"news_train_df" extended.')



    news_df['time'] = news_df['time'].dt.date



    print('Converted date_time to date only.')

    

    list_to_drop = ['headline', 'subjects', 'audiences', 'assetName', 'headlineTag', 'marketCommentary', 

                    'firstMentionSentence', 'firstCreated', 'sourceTimestamp']

    for col in list_to_drop:

        if col in news_df.columns:

            news_df = news_df.drop(col, axis=1)

    print('Cleaned the "news_train_df".')

    

    news_df = news_df.groupby(['assetCode', 'time']).agg(agg_dict).reset_index()

    news_df.columns = news_df.columns.get_level_values(0)



    print('Combined the "news_train_df" different hours of a day into one day.')

    print('Finished preparing "news_train_df".')

    return news_df



def prepare_market(market_df):

    print('Preparing "market_train_df".')

    market_df = market_df.drop('assetName', axis=1)

    

    print('Cleaning ...')

    market_df = market_df.dropna(axis = 1, how='all')

    market_df = market_df[market_df['universe'] == 1]



    print('Cleaning "market_train_df" finished.')

    market_df['time'] =  market_df['time'].dt.date

    

    print('Finished preparing  market_train_df')

    return market_df



def create_full_dataframe(news_df, market_df):

    

    news = prepare_news(news_df)

    market = prepare_market(market_df)

    

    m = pd.DataFrame(market['assetCode'].unique()).rename(columns={0 : 'assetCode'})

    n = pd.DataFrame(news['assetCode'].unique()).rename(columns={0 : 'assetCode'})

    shared_assets_pd = pd.DataFrame(m.merge(n, on='assetCode', how='inner')['assetCode'])

    

    print('Created a dataframe of shared assets between the market and news dataframes')

    

    news = news.merge(shared_assets_pd.reset_index(), on='assetCode', how='inner')

    

    market = market.merge(shared_assets_pd.reset_index(), on='assetCode', how='inner')

    

    print('Created a news and market dataframe only with shared assets')

    

    full_df = news.merge(market, on=['assetCode', 'time'], how='outer')

    

    del market, news, shared_assets_pd, m, n

    gc.collect()

    

    full_df = full_df.sort_values(by=['assetCode', 'time']).reset_index(drop=True)

    

    print('Merged the news and market datframes into one full dataframe with shared assets, retaining the times from both')

    print('Finished preparing data.')

    return full_df

assets_df = create_full_dataframe(news_train_df,market_train_df)
plt.figure(1, figsize=(15, 8))

plt.xticks(rotation='75')

plt.xlabel('Time')

plt.ylabel('Price')

plt.plot(assets_df[assets_df['assetCode'] == 'A.N']['time'], 

         scaler(pd.DataFrame(assets_df[assets_df['assetCode'] == 'A.N']['close'])))
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

class model(object):

    def __init__(self, asset_df = None,

                 status='close', 

                 roll_window_size = 10,

                 ewm_half_life = 1,

                 decomp_model = 'multiplicative',

                 trend_deg = 3,

                 l=0,

                 u=None):

        

        self.asset_df = asset_df

        self.status = status

        self.roll_window_size = roll_window_size

        self.ewm_half_life = ewm_half_life

        self.decomp_model = decomp_model

        self.trend_deg = trend_deg

        self.coef, self.t_coef, self.price = self.get_rolling_coef(l=l, u=u)

        

    ## Performs rolling window averaging on the time series and decomposes the data into its trend, seasonality and residuals.

    ## Residuals are the data we want to explore and predict abd then we add back the trend and seasonality

        

    def get_rolling_price(self):

        price = self.asset_df[['time', self.status]].dropna()

        price[self.status] = price[self.status].map(lambda x: np.log(x))

        price[self.status] = std.fit_transform(np.reshape(np.array(price[self.status]), (-1, 1)))

        #price[self.status] = price[self.status] - price[self.status].shift()

        price[self.status]  = price[self.status].ewm(halflife=self.ewm_half_life).mean().rolling(window=self.roll_window_size, win_type='blackman').mean()

        price['time'] = pd.to_datetime(price['time'])

        price = price.dropna().set_index('time').asfreq('d').fillna(method='ffill')

        decomposition = seasonal_decompose(price, model=self.decomp_model)

        price['resid'] = decomposition.resid

        price['trend'] = decomposition.trend

        price['seasonal'] = decomposition.seasonal

        price = price.dropna()

        return price.reset_index()

    

    ## Adfuller test for stationary time series, Time series should be stationary to ensure that the series distribution parameters are consistent.

    def test_stationary(self, time_series):

        print('Results of Dickey-Fuller test:')

        dftest = adfuller(time_series, autolag='AIC')

        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

        for key,value in dftest[4].items():

            dfoutput['Critical Value (%s)'%key] = value

        print(dfoutput)

        

    ## Rolling window regression for the residual and the trend. This is used to predict the future trend and the asset price.

    def get_rolling_coef(self, l, u):

        time_series = self.get_rolling_price()

        time_series = time_series.reset_index()[l:u]

        self.test_stationary(time_series.resid)

        p_splits = [time_series['resid'].shift(x).values[::-1][:self.roll_window_size] for x in range(len(time_series['resid']))[::-1]]

        t_splits = [time_series['index'].shift(x).values[::-1][:self.roll_window_size] for x in range(len(time_series['index']))[::-1]]

        size = p_splits[0].shape[0]

        rolling_coef = []

        for c in range(len(p_splits)):

            if c >= size - 1:

                coef = np.polyfit(t_splits[c], p_splits[c], 1)

                rolling_coef.append(coef[0])

        price_coef = np.polyfit(np.arange(0, len(rolling_coef)), rolling_coef, 1)

        trend_coef = np.polyfit(np.arange(0, len(time_series.trend)), time_series.trend, self.trend_deg)

        return price_coef, trend_coef, time_series

    

    ## Estimates the price at each time point. (for trainig and prediction) 

    ## Seasonality is periodical so we already know the value for it at each time point and there is no need for prediction.

    def estimate_price(self):

        coef, t_coef, price = self.coef, self.t_coef, self.price

        season = price.seasonal.where(price.seasonal.duplicated() == False).dropna()

        fit = np.poly1d(coef)

        split_coef = pd.Series(fit(price['index']))

        eprice = (split_coef * price['index']).rename('estimated_price')

        fit = np.poly1d(t_coef)

        trend = pd.Series(fit(price['index']))

        new_index = pd.Series(price.index).apply(lambda x: x % len(season))

        time = price.time

        eprice = eprice + new_index.apply(lambda x: season[x])

        eprice = eprice + trend - 1

        eprice = pd.concat((time, eprice.rename('estimated_price')), axis=1)

        return eprice

    

    def predict(self, time_points=None):

        coef, t_coef, price = self.coef, self.t_coef, self.price       

        season = price.seasonal.where(price.seasonal.duplicated() == False).dropna()

        fit = np.poly1d(coef)

        split_coef = pd.Series(fit(time_points))

        eprice = (split_coef * time_points).rename('estimated_price')

        fit = np.poly1d(t_coef)

        trend = pd.Series(fit(time_points))

        new_index = pd.Series(time_points).apply(lambda x: x % len(season))

        time = pd.Series(pd.date_range(price.time[len(price.time) - 1], periods=len(time_points))).rename('time')

        eprice = eprice + new_index.apply(lambda x: season[x])

        eprice = eprice + trend - 1

        eprice = pd.concat((time, eprice.rename('estimated_price')), axis=1)  

        return eprice
status = 'close'

roll_size = 30

halflife = 1

deg = 3



mo = model(assets_df[assets_df['assetCode'] == 'A.N'], 

           status=status,

          roll_window_size=roll_size,

          ewm_half_life=halflife,

          trend_deg=deg)



price = mo.get_rolling_price()

eprice = mo.estimate_price()

target = pd.Series(np.arange(len(eprice) - 1, len(eprice) + 15), name='time_points')

target_price = mo.predict(time_points=target)



plt.figure(2, figsize=(8, 4))

plt.xticks(rotation='75')

plt.xlabel('Time')

plt.ylabel('Price')

plt.plot(price.time, price.close, label='Real data')

plt.plot(eprice.time, eprice.estimated_price, label='Estimated price for the training set')

plt.plot(target_price.time, target_price.estimated_price, label='Predicted price for the future')

plt.legend()

#new_index = pd.Series(price.index).apply(lambda x: x % np.floor(len(price.seasonal)/len(season)).astype(np.int8))

#eprice + new_index.apply(lambda x: x + price.seasonal[x])

#print(price.seasonal + eprice)

#print(new_index.apply(lambda x: x + price.seasonal[x]) + eprice)
