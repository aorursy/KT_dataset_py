# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# This competition settings

from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

# Read the data

# Read market and news data

(market_train_df, news_train_df) = env.get_training_data()
# some settings

toy = False

debug = False

testing = True
# remove too old data

import datetime

start = pd.Timestamp(year=2011, month=5, day=5, tz='UTC')

market_train_df = market_train_df.loc[market_train_df['time'] >= start].reset_index(drop=True)

news_train_df = news_train_df.loc[news_train_df['time'] >= start].reset_index(drop=True)



# We will reduce the number of samples for memory reasons

if toy:

    market_train_df = market_train_df.tail(100_000)

    news_train_df = news_train_df.tail(300_000)

else:

    market_train_df = market_train_df.tail(3_000_000)

    news_train_df = news_train_df.tail(6_000_000)



print('market train size = {}, news train size = {}'.format(market_train_df.shape[0], news_train_df.shape[0]))



# map stock code to name

asset_name_df = market_train_df[['assetCode', 'assetName']].drop_duplicates(subset='assetCode').reset_index(drop=True)
# some utilities



def to_float32(df):

    df.loc[:, df.dtypes == np.float64] = df.loc[:, df.dtypes == np.float64].astype(np.float32)

    

def debug_msg(message):

    if debug:

        print(message)

        

import time



# simple timing util, %time creates a new scope makes it hard to use

def print_time(start_time, task_description):

    if debug:

        elapsed = time.time() - start_time

        print('{} time taken: {:.3f} s'.format(task_description, elapsed))

        

# using series append is also faster than using concat, but with the benefit that we keep

# correct type information

def fast_concat(df_list):

    df = pd.DataFrame()

    # assign each column

    for col in df_list[0].columns:

        if pd.api.types.is_categorical_dtype(df_list[0][col]):

            #print('categorical column: {}, using numpy concat'.format(col))

            # categorical type concat is extremely slow

            df[col] = np.concatenate([dfi[col].values for dfi in df_list])

        else:

            #print('column: {}, using series append'.format(col))

            df[col] = df_list[0][col].append([dfi[col] for dfi in df_list[1:]], ignore_index=True)

    return df
market_train_df.loc[market_train_df['assetCode'] == 'AAPL.O'].tail(5)
market_train_df.loc[:, market_train_df.dtypes == np.float64].head()
df = market_train_df.head(100).copy()
news_train_df.tail(5)
def addSplitAdjColumns(market_df):

    # try to automatically detect splits

    split_detect_df = market_df[['time', 'assetCode', 'close', 'returnsClosePrevRaw1']].copy()

    asset_groupby = split_detect_df.groupby('assetCode')

    split_detect_df['prevClose'] = asset_groupby['close'].transform(lambda x: x.shift(1))

    split_detect_df['prevClosePlusReturn'] = split_detect_df['prevClose'] * (1.0 + split_detect_df['returnsClosePrevRaw1'])

    split_detect_df['splitRatio'] = split_detect_df['prevClosePlusReturn'] / split_detect_df['close'] 

    split_detect_df.loc[split_detect_df['assetCode'] == 'AAPL.O'].tail(5)



    # now find points where ratio is > 1.5 or < 0.6, smaller splits guess we cannot correct for

    split_points = split_detect_df[(split_detect_df['splitRatio'] > 1.49) | (split_detect_df['splitRatio'] < 0.67)]

    split_points = split_points[['time', 'assetCode', 'splitRatio']]



    # create a date column

    split_correct_df = market_df[['time', 'assetCode', 'open', 'close']].copy()



    # merge in the split df

    split_correct_df = split_correct_df.merge(split_points, left_on=['assetCode','time'], right_on=['assetCode','time'], how='left')

    split_correct_df.loc[:,'splitRatio'].fillna(1.0, inplace=True)



    # add a adjusted close column for split

    asset_groupby = split_correct_df.groupby('assetCode')

    split_correct_df['splitAdj'] = asset_groupby['splitRatio'].transform(lambda x : x.cumprod())

    split_correct_df['splitAdj'] = asset_groupby['splitAdj'].transform(lambda x : x / x.iloc[-1])

    split_correct_df['adjClose'] = split_correct_df['close'] * split_correct_df['splitAdj']

    split_correct_df['adjOpen'] = split_correct_df['open'] * split_correct_df['splitAdj']



    # put back into market train df

    market_df['adjClose'] = split_correct_df['adjClose']

    market_df['adjOpen'] = split_correct_df['adjOpen']



    del split_correct_df

    del asset_groupby
addSplitAdjColumns(market_train_df)



# plot to make sure it is correct

market_train_df[(market_train_df.assetCode == 'AAPL.O') & (market_train_df.time.dt.year == 2014) & (market_train_df.time.dt.month == 6)]
def checkPropertyFreq(news_df, property_col):

    property_df = news_df[[property_col]].copy()

    # convert to string

    property_df[property_col] = property_df[property_col].str.findall(f"'([\w\./]+)'")

    property_df = pd.DataFrame({

          col:np.repeat(property_df[col].values, property_df[property_col].str.len())

          for col in property_df.columns.drop(property_col)}

        ).assign(**{property_col:np.concatenate(property_df[property_col].values)})[property_df.columns]

    return property_df[property_col].value_counts()



#checkPropertyFreq(news_train_df, 'audiences')

#checkPropertyFreq(news_train_df, 'subjects')
# clustering the stocks into 10 different groups

from sklearn import cluster, datasets



# cluster by the returnsOpenNextMktres10

# some stocks are newly listed so we need to use more recent data

# we sample for 100 times, and use that to generate the clustering

times_sample = np.random.choice(market_train_df.loc[market_train_df['time'].dt.year >= 2011].time.unique(), 500, replace=False)
#cluster_by = 'returnsClosePrevRaw1'

cluster_by = 'returnsOpenPrevRaw1'

asset_return_df = market_train_df.loc[(market_train_df.time.isin(times_sample))][['time', 'assetCode', cluster_by]].copy()

# set index

asset_return_df.set_index(['time', 'assetCode'], inplace=True)



# normalise return of each asset code

# we should normalise return with the mean as well, such that stock that trend vs stock that don't can still be clustered together

# logically we should clip the return such that we don't try to cluster stocks that just so happens to move 30% on the same day as

# another by chance

asset_return_df[cluster_by] = asset_return_df[cluster_by].clip(-0.2, 0.2)

asset_return_df['returnNormalised'] = asset_return_df.groupby('assetCode')[cluster_by].transform(lambda x : (x - x.mean() / x.std()) if x.std() > 1e-10 else x)

asset_return_df



return_by_date_df = asset_return_df.drop([cluster_by], axis=1).unstack(level=0).fillna(0)

return_by_date_df
market_train_df.loc[market_train_df['assetCode'] == 'PNK.O']
len(market_train_df['assetCode'].unique())
return_by_date_df.columns.get_level_values(0)
# here time is one feature, so we need to convert the data into

# assetCode, time1Return, time2Return ... timeNReturn



k_means = cluster.KMeans(n_clusters=100, n_init=500, max_iter=10000, tol=1e-6, verbose=0)

k_means.fit(return_by_date_df['returnNormalised'].values)
labels = k_means.labels_



return_by_date_df.index



asset_cluster = pd.DataFrame()

asset_cluster['assetCode'] = return_by_date_df.index.get_level_values(0)

asset_cluster['cluster'] = labels

# merge in the asset name

asset_cluster = asset_cluster.merge(asset_name_df, left_on='assetCode', right_on='assetCode', how='left')

print('fb cluster = {}'.format(asset_cluster[asset_cluster['assetCode'] == 'FB.O'].iloc[0]['cluster']))



asset_cluster[asset_cluster['cluster'] == 12]



asset_cluster.drop('assetName', axis=1).groupby('cluster').count()



#results = pd.DataFrame(data=labels, columns=['cluster'], index=collapsed.index)

#results

#return_by_date_df.shape

#market_train_df[market_train_df['assetCode'] == 'FB.O']
asset_cluster[asset_cluster['cluster'] == 19]
market_train_df[market_train_df.assetCode == 'MMR.N'].plot(x='time', y=['close'])
# 1. asset name makes more sense, don't expand data like in https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data

# 2. want to make sure news is out by the time stock price is there, so for training data, we shift all news forward by 6 hours for training

# 3.

class GeneratorEma:



    column_factors = {}

    asset_code_factors = {}

    asset_audience_mode = None

    

    non_feature_columns = ['time', 'date', 'assetCode', 'assetName', 'returnsOpenNextMktres10']

    

    @staticmethod

    def drop_non_feature_columns(df):

        return df[df.columns[~df.columns.isin(__class__.non_feature_columns)]]

    

    # start time is used to trim down the data a bit after EMA is

    # calculated, for speed reasons

    @staticmethod

    def generate(market_df, news_df, is_train, start_time=pd.Timestamp(year=2000, month=1, day=1, tz='UTC')):

        

        market_df = market_df.copy()

        news_df = news_df.copy()

        

        to_float32(market_df)

        to_float32(news_df)

        

        t1 = time.time()

        

        # merge in the cluster code

        market_df = market_df.merge(asset_cluster.drop(['assetName'], axis=1), left_on='assetCode', right_on='assetCode', how='left')

        

        # factorize asset code

        col = 'assetCodeFactorized'

        if is_train:

            market_df[col], uniques = pd.factorize(market_df.assetCode)

            # reserve 0 for unknown

            market_df[col] += 1

            __class__.asset_code_factors = { uniques[i]:(i + 1) for i in range(len(uniques)) }

        else:

            market_df[col] = market_df.assetCode.map(lambda a: __class__.asset_code_factors.get(a, 0))

        print_time(t1, 'factorize asset code')

        

        # preprocess

        t1 = time.time()

        market_df = __class__.process_mkt(market_df, start_time)

        print_time(t1, 'process mkt')

        

        t1 = time.time()

        news_df = __class__.process_news(news_df, is_train, start_time)

        print_time(t1, 'process news')



        # Join market and news

        t1 = time.time()

        train_df = market_df.join(news_df, on=['date', 'assetName'])

        print_time(t1, 'join market and news')

            

        del news_df

        del market_df



        # merge the most freq audience in

        # asset_df = asset_df.merge(_class__.asset_audience_mode, left_on='assetName', right_on='assetName', how='left')

                                

        # convert float64 columns to float32

        train_df.loc[:, train_df.dtypes == np.float64] = train_df.loc[:, train_df.dtypes == np.float64].astype(np.float32)

        

        # fill na, not sure why we need to exclude any category type columns

        train_df.loc[:,train_df.dtypes != 'category'] = train_df.loc[:,train_df.dtypes != 'category'].fillna(0)

        

        debug_msg('done')

        

        return train_df

    

    # we want to transform the news to index by date and asset

    @staticmethod

    def process_news(news_df, is_train, start_time):

        

        # following copied from

        # https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data

        news_cols_agg = {

            'urgency': ['count'],

            'takeSequence': ['max'],

            'bodySize': ['mean', 'sum', 'std'],

            'wordCount': ['mean'],

            'sentenceCount': ['mean'],

            'companyCount': ['mean'],

            'marketCommentary': ['mean'],

            'relevance': ['mean'],

            'sentimentNegative': ['sum', 'mean', 'std'],

            'sentimentNeutral': ['sum', 'mean', 'std'],

            'sentimentPositive': ['sum', 'mean', 'std'],

            'sentimentWordCount': ['sum', 'mean', 'std'],

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



        # Maybe drop those 3 days / 5 days count as they are highly correlated to the 7d



        # drop some columns

        news_df.drop(['assetCodes', 'sourceTimestamp', 'firstCreated', 'headline', 'audiences', 'subjects', 'sourceId'], axis=1, inplace=True)

        news_df = news_df.loc[news_df.time >= start_time].reset_index(drop=True)

        

        # factorize some columns

        # https://stackoverflow.com/questions/46761978/factorize-values-across-dataframe-columns-with-consistent-mappings

        # also maybe one hot is better, see https://stackoverflow.com/questions/34265102/xgboost-categorical-variables-dummification-vs-encoding

        # Factorize categorical columns

        for col in ['headlineTag', 'provider']:

            if is_train:

                news_df[col], uniques = pd.factorize(news_df[col])

                __class__.column_factors[col] = { uniques[i]:i for i in range(len(uniques)) }

            else:

                factors_dict = __class__.column_factors[col]

                news_df[col] = news_df[col].map(lambda a: factors_dict.get(a, 0))



        # convert these into lists of items

        # news_df['subjects'] = news_df['subjects'].str.findall(f"'([\w\./]+)'")



        # create a date column

        news_df['date'] = news_df.time.dt.date



        # drop the 'time' column

        news_df.drop(['time'], axis=1, inplace=True)



        # aggregate on date

        cols = list(news_cols_agg.keys())

        # Convert to float32 to save memory, also for bool type to work properly

        news_df[cols] = news_df[cols].apply(np.float32)

        news_df_agg = news_df.groupby(['date', 'assetName']).agg(news_cols_agg)

        

        # Flat columns

        news_df_agg.columns = ['_'.join(col).strip() for col in news_df_agg.columns.values]



        return news_df_agg



    # mkt data is simpler

    # but we want to convert volume to value

    @staticmethod

    def process_mkt(market_train, start_time):

        

        if 'returnsOpenNextMktres10' in market_train.columns:

            # remove outliers

            market_train = market_train.loc[(market_train['returnsOpenNextMktres10'] > -0.5) & (market_train['returnsOpenNextMktres10'] < 0.5)].reset_index(drop=True)

        

        if 'universe' in market_train.columns:

            # remove all points with universe != 1

            # they are not used for scoring, if we use time sequence data maybe will need to include them

            market_train = market_train.loc[market_train['universe'] > 0.0]

            market_train.drop(['universe'], axis=1, inplace=True)

        

        # create a value column, which is just approx value (cause we cannot be sure of price)

        # opening price is usually not as good as closing price as a reflection of average trading price

        market_train['value'] = market_train['volume'] * market_train['close']

        

        # create a open vs close column

        market_train['dayChange'] = (market_train['close'] - market_train['open']) / market_train['open']

        

        # add some EMAs

        asset_groupby = market_train.groupby('assetCode')

        ema10Days = asset_groupby['adjClose'].transform(lambda x : x.ewm(span=10).mean())

        ema50Days = asset_groupby['adjClose'].transform(lambda x : x.ewm(span=50).mean())

        market_train['closeOverEma10days'] = (market_train['adjClose'] - ema10Days) / ema10Days

        market_train['closeOverEma50days'] = (market_train['adjClose'] - ema50Days) / ema50Days

        market_train['valueEma50days'] = asset_groupby['value'].transform(lambda x : x.ewm(span=50).mean())

        market_train['valueOverEma'] = (market_train.value - market_train.valueEma50days) / market_train.valueEma50days

        

        # after EMA is calculated we can filter out by time

        market_train = market_train.loc[market_train.time >= start_time].reset_index(drop=True)

        

        # drop open / close / volume / value

        market_train.drop(['open', 'close', 'adjOpen', 'adjClose', 'volume', 'value'], axis=1, inplace=True)



        # create a date column

        market_train['date'] = market_train.time.dt.date



        return market_train

    

    # extract some property for asset codes

    # the idea is try to seperate them into sectors

    @staticmethod

    def extractAssetProperty(news_df, property_col):

        property_df = news_df[['assetName', property_col]].copy()

        # convert to string

        property_df[property_col] = property_df[property_col].str.findall(f"'([\w\./]+)'")

        property_df = pd.DataFrame({

              col:np.repeat(property_df[col].values, property_df[property_col].str.len())

              for col in property_df.columns.drop(property_col)}

            ).assign(**{property_col:np.concatenate(property_df[property_col].values)})[property_df.columns]



        property_df = property_df.groupby('assetName').agg(lambda x:x.mode().iloc[0]).dropna()

        

        # factorize the column

        property_df[property_col], uniques = pd.factorize(property_df[property_col])

        property_df[property_col] += 1

        

        return property_df

    

    # NOT used

    # this code is interesting but we don't want to do it

    @staticmethod

    def expandAssetCodes(news_df):

        # code copied from 

        # https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list/48532692#48532692

        # which expands all asset codes list into rows

        code_col = 'assetCodes'

        news_df = pd.DataFrame({

              col:np.repeat(news_df[col].values, news_df[code_col].str.len())

              for col in news_df.columns.drop(code_col)}

            ).assign(**{code_col:np.concatenate(news_df[code_col].values)})[news_df.columns]



        return news_df

    

    @staticmethod

    def unit_test():

        pass
# plot some data



market_df = market_train_df.copy()

market_df['ewm10'] = market_df.groupby('assetCode')['adjClose'].transform(lambda x : x.ewm(span=10).mean())

market_df['ewm50'] = market_df.groupby('assetCode')['adjClose'].transform(lambda x : x.ewm(span=50).mean())
assetCode = 'AAPL.O'

thisAssetMark_df = market_df[market_df['assetCode']==assetCode].sort_values(by='time',ascending=True) 



f, axs = plt.subplots(2,1, sharex=True, figsize=(12,8))



# Price vs time

thisAssetMark_df.plot(ax=axs[0], x='time', y=['adjClose','ewm50','ewm10'])

thisAssetMark_df.plot(ax=axs[1], x='time', y=['returnsOpenNextMktres10'])

f.suptitle('Close price and EMA vs time')

plt.show()
def train_test_split(x, bucket_days, test_size):

    """

    we cannot use normal train test split as data is time related

    we need to split by selecting chunks of time



    :param: DataFrame x, y:      

    :param: int bucket_days:  duration of bucket

    :param: test_size: 

    :return: tuple:              the (x_train, x_test, y_train, y_test) tuple



    """



    import random

    import datetime



    time_min = x.date.min()

    time_max = x.date.max()



    #print('time min: {} max: {}'.format(time_min, time_max))



    train_or_test = []



    # split the time into duration, and toss a dice to see which way it should go

    t = time_min

    while(t <= time_max):

        is_test = random.random() < test_size

        train_or_test.append((t, is_test))

        #print('time : {} test?: {}'.format(t, is_test))

        t = t + datetime.timedelta(days = bucket_days)



    def is_test(input_t):

        # pretty bad linear search, hope it is not too slow

        for t in train_or_test:

            if input_t <= t[0]:

                return t[1]

        return False



    is_test = x.date.map(lambda t : is_test(t))



    x_train = x[is_test == False].copy()

    x_test = x[is_test].copy()



    return (x_train, x_test, is_test)

import xgboost as xgb

import sklearn



def train_model(train_df, test_df, colsample_bytree = 0.8, learning_rate = 0.1, max_depth = 15,

                alpha = 10, n_estimators = 100):



    # only use a small subset

    #train_df = train_df.sample(frac=0.2).reset_index(drop=True)

    #test_df = test_df.sample(frac=0.2).reset_index(drop=True)



    y_train = train_df['returnsOpenNextMktres10']

    y_test = test_df['returnsOpenNextMktres10']



    # drop column

    train_df = GeneratorEma.drop_non_feature_columns(train_df)

    test_df = GeneratorEma.drop_non_feature_columns(test_df)



    eval_set = [(test_df, y_test)]

    xg_reg = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree = colsample_bytree, learning_rate = learning_rate, max_depth = max_depth, alpha = alpha, n_estimators = n_estimators)

    xg_reg.fit(train_df, y_train, eval_metric='rmse', eval_set=eval_set, verbose=True)

    preds = xg_reg.predict(test_df)

    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, preds))

    return xg_reg

train_df = GeneratorEma.generate(market_train_df, news_train_df, is_train=True)

train_df.tail(5)



# split into train / test

train_df, test_df, is_test = train_test_split(train_df, bucket_days=100, test_size=0.2)



xg_reg = train_model(train_df, test_df, colsample_bytree = 0.8, learning_rate = 0.1,

                     max_depth = 10, alpha = 1, n_estimators = 70)

print('finish training model!')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

xgb.plot_importance(xg_reg, ax=ax)
class Predictor:



    def __init__(self, market_obs_df, news_obs_df):

        # these are the dfs that we keep

        self.market_obs_df = market_obs_df

        self.news_obs_df = news_obs_df

        self.drop_old_data

        

    # drop all data older than 50 days, that is what we need

    def drop_old_data(self, current_time):

        cutoff_time = current_time - pd.Timedelta(days=50)

        self.market_obs_df = self.market_obs_df.loc[self.market_obs_df['time'] >= cutoff_time]

        self.news_obs_df = self.news_obs_df.loc[self.news_obs_df['time'] >= cutoff_time]



    # predictions template looks like:

    #    assetCode   confidenceValue

    #    A.N         0.0

    #    AA.N        0.0

    #    AAL.O       0.0

    #    AAN.N       0.0

    # we need to fill it in then call env.predict(predictions_template_df)

    #

    def make_predictions_day(self, xg_reg, predictions_template_df, market_day_df, news_day_df):



        # sanity check to make sure the time are sane

        time_min = market_day_df.time.min()

        time_max = market_day_df.time.max()

        

        assert(time_min == time_max)

                

        # essentially run the batch mode

        pred_df = self.make_predictions_impl(xg_reg, market_day_df, news_day_df)

        

        # join to get only the new rows

        

        #merge into the predictions df

        pred_df = predictions_template_df.drop(['confidenceValue'], axis=1).merge(pred_df, left_on='assetCode', right_on='assetCode', how='left')

        __class__.add_confidence_value(pred_df)

        #print(predictions_df.shape[0])

        #print(predictions_template_df.shape[0])

        

        # make sure row counts are correct

        assert pred_df.shape[0] == predictions_template_df.shape[0]

        return pred_df



    # batch mode allows making prediction on many days together, which is more useful for fast iteration

    # use case is save down the prediction days from the competition env and just run through it

    # once and score

    # return a df with

    #  date assetCode confidenceValue

    def make_predictions_batch(self, xg_reg, market_df, news_df):

        return self.make_predictions_impl(xg_reg, market_df, news_df)

    

    def make_predictions_impl(self, xg_reg, market_df, news_df):

        # sanity check to make sure the time are sane

        time_min = market_df.time.min()

        

        # append to the running dfs

        self.market_obs_df = fast_concat([self.market_obs_df[market_df.columns], market_df])

        self.news_obs_df = fast_concat([self.news_obs_df[news_df.columns], news_df])



        self.drop_old_data(time_min)

        

        # essentially run the batch mode

        pred_df = GeneratorEma.generate(self.market_obs_df, self.news_obs_df, is_train=False, start_time=time_min)

        pred_df['prediction'] = xg_reg.predict(GeneratorEma.drop_non_feature_columns(pred_df))

        

        __class__.add_confidence_value(pred_df)

        return pred_df

    

    @staticmethod

    def add_confidence_value(pred_df):

        # normalize predictions into confidence

        # anything outside 3x stddev is max confidence

        pred_df['confidenceValue'] = pred_df.groupby('time')['prediction'].transform(lambda x : (x - x.mean()) / x.std())

        pred_df['confidenceValue'] = pred_df['confidenceValue'].clip(lower=-1.0, upper=1.0)

        



# using the description given by the competition to score

# see https://www.kaggle.com/c/two-sigma-financial-news#evaluation

# predictions df must have following columns:

# date assetCode confidenceValue

def score_predictions(predictions_df, market_df):

    # just get the columns we needed

    return_res_df = market_df[['time', 'assetCode', 'returnsOpenPrevMktres10']].copy()



    # Shift -11 days gives us returnsOpenNextMktres10

    return_res_df['returnsOpenNextMktres10'] = return_res_df.groupby(['assetCode'])['returnsOpenPrevMktres10'].shift(-11)



    # merge the results column into the predictions

    predictions_df = predictions_df.merge(return_res_df, left_on=['assetCode', 'time'], right_on=['assetCode', 'time'], how='left')

    predictions_df.dropna(inplace=True)



    #return predictions_df

    

    # for each day, take confidence 

    daily_score_x = predictions_df.groupby('time').apply(lambda x : (x['confidenceValue'] * x['returnsOpenNextMktres10']).sum())

    #return daily_score_x



    score = daily_score_x.mean() / daily_score_x.std()

    return score
days = []

predictor = Predictor(market_train_df, news_train_df)

pred_dfs = []



# copy it so we can reuse it

for (market_day_df, news_day_df, predictions_template_df) in tqdm(env.get_prediction_days()):

    days.append((market_day_df, news_day_df, predictions_template_df))

    if testing:

        env.predict(predictions_template_df)

    else:

        pred_df = predictor.make_predictions_day(xg_reg, predictions_template_df, market_day_df, news_day_df)

        pred_dfs.append(pred_df)

        env.predict(pred_df[['assetCode','confidenceValue']])

(market_day_df, news_day_df, predictions_template_df) = days[0]

market_day_df[market_day_df.assetCode == 'AAPL.O'].tail(5)
market_obs_df = fast_concat([day[0] for day in days])

news_obs_df = fast_concat([day[1] for day in days])

addSplitAdjColumns(market_obs_df)



# try scoring what we just did

if not testing:

    pred_df = fast_concat(pred_dfs)

    score = score_predictions(fast_concat(pred_dfs), market_obs_df)

    print('my calculated score = {:.3f}'.format(score))
news_obs_df.tail(3)

market_obs_df[(market_obs_df.assetCode == 'FB.O') & (market_obs_df.time.dt.year > 2018)].plot(x='time', y=['close'])
last_day_mkt_df = days[-1][0]

last_day_mkt_df[last_day_mkt_df.assetCode == 'AAPL.O']
predictor = Predictor(market_train_df, news_train_df)

predictions_df = predictor.make_predictions_batch(xg_reg, market_obs_df, news_obs_df)



# those market ob for dates that have not occurred yet are just dummy data, remove them

end = pd.Timestamp(year=2019, month=2, day=1, tz='UTC')

score = score_predictions(predictions_df.loc[predictions_df['time'] < end].reset_index(drop=True), market_obs_df)



print('Score = {:.3f}'.format(score))
pred_df = predictions_df

pred_df['confidenceValue'] = (pred_df['prediction'] - pred_df['prediction'].mean()) / pred_df['prediction'].std()

pred_df['confidenceValue'] = pred_df['confidenceValue'].clip(lower=-1.0, upper=1.0)

score_predictions(pred_df, market_obs_df)
if testing:

    # try doing day by day pred again

    debug = False

    predictor = Predictor(market_train_df, news_train_df)

    #(market_day_df, news_day_df, predictions_template_df) = days[0]

    #pred_df = predictor.make_predictions(xg_reg, predictions_template_df, market_day_df, news_day_df)



    pred_dfs = []

    for (market_day_df, news_day_df, predictions_template_df) in tqdm(days[0:2]):

        pred_dfs.append(predictor.make_predictions_day(xg_reg, predictions_template_df, market_day_df, news_day_df))



    # verify the pred

    pred_df = fast_concat(pred_dfs)

    score = score_predictions(fast_concat(pred_dfs), market_obs_df)

    print('day by day pred score = {:.3f}'.format(score))
predictions_df.head()
pred_df.head()
# write submission file

print('Done!')

env.write_submission_file()