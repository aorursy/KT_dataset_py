# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from itertools import chain

import seaborn as sns

from joblib import Parallel, delayed

import multiprocessing

import time



import lightgbm as lgb



# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!

env = twosigmanews.make_env()
(m_train_df, n_train_df) = env.get_training_data()
# returnsClosePrevMktres1, returnsOpenPrevMktres1, returnsClosePrevMktres10, returnsOpenPrevMktres10

global truncList



truncList = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1',

             'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',

             'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',

             'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']



# Drop years with high fluctuation

def transer_MarketDate(df):

    df['date'] = df['time'].dt.strftime('%Y%m%d').astype(int)

    df = df[df['date'] > 20101231].reset_index()

    df.drop(['index'], axis=1, inplace=True)

    return df



# Fix wrong (open | close) price

def refix_diffWrongPrice(df, maxLine, minLine):

    assert isinstance(df, pd.DataFrame)

    assert len(df) > 0

    assert maxLine > minLine

    df['c_o_divff'] = df['close'] / df['open']

    # Fix wrong closePrice

    max_divff = df['c_o_divff'].sort_values(ascending=False)

    max_list = max_divff[max_divff >= maxLine].index.astype(list)

    print('Fixing wrong closePrice, indexes: ', max_list)

    for m in max_list:

        stock_wrongInfo = df[df.index == m]

        # get column value: date

        get_nextDayInfo = df[(df['assetCode'] == stock_wrongInfo['assetCode'].values[0]) & 

                             (df['date'] == (stock_wrongInfo['date'].values[0])+1)]

        # get column value: close

        trueCloseNext1 = get_nextDayInfo['close'].values[0]

        # get column value: returnsClosePrevRaw1

        trueClosePrevRaw1 = get_nextDayInfo['returnsClosePrevRaw1'].values[0]

        # calculate result

        df.loc[m, 'close'] = np.array([trueCloseNext1 / (trueClosePrevRaw1 + 1)])

    print('Fixing wrong closePrice complete !')

    # Fix wrong openPrice

    min_divff = df['c_o_divff'].sort_values(ascending=True)

    min_list = min_divff[min_divff <= minLine].index.astype(list)

    print('Fixing wrong openPrice, indexes: ', min_list)

    for m in min_list:

        stock_wrongInfo = df[df.index == m]

        # get column value: date

        get_nextDayInfo = df[(df['assetCode'] == stock_wrongInfo['assetCode'].values[0]) & 

                             (df['date'] == (stock_wrongInfo['date'].values[0])+1)]

        # get column value: open

        trueOpenNext1 = get_nextDayInfo['open'].values[0]

        # get column value: returnsOpenPrevRaw1

        trueOpenPrevRaw1 = get_nextDayInfo['returnsOpenPrevRaw1'].values[0]

        # calculate result

        df.loc[m, 'open'] = np.array([trueOpenNext1 / (trueOpenPrevRaw1 + 1)])

    print('Fixing wrong openPrice complete !')

    df.drop(['c_o_divff'], axis=1, inplace=True)

    return df
class TruncOutliter:

    def __init__(self, df, trans_column, max_value, min_value, resultType):

        self.df = df

        self.trans_column = trans_column

        self.max_value = max_value

        self.min_value = min_value

        self.resultType = resultType

        

    def _trunc_outliter(self,s):

        if s < self.min_value:

            return self.min_value

        elif s > self.max_value:

            return self.max_value

        else:

            return s

        

    def trunc(self):

        start = time.time()

        pool = multiprocessing.Pool(processes=3)

        result = pool.map(self._trunc_outliter, self.df.loc[:, self.trans_column])

        print('truncating -- time cost : {0:.2f}'.format(time.time() - start))

        df_trunc = pd.DataFrame(data=result, columns=[self.trans_column], dtype=self.resultType)

#         final_df = self.df.drop([self.trans_column], axis=1)

#         final_df = pd.concat([final_df, df_trunc], axis=1)

        pool.close()

        #del result, df_trunc

        print('column -- {} -- completed !!'.format(self.trans_column))

        return df_trunc



    

def trunc_ouliter(columns, df, coef=1.5):

    if coef < 1.5:

        coef = 1.5

    assert isinstance(columns, list)

    assert len(df.columns) == len(set(columns).union(set(df.columns)))

    for col in columns:

        if df[col].isnull().sum() > 0:

            fill_na_50 = df[df[col].notnull()][col].describe()['50%']

            df[col].fillna(fill_na_50, inplace=True)

        col_q1 = df[col].describe()['25%']

        col_q3 = df[col].describe()['75%']

        _min = col_q1 - (col_q3 - col_q1) * coef

        _max = col_q3 + (col_q3 - col_q1) * coef

        truncObj = TruncOutliter(df, col, _max, _min, 'float32')

        _df = truncObj.trunc()

        df.drop([col], axis=1, inplace=True)

        df = pd.concat([df, _df], axis=1)

    return df
def extract_MarketFeatures(market_train_df):

    start_time = time.time()

    print('extract_MarketFeatures -- start...')

    # Extract: divdiff_volume & divdiff_volume_labeled

    market_train_df['avg_volume'] = market_train_df.groupby(by=['date'])['volume'].transform('mean')

    market_train_df['divdiff_volume'] = market_train_df['volume'] / market_train_df['avg_volume']

    market_train_df['divdiff_volume_labeled'] = (market_train_df['divdiff_volume'] >= 1).astype(int)

    # Extract: price_divff & price_divff_labeled

    market_train_df['price_divff'] = market_train_df['close'] / market_train_df['open']

    market_train_df['price_divff_labeled'] = (market_train_df['price_divff'] >= 1).astype(int)

    # Extract: vol_price_div & vol_price_div_labeled

    market_train_df['vol_price_div'] = market_train_df['price_divff'] * market_train_df['divdiff_volume']

    market_train_df['vol_price_div_labeled'] = (market_train_df['vol_price_div'] >= 1).astype(int)

    # Extract: vol_price_div_sqrt & vol_price_div_sqrt_labeled

    market_train_df['vol_price_div_sqrt'] = market_train_df['price_divff'].apply(np.square) * market_train_df['divdiff_volume'].apply(np.square)

    market_train_df['vol_price_div_sqrt_labeled'] = (market_train_df['vol_price_div_sqrt'] >= 1).astype(int)

    # Extract: all_1_return & all_1_return_labeled

    market_train_df['all_1_return'] = market_train_df['returnsClosePrevRaw1'] * market_train_df['returnsOpenPrevRaw1'] * market_train_df['returnsClosePrevMktres1'] * market_train_df['returnsOpenPrevMktres1']

    market_train_df['all_1_return_labeled'] = (market_train_df['all_1_return'] > 0).astype(int)



    # Extract: all_1_return_p_d & all_1_return_p_d_labeled

    # market_train_df['all_1_return_p_d'] = (market_train_df['returnsClosePrevRaw1'] + market_train_df['returnsOpenPrevRaw1'] + market_train_df['returnsClosePrevMktres1'] + market_train_df['returnsOpenPrevMktres1']) / ((market_train_df['returnsClosePrevRaw1'] * market_train_df['returnsOpenPrevRaw1'] * market_train_df['returnsClosePrevMktres1'] * market_train_df['returnsOpenPrevMktres1']) + 1)###

    # market_train_df['all_1_return_p_d_labeled'] = (market_train_df['all_1_return_p_d'] > 0).astype(int)



    # Extract: all_1_return_cont_p & all_1_return_cont_p_labeled

    market_train_df['all_1_return_cont_p'] = market_train_df['returnsClosePrevRaw1'] + market_train_df['returnsOpenPrevRaw1'] + market_train_df['returnsClosePrevMktres1'] + market_train_df['returnsOpenPrevMktres1']

    market_train_df['all_1_return_cont_p_labeled'] = (market_train_df['all_1_return_cont_p'] > 0).astype(int)

    # Extract: ret_c_o_pr10 & ret_c_o_pr10_labeled

    market_train_df['ret_c_o_pr10'] = (market_train_df['returnsClosePrevRaw10'] + 0.00001) / (market_train_df['returnsOpenPrevRaw10'] + 0.00001)

    market_train_df['ret_c_o_pr10_labeled'] = (market_train_df['ret_c_o_pr10'] >= 1).astype(int)

    # Extract: ret_c_o_mk10 & ret_c_o_mk10_labeled

    market_train_df['ret_c_o_mk10'] = (market_train_df['returnsClosePrevMktres10'] + 0.00001) / (market_train_df['returnsOpenPrevMktres10'] + 0.00001)

    market_train_df['ret_c_o_mk10_labeled'] = (market_train_df['ret_c_o_mk10'] >= 1).astype(int)



    # Extract: label

    if 'returnsOpenNextMktres10' in market_train_df.columns:

        market_train_df['label'] = (market_train_df['returnsOpenNextMktres10'] > 0).astype(int)

    print('extract_MarketFeatures -- completed! cost time :{} s'.format(round(time.time() - start_time, 2)))

    

    return market_train_df
def check_nullValues(df):

    df_columns = df.columns

    df_total_len = df.shape[0]

    null_nums = []

    null_percents = []

    for c in df_columns:

        null_num = df[c].isnull().sum()

        null_nums.append(null_num)

        null_percents.append(null_num / df_total_len * 100)

    return pd.DataFrame({'col_name':df.columns, 'null_nums':null_nums, 'null_percents(%)':null_percents})
def get_news_dropList():

    return ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider', 

            'bodySize','headlineTag','marketCommentary','subjects','audiences',

            'assetName','urgency', 'sentimentClass', 'companyCount', 'relevance', 

            'noveltyCount12H', 'noveltyCount24H','sentenceCount','wordCount',

            'firstMentionSentence','sentimentWordCount',

            'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',

            'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D',]

def dorp_useless_newsInfo(news_train_df):

    try:

        news_train_df.drop(get_news_dropList(), axis=1, inplace=True)

    except KeyError as e:

        print('dorp_useless_newsInfo -- KeyError')

    return news_train_df



def transform_news_info(news_train_df):

    start_time = time.time()

    print('transform_news_info -- start...')

    try:

        news_train_df['time'] = news_train_df['time'].dt.strftime('%Y%m%d').astype(int)

    except AttributeError as e:

        print('transform_news_info -- news_train_df[time] -- AttributeError')

    # extracting assetCode from assetCodes

    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")

    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))

    assetCodes_indexes = news_train_df.index.repeat(news_train_df['assetCodes'].apply(len))

    

    assert len(assetCodes_expanded) == len(assetCodes_indexes)

    

    df_assetCodes = pd.DataFrame({'lv_0':assetCodes_indexes, 'assetCode':assetCodes_expanded})

    # merge asset codes data

    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df, left_on='lv_0', right_index=True, suffixes=('', '_old'))

    # for saving precious memory

    del assetCodes_expanded, assetCodes_indexes, df_assetCodes

    print('transform_news_info -- completed! cost time :{} s'.format(round(time.time() - start_time, 2)))

    return news_train_df_expanded



# Extract News Features

def extract_NewsData(news_train_df):

    #firstMentionSentence , sentenceCount

    news_train_df['position'] = news_train_df['firstMentionSentence'] / news_train_df['sentenceCount']

    news_train_df['coverage'] = news_train_df['sentimentWordCount'] / news_train_df['wordCount']

    return news_train_df



class FillSentiment:

    

    def __init__(self, df, senti_name, sg_name):

        self.df = df

        self.senti_name = senti_name

        self.sg_name = sg_name



    def _fill_sentiment(self, s):

        if np.isnan(s[0]):

            return s[1]

        else:

            return s[0]



    def fillin(self):

        f_start_time = time.time()

        pool = multiprocessing.Pool(processes=3)

        try:

            self.df[self.senti_name] = pool.map(self._fill_sentiment, 

                                           self.df.loc[:, [self.senti_name,self.sg_name]].values)

        except Exception as e:

            print('convert_sentiment -- fillin -- {}'.format(e))

        finally:

            pool.close()

            print('Filling -- {} -- null values, cost time :{} s'.format(self.senti_name, 

                                                                        round((time.time() - f_start_time), 2)))

        return self.df[self.senti_name]





def convert_sentiment(mergeData):

    

    def quantile_09(columns):

        return columns.quantile(0.9)

    

    start_time = time.time()

    print('convert_sentiment -- start...')

    # get quantile-0.9-open-price data every day

    group_q09 = mergeData[mergeData['sentimentNegative'].notnull()].groupby(by=['date'])['open'].apply(quantile_09).reset_index()

    # merge quantile-0.9-open-price into dataframe

    print('convert_sentiment -- merging -- group_q09')

    mergeData = pd.merge(mergeData, group_q09, how='left', on=['date'], suffixes=('', '_q09'))

    # for save memory

    del group_q09

    # group out steniments of barometer-stocks every day

    group_senti = mergeData[(mergeData['sentimentNegative'].notnull()) & 

              (mergeData['open'] >= mergeData['open_q09'])].groupby(by=['date']).agg({'sentimentNegative':'mean',

                                                                          'sentimentNeutral':'mean', 

                                                                          'sentimentPositive':'mean'}).reset_index()

    group_senti.rename(columns={'sentimentNegative':'sg_neg',

                                'sentimentNeutral':'sg_neu',

                                'sentimentPositive':'sg_pos'}, inplace=True)

    # merge steniments of barometer-stocks into dataframe

    print('convert_sentiment -- merging -- group_senti')

    mergeData = pd.merge(mergeData, group_senti, how='left', on=['date'])

    # for save memory

    del group_senti

    # fill null-value for sentiments

    mergeData['sentimentNegative'] = FillSentiment(mergeData, 'sentimentNegative', 'sg_neg').fillin()

    mergeData['sentimentNeutral'] = FillSentiment(mergeData, 'sentimentNeutral', 'sg_neu').fillin()

    mergeData['sentimentPositive'] = FillSentiment(mergeData, 'sentimentPositive', 'sg_pos').fillin()

    

    mergeData.drop(['sg_neg','sg_neu','sg_pos', 'open_q09'], axis=1, inplace=True)

    mergeData['sentimentClass'] = mergeData[['sentimentNegative','sentimentNeutral','sentimentPositive']].apply(lambda x:np.argmax(x.values), axis=1)

    print('convert_sentiment -- completed! cost time :{} s'.format(round(time.time() - start_time, 2)))

    return mergeData
global all_data_dropList

all_data_dropList = ['time', 'assetName', 

                     'volume', 'close', 'open', 'date', 'avg_volume', 

                     'universe', 'returnsOpenNextMktres10']



def prepare_data(marketdf, newsdf, deleteAfterTransfer=True, train=True):

    ## Market data

    marketdf = transer_MarketDate(marketdf)

    if train:

        # Fix market WrongPrice

        marketdf = refix_diffWrongPrice(marketdf, 8, 0.001)

    # Truncate market ouliters

    marketdf = trunc_ouliter(truncList, marketdf, 5)

    # Extract market features

    marketdf = extract_MarketFeatures(marketdf)

    

    ## News data

    newsdf = dorp_useless_newsInfo(newsdf)

    newsdf = transform_news_info(newsdf)

    newsdf.drop(['lv_0', 'assetCodes'], axis=1, inplace=True)

    # Extract News Features

    #newsdf = extract_NewsData(newsdf)

    newsdf.rename(columns={'time':'date'}, inplace=True)

    newsdf = newsdf.groupby(by=['date', 'assetCode']).agg('mean').reset_index()

    

    ## Merge Market & News

    dataSetDf = pd.merge(marketdf, newsdf, on=['date','assetCode'], how='left', copy=False)

    if deleteAfterTransfer:

        del marketdf, newsdf

    # fill sentiment null values and conver into 3 classes

    dataSetDf = convert_sentiment(dataSetDf)

    if train:

        dataSetDf.drop(all_data_dropList, axis=1, inplace=True)

    else:

        tmp_drop_list = [col for col in all_data_dropList if col not in ['universe', 'returnsOpenNextMktres10']]

        dataSetDf.drop(tmp_drop_list, axis=1, inplace=True)

    #dataSetDf.iloc[:,-16:] = dataSetDf.iloc[:,-16:].fillna(0)

    dataSetDf = pd.get_dummies(dataSetDf, columns=['sentimentClass'], drop_first=True)

    dataSetDf.drop(['sentimentNegative', 'sentimentNeutral', 'sentimentPositive'], axis=1, inplace=True)

    return dataSetDf
dataSetDf = prepare_data(m_train_df, n_train_df, True)

dataSetDf.drop(['assetCode'], axis=1, inplace=True)

data_x = dataSetDf[[col for col in dataSetDf.columns if col != 'label']]

data_y = dataSetDf['label']

del dataSetDf
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=0)

del data_x, data_y
continueList = []

sparesList = []

for c, d in zip(train_x.columns, train_x.dtypes):

    if str(d).find('int') != -1:

        sparesList.append(c)

    else:

        continueList.append(c)
ssc_x = StandardScaler()

ssc_x.fit(train_x[continueList])

train_x[continueList] = ssc_x.transform(train_x[continueList])



ssc_t_x = StandardScaler()

ssc_t_x.fit(test_x[continueList])

test_x[continueList] = ssc_t_x.transform(test_x[continueList])
params = {

    'max_depth':-1,

    'learning_rate':0.01,

    'num_leaves':60,

    'bagging_fraction':0.9,

    'feature_fraction':0.9,

    'bagging_freq':5,

    'bagging_seed':2019,

    'verbosity':-1,

    'metric':'binary_logloss',

    'objective':'binary',

}
lgb_train = lgb.Dataset(train_x.values, train_y.values)

lgb_test = lgb.Dataset(test_x.values, test_y.values)
lgbmodel = lgb.train(params, lgb_train, 5000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval=200)
train_x.head()
# params_2 = {

#     'max_depth':-1,

#     'learning_rate':0.01,

#     'num_leaves':60,

#     'bagging_fraction':0.9,

#     'feature_fraction':0.9,

#     'bagging_freq':5,

#     'bagging_seed':2019,

#     'verbosity':-1,

#     'metric':'auc',

#     'objective':'binary',

# }
# lgbmodel_2 = lgb.train(params_2, lgb_train, 2000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval=200)
print("generating predictions...")

preddays = env.get_prediction_days()

for marketdf, newsdf, predtemplatedf in preddays:

    dataSetDf = prepare_data(marketdf, newsdf, False, False)

    for_pred_x = dataSetDf[[col for col in dataSetDf.columns if col != 'label']]

    pred_assetCode = for_pred_x['assetCode'].copy(deep=True)

    for_pred_x.drop(['assetCode'], axis=1, inplace=True)

    

    ssc_x_p = StandardScaler()

    ssc_x_p.fit(for_pred_x[continueList])

    for_pred_x[continueList] = ssc_x_p.transform(for_pred_x[continueList])

    

    preds = lgbmodel.predict(for_pred_x, num_iteration=lgbmodel.best_iteration) * 2 - 1

    predsdf = pd.DataFrame({'ast':pred_assetCode,'conf':preds})

    predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values

    env.predict(predtemplatedf)



env.write_submission_file()
# env.write_submission_file()