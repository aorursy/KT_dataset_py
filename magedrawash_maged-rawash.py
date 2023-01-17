!pip3 install --upgrade pandas --force-reinstall # restart the runtime 
!pip3 install seaborn==0.9.0

import os

from numba import jit, vectorize, autojit
import datetime

import time

from scipy.cluster.vq import kmeans, vq 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import copy 
# import os
# print(os.listdir("../input"))
# for carbage collection 
import gc
gc.enable()
# to convert string of set to set 
from datetime import timedelta
# import copy
import dask
from dask import dataframe as dd


import tensorflow as tf 

%matplotlib inline

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers import Dense, Conv1D, BatchNormalization , Flatten, Reshape, LSTM, RepeatVector, TimeDistributed
from keras.activations import tanh
from keras.utils import plot_model


import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
sns.set_context(None)
plt.rcParams['figure.figsize'] = [25, 10]

print('Pandas', pd.__version__)
print('Seaborn', sns.__version__)
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
import re


# get unique codes - not repeated - in one big set 
@jit(parallel=True )
def get_unique_code(cloumn_cat):
  unique_set = cloumn_cat.cat.categories.values.flatten().tolist()
  return  set( re.sub(r'[{\ ]','', "".join( unique_set ) ).replace("'","").replace('}',",").split(',') ) 

@jit(parallel=True )
def get_hits_news(el):
  for code in eval(el):
    if code in subjects_hits:
#       next convert this to list of strings then use count function 
      subjects_hits[code] +=1

@jit
def get_count_news(column):
  column.values.map(get_hits_news)


def get_most_repeated(hits) :
  lis_dict_values =  list(hits.values())
  codebook, _ = kmeans(lis_dict_values, 2)
  return { k:v for k, v in  hits.items() if v >  min(codebook) }


@jit(parallel=True )
def convert_to_stock(item):
  for code in eval(item):
    if  code in stock_universe_set: 
      return code  


@jit(parallel=True )
def sum_sentment(row, subjects):  
#     print("-",end="")
    joined_codes = re.sub(r'[{\ ]','', "".join(  row[1] + str( row[2])  ) ).replace("'","").replace('}',",").split(',') 
    for index in joined_codes  :
      if index in subjects:
        subjects[index]   +=  row[3]
      
# @jit
def input_total_sentment(grouped_news):
  subjects = copy.deepcopy(stocks_sectors_dict)
  # loop over array and set the sentmint 
  _ = [ sum_sentment(row, subjects) for row in grouped_news.values.tolist() ]
  
  return subjects

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

target_index = []
stocks_sectors_dict = {}
stock_last_3M = {}
stock_universe_set = {}
subjects_hits = {}

def train_my_model(market_train_df, news_train_df):
    global target_index, stocks_sectors_dict, stock_last_3M , subjects_hits , stock_universe_set , scaler

    # clean some columns
    news_train_df_cleaned = news_train_df.drop(columns=['sourceId','bodySize','firstCreated','sourceTimestamp','provider','companyCount','sentenceCount','marketCommentary','wordCount','firstMentionSentence'
                                                       ,'sentimentWordCount'])
    del news_train_df

    market_train_df_cleaned = market_train_df #.drop(columns=['assetName'])
    del market_train_df

    # move any news after closing time 22:00 to next day 
    news_train_df_cleaned['time'] = news_train_df_cleaned['time'] + timedelta(hours=2)

    # get all subjects in one big set 
    subject_set = get_unique_code(news_train_df_cleaned['subjects'])
    print('subject_set : Done ')

    # get all assetCodes in one big set 
    stock_universe_set = market_train_df_cleaned['assetCode'][market_train_df_cleaned['universe']> 0].unique()
    print('stock_universe_set : Done ')



    subjects_hits = dict.fromkeys(subject_set, 0.0)

    # count total hits for each subject to get the most important subjects 
    get_count_news(news_train_df_cleaned['subjects'])


    subjects_filtered = get_most_repeated(subjects_hits)


    del subjects_hits


    stocks_sectors = np.concatenate(( list(stock_universe_set), list(subjects_filtered.keys())))

    stocks_sectors_dict = dict.fromkeys(set(stocks_sectors), 0.0) 

    print('Stock and subjects preparation ended ')


    ### NEWS
    news_train_df_cleaned['sentiment'] = ( news_train_df_cleaned['sentimentPositive'] - news_train_df_cleaned['sentimentNegative'] ) * news_train_df_cleaned['relevance'] 

    news_df_sentiment = news_train_df_cleaned[['time', 'subjects', 'assetCodes', 'sentiment']].copy()

    ### Stock
    market_train_df = market_train_df_cleaned[['time', 'assetCode', 'returnsClosePrevMktres1', 'returnsClosePrevMktres10', 'returnsOpenNextMktres10','universe' ]].copy() #, 'universe'

    del market_train_df_cleaned , news_train_df_cleaned
    
    print('news ans market minimized ')

    # clean RAM  
    gc.collect()


    # map assets codes in news to stock asset code 
    news_df_sentiment['assetCodes'] = news_df_sentiment['assetCodes'].map(convert_to_stock) 



    stock_last_3M = market_train_df.set_index('time').last('5M')['assetCode'][market_train_df.set_index('time').last('5M')['universe']>0].unique()
    market_train_df =  market_train_df.drop(['universe'], axis=1)

    #  group by year, month, day 
    sort_news_by_day = news_df_sentiment.groupby([news_df_sentiment.time.dt.year, news_df_sentiment.time.dt.month, news_df_sentiment.time.dt.day]) # days  # 1min 36s per loop 
    sort_stock_by_day = market_train_df.groupby([market_train_df.time.dt.year, market_train_df.time.dt.month, market_train_df.time.dt.day]) # days  # 1min 36s per loop 

    print('grouped by days ')
    print('merge news and stock started ')

    data_set = []
    for news_name, news_group in sort_news_by_day:
      total_sentmint_day = input_total_sentment(news_group)
      total_sentmint_day_df = pd.DataFrame.from_dict(total_sentmint_day, orient='index').reset_index()
      total_sentmint_day_df.rename(columns={'index': 'assetCode', 0: 'sentiment' }, inplace=True)
      if news_name in sort_stock_by_day.groups :
        stock_group = sort_stock_by_day.get_group(news_name)
        stock_group.drop( columns=['time'], inplace=True)
        df = pd.merge(total_sentmint_day_df, stock_group , how='left', on=['assetCode'])
      else:
        stock_group = pd.DataFrame(columns=('assetCode','returnsClosePrevMktres1','returnsClosePrevMktres10', 'returnsOpenNextMktres10') )
        df = pd.merge(total_sentmint_day_df, stock_group , how='left', on=['assetCode'])
      data_set.append( df.values )
#       print("-",end="")

    
    print('data set collected')

    np_data_set = np.stack( data_set, axis=0 )

    del data_set, news_df_sentiment, market_train_df, sort_news_by_day, sort_stock_by_day, subject_set,  subjects_filtered 

    print('Scaling data')

    # remove nan to zero 
    data_set = np_data_set[:,:,1:].astype(np.float) 

    where_are_NaNs = np.isnan(data_set)
    data_set[where_are_NaNs] = 0

    # prepare the data to scalling process 
    days, codes, data = data_set.shape
    data_set = data_set.reshape((days, codes*data))

    # scal the data set 
    data_set = scaler.fit_transform( data_set )
    # reshape back the data 
    data_set = data_set.reshape((days, codes,data))

    # slice data set to features and targets 
    features = data_set[:,:,:-1]
    code_list = np_data_set[:,:,0]
    target = data_set[:,:,-1]

    target_index = [indx for indx ,code in enumerate(code_list[0]) if code  in stock_last_3M]

    stock_target =  target[:,target_index]

    print('target shape',target.shape)
    print('stock_target shape',stock_target.shape)
    print('features shape ', features.shape)


    # roll back two days to get the future of the 
    # axis = 0 to get the roll on just days days 
    # stock_target[:,:,0]  = np.roll( stock_target[:,:,0], -1 , axis=0)
    
    output_length = stock_target.shape[1] 
    input_shape = features.shape[1:]

    print('start model creation')

    x_train, x_test, y_train, y_test = train_test_split(features , stock_target , test_size = 0.2, random_state = None)


    epochs = 750
    batch_size = 30


    #  model creation 
    x_placeholder = Input(shape=input_shape, name='news_input') 

    # hidden
    layer_conv1 = Conv1D(5, 1 , activation='tanh')(x_placeholder)
    layer_conv2 = Conv1D(1, 1 , activation='tanh')(layer_conv1)

    layer_flatten = Flatten()(layer_conv2)
    layer_batch1 = BatchNormalization()(layer_flatten)
    layer_dense = Dense(1000,  activation='tanh' )(layer_batch1)
    layer_dense = Dense(200, activation='tanh')(layer_dense)

    layer_dense = Dense(200, activation='tanh')(layer_dense)
    output = Dense(output_length,  activation='linear')(layer_dense)

    model = Model(inputs=x_placeholder,outputs=output)
    plot_model(model,to_file='demo.png',show_shapes=True)


    model.compile(loss='mean_absolute_error', optimizer='adadelta',
                metrics=['mse'])


    print('start fitting the model ')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test ),verbose=0)

    result = y_test
    predicted = model.predict(x_test)

    print('fitting model ended, and model created ')

    # print(hist.history.keys())
    # "Loss"
    plt.plot(result[:, 10] )
    plt.plot(predicted[:, 10] )
    plt.title('model loss') 
    plt.ylabel('change')
    plt.xlabel('dayes')
    plt.legend(['test', 'predicted'], loc='upper left')
    plt.show()

    return model

def make_my_predictions(market_obs_df, news_obs_df, predictions_template_df):
    global target_index, stocks_sectors_dict, stock_last_3M , subjects_hits , scaler

    ### minify NEWS
    news_obs_df['sentiment'] = ( news_obs_df['sentimentPositive'] - news_obs_df['sentimentNegative'] ) * news_obs_df['relevance'] 

    news_obs_df = news_obs_df[['time', 'subjects', 'assetCodes', 'sentiment']].copy()

    ### minify Stock
    market_obs_df = market_obs_df[[ 'assetCode', 'returnsClosePrevMktres1', 'returnsClosePrevMktres10']].copy() #, 'universe'

    # map assets codes in news to stock asset code 
#     news_obs_df['assetCodes'] = news_obs_df['assetCodes'].map(convert_to_stock) 
    

    total_sentmint_day = input_total_sentment(news_obs_df)
#     del news_obs_df 
    total_sentmint_day_df = pd.DataFrame.from_dict(total_sentmint_day, orient='index').reset_index()
    total_sentmint_day_df.rename(columns={'index': 'assetCode', 0: 'sentiment' }, inplace=True)
    
    df = pd.merge(total_sentmint_day_df, market_obs_df , how='left', on=['assetCode'])
    

    np_data_set = df.values

    # remove nan to zero 
    data_set = np_data_set[:,1:].astype(np.float) 

    where_are_NaNs = np.isnan(data_set)
    data_set[where_are_NaNs] = 0.0
    
    data_set = np.array([data_set])
   
    # prepare the data to scalling process 
    days, codes, data = data_set.shape
    data_set = data_set.reshape((days, codes*data))

    # scal the data set 
    data_set = scaler.fit_transform( data_set )
    # reshape back the data 
    data_set = data_set.reshape((days, codes,data))

    # slice data set to features and targets 
    features = data_set
    code_list = np_data_set[:,0]
    # predict and get univers codes 
    predicted = model.predict(features)
    predicted = predicted.flatten()

    #     merge predicted values of univers stock with it stock code  
    predicted_codes_dict = { code : predicted[i] for i,code in enumerate( code_list[target_index] ) }
    #     assign each value to  predictions_template_df pandas dataframe 
    predictions_template_df = predictions_template_df.set_index('assetCode')
    for code, value in predicted_codes_dict.items():
        if code in predictions_template_df.index:
            predictions_template_df.at[code,'confidenceValue' ] =  value
        
    predictions_template_df['confidenceValue'] = predictions_template_df['confidenceValue'].clip(-1,1)
    return predictions_template_df.reset_index()
    
    
    

    
(market_train_df, news_train_df) = env.get_training_data()
model = train_my_model(market_train_df, news_train_df)
gc.collect()
days = env.get_prediction_days()
for (market_obs_df, news_obs_df, predictions_template_df) in days:    
  predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
  env.predict(predictions_df)


env.write_submission_file()