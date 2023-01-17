from kaggle.competitions import twosigmanews
# get twosigma environment
env = twosigmanews.make_env()
print('Done!')
#get data
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
news_train_df.head()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.models import model_from_json
import os
import datetime

%matplotlib inline
# flattening asset codes and making one row/per asset code
news_train_df['assetCodesList'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")
assetCodes_expanded = list(chain(*news_train_df['assetCodesList']))
assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodesList'].apply(len) )

# To aggregate for multiple news data on same asset code and same date
news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}
df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
# Create expandaded news (will repeat every assetCodes' row)
news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

# Free memory
del news_train_df, df_assetCodes
news_train_df_expanded.drop('assetCodes', inplace=True, axis=1)
# splitting time by 22H in news to synch with market
news_train_df_expanded['time'] = (news_train_df_expanded['time'] - np.timedelta64(22,'h')).dt.ceil('1D')

# Round time of market_train_df to 0h of curret day
market_train_df['time'] = market_train_df['time'].dt.floor('1D')
# creating dummy variables for day of week and month
market_train_df['dow'] = market_train_df['time'].dt.weekday_name
market_train_df['month'] = market_train_df['time'].dt.month
market_train_df=pd.get_dummies(market_train_df,columns=['dow','month'],drop_first =True,prefix=['d','m'])
#Find list of unique asset codes
unique_asset_codes = market_train_df['assetCode'].unique()
# unique_asset_codes = unique_asset_codes[1:]
refresh = True
file_list = os.listdir(".")
len(unique_asset_codes)
count = 0
model_count =0
model_dict = dict()
y_test_predict_list = list()
predicted_asset_list = list()
for asset in unique_asset_codes:
    # for each asset code
    count = count+1
    print('count :',count)
    print('asset :',asset)
    json_name = asset+".json"
    h5_name = asset+".h5"
    # I am also saving each model by asset names
    if refresh or json_name not in file_list or h5_name not in file_list:
        #refresh is just a switch if required to rebuild certain models already created or proceed with those
        # which does not exit
        market_asset_data = market_train_df[market_train_df['assetCode']==asset]
        #I am filling the missing data , which is mostly the first few rows for returnsClose/OpenPrev1/10 with average
        market_asset_data.fillna(market_asset_data.mean(),axis=0)
        news_asset_data = news_train_df_expanded[news_train_df_expanded['assetCode']==asset]
        news_asset_data_agg = news_asset_data.groupby(['time', 'assetCode']).agg(news_cols_agg)
        del news_asset_data
        news_asset_data_agg = news_asset_data_agg.apply(np.float32)

        # Flat columns after aggregation
        news_asset_data_agg.columns = ['_'.join(col).strip() for col in news_asset_data_agg.columns.values]
        combined_asset_df = market_asset_data.join(news_asset_data_agg, on=['time', 'assetCode'])
        
        if combined_asset_df['time'].max().date()==datetime.date(2016,12,30):
            # if complete time series is present till end of December 2016 then we take this approach
            predicted_asset_list.append(asset)
            # fill nana values where news data does not exit
            fill_na_dict = dict()
            for col in news_asset_data_agg.columns:
                fill_na_dict[col]=0
            del market_asset_data
            del news_asset_data_agg
            combined_asset_df.fillna(fill_na_dict,inplace=True)
            combined_asset_df.drop(combined_asset_df.head(10).index, inplace=True)
            combined_asset_df.set_index('time',inplace=True)
            Y = combined_asset_df['returnsOpenNextMktres10'].values
            X = combined_asset_df.drop(['returnsOpenNextMktres10','assetCode','assetName'],axis=1).values
            sc = MinMaxScaler(feature_range=(0,1))
            X = sc.fit_transform(X)

            x_input = []
            y_input = []
            # take 50 time steps for lstm input (samples, time_steps,features)
            for i in range(50,X.shape[0]):
                x_input.append(X[i-50:i,:])
                y_input.append(Y[i-1])
            x_input,y_input = np.array(x_input),np.array(y_input)
            x_train = x_input[0:round(0.9*x_input.shape[0]),:]
            y_train = y_input[0:round(0.9*x_input.shape[0])]
            x_test = x_input[round(0.9*x_input.shape[0]):,:]
            y_test = y_input[round(0.9*x_input.shape[0]):]

            #regression based LSTM model
            regresser = Sequential()
            regresser.add(LSTM(units = 50, return_sequences = True, input_shape=(x_train.shape[1],x_train.shape[2])))
            regresser.add(Dropout(0.5))

            regresser.add(LSTM(units = 50, return_sequences = True))
            regresser.add(Dropout(0.5))

            regresser.add(LSTM(units = 50, return_sequences = True))
            regresser.add(Dropout(0.5))

            regresser.add(LSTM(units = 50))
            regresser.add(Dropout(0.5))

            regresser.add(Dense(units = 1))
            regresser.compile(optimizer = 'adam', loss = 'mean_squared_error')
            regresser.fit(x_train,y_train,epochs=100,batch_size=32)

            #model_dict[asset] = regresser
            regresser.save_weights(h5_name)
            model_json = regresser.to_json()
            with open(json_name, "w") as json_file:
                json_file.write(model_json)
            
            y_predict = regresser.predict(x_test)
            # scoring and storing
            score_test = regresser.evaluate(x_test,y_test)
            score_train = regresser.evaluate(x_train,y_train)
            detail = dict()
            detail['model'] = regresser
            detail['score_train'] = score_train
            detail['score_test'] = score_test
            model_dict[asset] = detail
            y_test_predict_list.append({'y_test':y_test,'y_predict':y_predict})
            print(score_test)
            model_count = model_count+1
            print('no of model created ',model_count)
            if model_count == 3:
                break
            # print(regresser.evaluate(x_test))
    
    
    
    #break
print('train-',model_dict[predicted_asset_list[0]]['score_train'])
print('test-',model_dict[predicted_asset_list[0]]['score_test'])
plt.plot(y_test_predict_list[0]['y_test'])
plt.plot(y_test_predict_list[0]['y_predict'])
print('train-',model_dict[predicted_asset_list[1]]['score_train'])
print('test-',model_dict[predicted_asset_list[1]]['score_test'])
plt.plot(y_test_predict_list[1]['y_test'])
plt.plot(y_test_predict_list[1]['y_predict'])
print('train-',model_dict[predicted_asset_list[2]]['score_train'])
print('test-',model_dict[predicted_asset_list[2]]['score_test'])
plt.plot(y_test_predict_list[2]['y_test'])
plt.plot(y_test_predict_list[2]['y_predict'])
