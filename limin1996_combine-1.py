# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(marketdf, newsdf) = env.get_training_data()

def preprocess(marketdf, newsdf,encode=None):

    news=newsdf.drop(['sourceTimestamp', 'firstCreated', 'sourceId', 'headline','takeSequence','provider', 'subjects', 'audiences', 'headlineTag'],axis=1)

    news['sentiment_v']=(abs(news['sentimentNegative']-news['sentimentPositive']))*news['sentimentWordCount']/news['wordCount']

    news_1=news[['time', 'urgency','companyCount','assetName','relevance', 'sentimentClass','sentiment_v','noveltyCount12H', 'noveltyCount24H',
           'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
           'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']]

    news_1['time'] = (news_1['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
    marketdf['time'] = marketdf['time'].dt.floor('1D')
    # use how = left to aviod vanish of some company without news
    com=pd.merge(marketdf, news_1, how='left', on=['time', 'assetName'])
    com_1=com.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    com_1=com_1.fillna(0)
    if encode==None:
        encode={name:id_asset for id_asset,name in enumerate(com_1.assetCode.astype(str).unique())}

    com_2=com_1.assetCode.apply(lambda x :get_encoder(name=x,encode=encode))   # encoded series
    com_1.assetCode=com_2
    return com_1,encode

def get_encoder(name,encode):
    if name in encode.keys():
        return encode[name]
    else:
        return 5

com_1,encode = preprocess(marketdf, newsdf)




col=[i for i in com_1.columns if i not in ['returnsOpenNextMktres10','universe']]
target='returnsOpenNextMktres10'

# use all data to train the model
train_r=com_1[target]
train_data=com_1[col]


train_r=np.clip(train_r,a_min=-1,a_max=1)

def get_train_x(train_data):
    x=dict()
    x['Code']=train_data['assetCode']
    x['num']=train_data.iloc[:,2:].values
    return x

X=get_train_x(train_data)       


import copy

def create_multi(r_train):
    y=copy.deepcopy(r_train)
    y1=pd.Series(copy.deepcopy(y))
    z=pd.Series([0]*len(y))
    z[(y<y1.quantile(0.1)).tolist()]=1
    z[((y1.quantile(0.1)<=y)&(y<y1.quantile(0.2))).tolist()]=2
    z[((y1.quantile(0.2)<=y)&(y<y1.quantile(0.3))).tolist()]=3
    z[((y1.quantile(0.3)<=y)&(y<y1.quantile(0.4))).tolist()]=4
    z[((y1.quantile(0.4)<=y)&(y<y1.quantile(0.5))).tolist()]=5
    z[((y1.quantile(0.5)<=y)&(y<y1.quantile(0.6))).tolist()]=6
    z[((y1.quantile(0.6)<=y)&(y<y1.quantile(0.7))).tolist()]=7
    z[((y1.quantile(0.7)<=y)&(y<y1.quantile(0.8))).tolist()]=8
    z[((y1.quantile(0.8)<=y)&(y<y1.quantile(0.9))).tolist()]=9
    z[(y1.quantile(0.9)<=y).tolist()]=10
    return z

cata_train_r=create_multi(train_r)
# one-hot encoding the label 1-6
from keras.utils import to_categorical
encoded_z = to_categorical(cata_train_r)
encoded_z=encoded_z[:,1:]
#encoded_z_valid=to_categorical(z_valid)[:,1:]
cata_input=Input(shape=[1], name='Code')
embed=Embedding(input_dim=len(encode.keys()), output_dim=30, input_length=1)(cata_input)
categorical_logits = Flatten()(embed)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(26,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(64,activation='relu')(numerical_logits)
numerical_logits=Dropout(0.5)(numerical_logits)
numerical_logits = Dense(32,activation='relu')(numerical_logits)


logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
logits=Dropout(0.5)(logits)
logits = Dense(32,activation='relu')(logits)
logits=Dropout(0.5)(logits)
logits = Dense(16,activation='relu')(logits)
out = Dense(10,activation='sigmoid')(logits)  

model = Model(inputs = [cata_input] + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint
check_point = ModelCheckpoint('model.hdf_4p',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)

model.fit(X,encoded_z,
          epochs=2,
          verbose=True,
          callbacks=[early_stop,check_point]) 
# model.fit(X,encoded_z,
#           epochs=1,
#           verbose=True,
#           callbacks=[early_stop,check_point]) 
#model.predict()
y=pd.Series(copy.deepcopy(train_r))
a=y.quantile(np.arange(0.1,1,0.1).tolist())

def decode_pre(d1):
    decode_z_valid_pre=[np.argmax(d1[i,:])+1 for i in range(d1.shape[0])]

    q_list=[]
    for i in range(len(a)+1):
        if i==0:
            tmp=(a.iloc[0]-a.iloc[1])/2+a.iloc[0]
        elif i==(len(a)):
            tmp=(a.iloc[-1]-a.iloc[-2])/2+a.iloc[-1]
        else:
            tmp=(a.iloc[i-1]+a.iloc[i])/2
        q_list.append(tmp)
    pre=pd.Series(decode_z_valid_pre)
    pre=pre.replace([i+1 for i in range(len(a)+1)],q_list)
    return pre
#p=model.predict({'Code': train_data['assetCode'][:100],'num': train_data.iloc[:100,2:].values})
days = env.get_prediction_days()
#(market_obs_df, news_obs_df, predictions_template_df) = next(days)

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    com,_=preprocess(market_obs_df, news_obs_df,encode=encode)
    X=get_train_x(com[col])
    y=model.predict(X)
    pre=decode_pre(y)
    predictions_template_df.confidenceValue=pre
    env.predict(predictions_template_df)
    
    
    
env.write_submission_file()