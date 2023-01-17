#importing basic libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix

from keras.models import Sequential,load_model

from keras.layers import Dense,Dropout

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.optimizers import Adam



pd.set_option('display.max_column',None)

pd.set_option('display.max_row',None)



# importing data

train_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/online_news_popularity/train_set_label.csv")

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/online_news_popularity/test_set_label.csv')



# inspecting train data

display(train_data.head(5))
test_data.shape
test_data.describe()
# inspecting train data

display(train_data.columns)
# column names have space at front

train_data.columns=['url','timedelta', 'n_tokens_title', 'n_tokens_content','n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens','num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',

       'average_token_length', 'num_keywords', 'data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech',

       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg','kw_max_avg', 'kw_avg_avg', 'self_reference_min_shares',

       'self_reference_max_shares', 'self_reference_avg_sharess','weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday','weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',

       'weekday_is_sunday', 'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02','LDA_03', 'LDA_04', 'global_subjectivity','global_sentiment_polarity', 'global_rate_positive_words',

       'global_rate_negative_words', 'rate_positive_words','rate_negative_words', 'avg_positive_polarity','min_positive_polarity', 'max_positive_polarity','avg_negative_polarity', 'min_negative_polarity',

       'max_negative_polarity', 'title_subjectivity','title_sentiment_polarity', 'abs_title_subjectivity','abs_title_sentiment_polarity', 'shares']

test_data.columns=['url','timedelta', 'n_tokens_title', 'n_tokens_content','n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens','num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',

       'average_token_length', 'num_keywords', 'data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech',

       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg','kw_max_avg', 'kw_avg_avg', 'self_reference_min_shares',

       'self_reference_max_shares', 'self_reference_avg_sharess','weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday','weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',

       'weekday_is_sunday', 'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02','LDA_03', 'LDA_04', 'global_subjectivity','global_sentiment_polarity', 'global_rate_positive_words',

       'global_rate_negative_words', 'rate_positive_words','rate_negative_words', 'avg_positive_polarity','min_positive_polarity', 'max_positive_polarity','avg_negative_polarity', 'min_negative_polarity',

       'max_negative_polarity', 'title_subjectivity','title_sentiment_polarity', 'abs_title_subjectivity','abs_title_sentiment_polarity',]
# feature Engineering

#Creating new Feature from 'url' column

train_data['date']=train_data['url'].str[20:30]

train_data['url']=train_data['url'].str[30:].str.replace("/","").str.split('-')

train_data['date']=pd.to_datetime(train_data['date'])

train_data['week']=train_data['date'].dt.isocalendar().week

train_data['month']=train_data['date'].dt.month



test_data['date']=test_data['url'].str[20:30]

test_data['url']=test_data['url'].str[30:].str.replace("/","").str.split('-')

test_data['date']=pd.to_datetime(test_data['date'])

test_data['week']=test_data['date'].dt.isocalendar().week

test_data['month']=test_data['date'].dt.month
train_data['kw_min_min']=train_data['kw_min_min']+1

train_data['kw_avg_min']=train_data['kw_avg_min']+1

train_data['kw_min_avg']=train_data['kw_min_avg']+1



test_data['kw_min_min']=test_data['kw_min_min']+1

test_data['kw_avg_min']=test_data['kw_avg_min']+1

test_data['kw_min_avg']=test_data['kw_min_avg']+1
train_data['shares']=np.log(train_data['shares'])



train_data['sharesperday']=train_data['shares']/train_data['timedelta']

#########



train_data['weekday_is_monday']=train_data['weekday_is_monday']*2

train_data['weekday_is_tuesday']=train_data['weekday_is_tuesday']*3

train_data['weekday_is_wednesday']=train_data['weekday_is_wednesday']*4

train_data['weekday_is_thursday']=train_data['weekday_is_thursday']*5

train_data['weekday_is_friday']=train_data['weekday_is_friday']*6

train_data['weekday_is_saturday']=train_data['weekday_is_saturday']*7

train_data['weekday_is_sunday']=train_data['weekday_is_sunday']*8



test_data['weekday_is_monday']=test_data['weekday_is_monday']*2

test_data['weekday_is_tuesday']=test_data['weekday_is_tuesday']*3

test_data['weekday_is_wednesday']=test_data['weekday_is_wednesday']*4

test_data['weekday_is_thursday']=test_data['weekday_is_thursday']*5

test_data['weekday_is_friday']=test_data['weekday_is_friday']*6

test_data['weekday_is_saturday']=test_data['weekday_is_saturday']*7

test_data['weekday_is_sunday']=test_data['weekday_is_sunday']*8



train_data['weekday']=train_data['weekday_is_monday']+train_data['weekday_is_tuesday']+train_data['weekday_is_wednesday']+train_data['weekday_is_thursday']+train_data['weekday_is_friday']+train_data['weekday_is_saturday']+train_data['weekday_is_sunday']

test_data['weekday']=test_data['weekday_is_monday']+test_data['weekday_is_tuesday']+test_data['weekday_is_wednesday']+test_data['weekday_is_thursday']+test_data['weekday_is_friday']+test_data['weekday_is_saturday']+test_data['weekday_is_sunday']





train_data['data_channel_is_lifestyle']=train_data['data_channel_is_lifestyle']*2

train_data['data_channel_is_entertainment']=train_data['data_channel_is_entertainment']*3

train_data['data_channel_is_bus']=train_data['data_channel_is_bus']*4

train_data['data_channel_is_socmed']=train_data['data_channel_is_socmed']*5

train_data['data_channel_is_tech']=train_data['data_channel_is_tech']*6

train_data['data_channel_is_world']=train_data['data_channel_is_world']*7



test_data['data_channel_is_lifestyle']=test_data['data_channel_is_lifestyle']*2

test_data['data_channel_is_entertainment']=test_data['data_channel_is_entertainment']*3

test_data['data_channel_is_bus']=test_data['data_channel_is_bus']*4

test_data['data_channel_is_socmed']=test_data['data_channel_is_socmed']*5

test_data['data_channel_is_tech']=test_data['data_channel_is_tech']*6

test_data['data_channel_is_world']=test_data['data_channel_is_world']*7



train_data['data_channel']=train_data['data_channel_is_lifestyle']+train_data['data_channel_is_entertainment']+train_data['data_channel_is_bus']+train_data['data_channel_is_socmed']+train_data['data_channel_is_tech']+train_data['data_channel_is_world']

test_data['data_channel']=test_data['data_channel_is_lifestyle']+test_data['data_channel_is_entertainment']+test_data['data_channel_is_bus']+test_data['data_channel_is_socmed']+test_data['data_channel_is_tech']+test_data['data_channel_is_world']

embedding_share_weekday=train_data.groupby('weekday')['sharesperday'].median()

train_data['shares_week']=train_data['weekday'].map(embedding_share_weekday)

train_data['shares_1']=train_data['shares_week']*train_data['timedelta']

test_data['shares_week']=test_data['weekday'].map(embedding_share_weekday)

test_data['shares_1']=test_data['shares_week']*test_data['timedelta']



embedding_share_datachannel=train_data.groupby('data_channel')['sharesperday'].median()

train_data['shares_datachannel']=train_data['data_channel'].map(embedding_share_datachannel)

train_data['shares_2']=train_data['shares_datachannel']*train_data['timedelta']

test_data['shares_datachannel']=test_data['data_channel'].map(embedding_share_datachannel)

test_data['shares_2']=test_data['shares_datachannel']*test_data['timedelta']



embedding_share_date=train_data.groupby('timedelta')['sharesperday'].median()

train_data['shares_timedelta']=train_data['timedelta'].map(embedding_share_date)

train_data['shares_3']=train_data['shares_timedelta']*train_data['timedelta']

test_data['shares_timedelta']=test_data['timedelta'].map(embedding_share_date)

test_data['shares_3']=test_data['shares_timedelta']*test_data['timedelta']
train_data['popularity']=train_data['sharesperday'].apply(lambda x:1 if (x>.2) else 0)
train_data.info()
numerical_columns=train_data.select_dtypes(exclude=['object','datetime']).columns



train_data=train_data[numerical_columns]

#test_data=test_data[numerical_columns]

X=train_data.drop(['sharesperday','shares','shares_week','shares_1','shares_datachannel','shares_2','shares_timedelta','shares_3','popularity'],axis=1)

y=train_data[['popularity']]



test_X=test_data[X.columns]
scale=MinMaxScaler()

X=pd.DataFrame(scale.fit_transform(X),columns=X.columns)

test_X=pd.DataFrame(scale.transform(test_X),columns=X.columns)
#train_data=train_data[train_data['popularity']==0]

train_data['popularity'].value_counts()
sampler = SMOTE(sampling_strategy='minority')

X_train_smote, y_train_smote = sampler.fit_sample(X,y)

X_train,X_test,y_train,y_test=train_test_split(X_train_smote, y_train_smote,test_size=0.2,random_state=5,stratify=y_train_smote)





model=Sequential()

model.add(Dense(50,activation='relu',input_shape=(X.shape[1],)))

model.add(Dropout(0.3))

model.add(Dense(120,activation='sigmoid'))

model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

opti=Adam(lr=0.01)

model.compile(optimizer=opti,loss='binary_crossentropy',metrics=['accuracy'])

ear=EarlyStopping(patience=2)

mod=ModelCheckpoint('best.hd5',save_best_only=True)

model.fit(X_train,y_train,validation_split=0.4,batch_size=100,epochs=50,callbacks=[ear,mod])
model=load_model('./best.hd5')

y_predict=model.predict(X_test)

y_predict=y_predict>0.5

print(classification_report(y_test,y_predict))

print(confusion_matrix(y_test,y_predict))
test_y_predict=model.predict(test_X)

test_data['popularity']=test_y_predict>0.5

test_data['popularity']=test_data['popularity'].astype('int')
test_data['popularity'].value_counts()
train_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/online_news_popularity/train_set_label.csv",index_col='url')

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/online_news_popularity/test_set_label.csv',index_col='url')

test_data['popularity']=test_y_predict>0.5

test_data['popularity']=test_data['popularity'].astype('int')
train_data.columns=['timedelta', 'n_tokens_title', 'n_tokens_content','n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens','num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',

       'average_token_length', 'num_keywords', 'data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech',

       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg','kw_max_avg', 'kw_avg_avg', 'self_reference_min_shares',

       'self_reference_max_shares', 'self_reference_avg_sharess','weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday','weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',

       'weekday_is_sunday', 'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02','LDA_03', 'LDA_04', 'global_subjectivity','global_sentiment_polarity', 'global_rate_positive_words',

       'global_rate_negative_words', 'rate_positive_words','rate_negative_words', 'avg_positive_polarity','min_positive_polarity', 'max_positive_polarity','avg_negative_polarity', 'min_negative_polarity',

       'max_negative_polarity', 'title_subjectivity','title_sentiment_polarity', 'abs_title_subjectivity','abs_title_sentiment_polarity', 'shares']

test_data.columns=['timedelta', 'n_tokens_title', 'n_tokens_content','n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens','num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',

       'average_token_length', 'num_keywords', 'data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech',

       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg','kw_max_avg', 'kw_avg_avg', 'self_reference_min_shares',

       'self_reference_max_shares', 'self_reference_avg_sharess','weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday','weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',

       'weekday_is_sunday', 'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02','LDA_03', 'LDA_04', 'global_subjectivity','global_sentiment_polarity', 'global_rate_positive_words',

       'global_rate_negative_words', 'rate_positive_words','rate_negative_words', 'avg_positive_polarity','min_positive_polarity', 'max_positive_polarity','avg_negative_polarity', 'min_negative_polarity',

       'max_negative_polarity', 'title_subjectivity','title_sentiment_polarity', 'abs_title_subjectivity','abs_title_sentiment_polarity','popularity']



# feature Engineering

#Creating new Feature from 'url' column

"""train_data['date']=train_data['url'].str[20:30]

train_data['url']=train_data['url'].str[30:].str.replace("/","").str.split('-')

train_data['date']=pd.to_datetime(train_data['date'])

train_data['week']=train_data['date'].dt.isocalendar().week

train_data['month']=train_data['date'].dt.month



test_data['date']=test_data['url'].str[20:30]

test_data['url']=test_data['url'].str[30:].str.replace("/","").str.split('-')

test_data['date']=pd.to_datetime(test_data['date'])

test_data['week']=test_data['date'].dt.isocalendar().week

test_data['month']=test_data['date'].dt.month"""



train_data['kw_min_min']=train_data['kw_min_min']+1

train_data['kw_avg_min']=train_data['kw_avg_min']+1

train_data['kw_min_avg']=train_data['kw_min_avg']+1



test_data['kw_min_min']=test_data['kw_min_min']+1

test_data['kw_avg_min']=test_data['kw_avg_min']+1

test_data['kw_min_avg']=test_data['kw_min_avg']+1



#train_data['shares']=np.log(train_data['shares'])



train_data['sharesperday']=train_data['shares']/train_data['timedelta']

#########



train_data['weekday_is_monday']=train_data['weekday_is_monday']*2

train_data['weekday_is_tuesday']=train_data['weekday_is_tuesday']*3

train_data['weekday_is_wednesday']=train_data['weekday_is_wednesday']*4

train_data['weekday_is_thursday']=train_data['weekday_is_thursday']*5

train_data['weekday_is_friday']=train_data['weekday_is_friday']*6

train_data['weekday_is_saturday']=train_data['weekday_is_saturday']*7

train_data['weekday_is_sunday']=train_data['weekday_is_sunday']*8



test_data['weekday_is_monday']=test_data['weekday_is_monday']*2

test_data['weekday_is_tuesday']=test_data['weekday_is_tuesday']*3

test_data['weekday_is_wednesday']=test_data['weekday_is_wednesday']*4

test_data['weekday_is_thursday']=test_data['weekday_is_thursday']*5

test_data['weekday_is_friday']=test_data['weekday_is_friday']*6

test_data['weekday_is_saturday']=test_data['weekday_is_saturday']*7

test_data['weekday_is_sunday']=test_data['weekday_is_sunday']*8



train_data['weekday']=train_data['weekday_is_monday']+train_data['weekday_is_tuesday']+train_data['weekday_is_wednesday']+train_data['weekday_is_thursday']+train_data['weekday_is_friday']+train_data['weekday_is_saturday']+train_data['weekday_is_sunday']

test_data['weekday']=test_data['weekday_is_monday']+test_data['weekday_is_tuesday']+test_data['weekday_is_wednesday']+test_data['weekday_is_thursday']+test_data['weekday_is_friday']+test_data['weekday_is_saturday']+test_data['weekday_is_sunday']





train_data['data_channel_is_lifestyle']=train_data['data_channel_is_lifestyle']*2

train_data['data_channel_is_entertainment']=train_data['data_channel_is_entertainment']*3

train_data['data_channel_is_bus']=train_data['data_channel_is_bus']*4

train_data['data_channel_is_socmed']=train_data['data_channel_is_socmed']*5

train_data['data_channel_is_tech']=train_data['data_channel_is_tech']*6

train_data['data_channel_is_world']=train_data['data_channel_is_world']*7



test_data['data_channel_is_lifestyle']=test_data['data_channel_is_lifestyle']*2

test_data['data_channel_is_entertainment']=test_data['data_channel_is_entertainment']*3

test_data['data_channel_is_bus']=test_data['data_channel_is_bus']*4

test_data['data_channel_is_socmed']=test_data['data_channel_is_socmed']*5

test_data['data_channel_is_tech']=test_data['data_channel_is_tech']*6

test_data['data_channel_is_world']=test_data['data_channel_is_world']*7



train_data['data_channel']=train_data['data_channel_is_lifestyle']+train_data['data_channel_is_entertainment']+train_data['data_channel_is_bus']+train_data['data_channel_is_socmed']+train_data['data_channel_is_tech']+train_data['data_channel_is_world']

test_data['data_channel']=test_data['data_channel_is_lifestyle']+test_data['data_channel_is_entertainment']+test_data['data_channel_is_bus']+test_data['data_channel_is_socmed']+test_data['data_channel_is_tech']+test_data['data_channel_is_world']



embedding_share_weekday=train_data.groupby('weekday')['sharesperday'].median()

train_data['shares_week']=train_data['weekday'].map(embedding_share_weekday)

train_data['shares_1']=train_data['shares_week']*train_data['timedelta']

test_data['shares_week']=test_data['weekday'].map(embedding_share_weekday)

test_data['shares_1']=test_data['shares_week']*test_data['timedelta']



embedding_share_datachannel=train_data.groupby('data_channel')['sharesperday'].median()

train_data['shares_datachannel']=train_data['data_channel'].map(embedding_share_datachannel)

train_data['shares_2']=train_data['shares_datachannel']*train_data['timedelta']

test_data['shares_datachannel']=test_data['data_channel'].map(embedding_share_datachannel)

test_data['shares_2']=test_data['shares_datachannel']*test_data['timedelta']



embedding_share_date=train_data.groupby('timedelta')['sharesperday'].median()

train_data['shares_timedelta']=train_data['timedelta'].map(embedding_share_date)

train_data['shares_3']=train_data['shares_timedelta']*train_data['timedelta']

test_data['shares_timedelta']=test_data['timedelta'].map(embedding_share_date)

test_data['shares_3']=test_data['shares_timedelta']*test_data['timedelta']



train_data['popularity']=train_data['sharesperday'].apply(lambda x:1 if (x>49) else 0)



#train_data=train_data[numerical_columns]



train_numerical_columns=train_data.select_dtypes(include=['object','datetime']).columns

test_numerical_columns=test_data.select_dtypes(include=['object','datetime']).columns

train_data=train_data.drop(train_numerical_columns,axis=1)

test_data=test_data.drop(test_numerical_columns,axis=1)
test_data.head()
train_data_0=train_data[train_data['popularity']==0]

train_data_1=train_data[train_data['popularity']==1]



test_data_0=test_data[test_data['popularity']==0]

test_data_1=test_data[test_data['popularity']==1]
X_0=train_data_0.drop(['shares', 'sharesperday'],axis=1)

y_0=np.log(train_data_0['shares'])



X_1=train_data_1.drop(['shares', 'sharesperday'],axis=1)

y_1=np.log(train_data_1['shares'])



test_data_0=test_data_0[X_0.columns]

test_data_1=test_data_1[X_0.columns]
scale=MinMaxScaler()

X_0=pd.DataFrame(scale.fit_transform(X_0),columns=X_0.columns)

test_data_0=pd.DataFrame(scale.transform(test_data_0),columns=X_0.columns,index=test_data_0.index)



scale1=MinMaxScaler()

X_1=pd.DataFrame(scale.fit_transform(X_1),columns=X_1.columns)

test_data_1=pd.DataFrame(scale.transform(test_data_1),columns=X_0.columns,index=test_data_1.index)
test_data_1.head()
test_data_0.shape,test_data_1.shape
X0_train,X0_test,y0_train,y0_test=train_test_split(X_0, y_0,test_size=0.3,random_state=3)

X1_train,X1_test,y1_train,y1_test=train_test_split(X_1, y_1,test_size=0.3,random_state=3)
"""model=Sequential()

model.add(Dense(100,activation='relu',input_shape=(X1_train.shape[1],)))

model.add(Dropout(0.3))

model.add(Dense(500,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(500,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(500,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(500,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1,activation='linear'))

opti=Adam(lr=0.01)

model.compile(optimizer=opti,loss='mean_squared_error')

ear=EarlyStopping(patience=5)

mod=ModelCheckpoint('best1.hd5',save_best_only=True)

model.fit(X0_train,y0_train,validation_split=0.2,batch_size=100,epochs=50,callbacks=[ear,mod])"""



from lightgbm import LGBMRegressor

model = LGBMRegressor(max_depth=-1,n_estimators=10000, importance_type='gain')

model.fit(X0_train,y0_train,eval_set=(X0_test,y0_test),verbose=100,early_stopping_rounds=5)



"""import xgboost as xgb

model=xgb.XGBClassifier(n_estimators=100)

model.fit(X0_train,y0_train)"""
#model=load_model('./best1.hd5')

pred=model.predict(X0_test)

pred_test0=model.predict(test_data_0)

print(np.sqrt(mean_squared_error(y0_test,pred)))
X0_test['prediction']=pred

test_data_0['prediction']=pred_test0
y0_test[:5]
pred[:5]
model=Sequential()

model.add(Dense(50,activation='relu',input_shape=(X1_train.shape[1],)))

model.add(Dropout(0.3))

model.add(Dense(120,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(120,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(120,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1,activation='linear'))

opti=Adam(lr=0.01)

model.compile(optimizer=opti,loss='mean_absolute_error')

ear=EarlyStopping(patience=3)

mod=ModelCheckpoint('best2.hd5',save_best_only=True)

model.fit(X1_train,y1_train,validation_split=0.2,batch_size=100,epochs=50,callbacks=[ear,mod])
model=load_model('./best2.hd5')

pred1=model.predict(X1_test)

pred_test1=model.predict(test_data_1)

print(np.sqrt(mean_squared_error(y1_test,pred1)))
X1_test['prediction']=pred1

test_data_1['prediction']=pred_test1
X1_test.head(5)
y1_test[:10]
pred1[:10]
test_data_final=test_data_0.append(test_data_1)
test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/online_news_popularity/test_set_label.csv',index_col='url')



test_data.head()
test_data_1.head()
test_data_final.shape,test_data_0.shape,test_data_1.shape
test_data_1.head()
test_data_final.head()
test_data.head()
test_data=test_data.merge(test_data_final,how='left',right_index=True,left_on=test_data.index)
test_data.head()
output=np.round(np.exp(test_data[['prediction']]))

output.to_csv('output.csv',index=False)