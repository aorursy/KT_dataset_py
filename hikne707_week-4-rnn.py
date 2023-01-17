import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
import numpy as np
from numpy import random
import warnings
warnings.filterwarnings('ignore')
import datetime
from dateutil.relativedelta import relativedelta

from sklearn import metrics
from sklearn.model_selection import train_test_split

##
import sys
from IPython.display import display
from scipy.stats import skew
import lightgbm as lgb
import os
from tqdm import tqdm
from sklearn import preprocessing
from itertools import product
# DNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from keras import backend as K
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
            
train.head()
train.info()
catcols=train.select_dtypes(include='object').columns.values.tolist()
catcols.remove('Date')
numcols=train.select_dtypes(include='number').columns.values[1:-1].tolist()
train[catcols].describe().merge(test[catcols].describe(),left_index=True,right_index=True,suffixes=('_train','_test'))
print(f'TRAIN -> date_min= {train["Date"].min()} ; date_max= {train["Date"].max()}')
print(f'TEST -> date_min= {test["Date"].min()} ; date_max= {test["Date"].max()}')
len(set(train.Date.unique()) & set(test.Date.unique()))
def clean_loc_features(data):
    # lower Province_State & Country_Region in order to use them to add more information
    data['Province_State']=data['Province_State'].str.lower()
    data['Country_Region']=data['Country_Region'].str.lower()
    

    # Fill missing Province_State & Country_Region missing values 
    data.fillna({'Province_State':'','Country_Region':''},inplace=True)
    
    # Remove non-alpha charachters 
    data['Province_State']=data['Province_State'].apply(lambda x: ''.join([ch for ch in x if ch.isalpha()]))
    data['Country_Region']=data['Country_Region'].apply(lambda x: ''.join([ch for ch in x if ch.isalpha()]))
    
    return data
# Fill missing Province_State & Country_Region missing values 
train=clean_loc_features(train)
test=clean_loc_features(test)
#
test.head(2)
# convert Date into datetime format 
train['Date'] = pd.to_datetime(train['Date'], format = '%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')
# Extract Date characteristics
def create_date_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df
# 
train=create_date_features(train)
test=create_date_features(test)
# 
test.head(3)
# reverse Date transformation
train['Date']=train['Date'].apply(lambda x:x.strftime('%Y-%m-%d'))
test['Date']=test['Date'].apply(lambda x:x.strftime('%Y-%m-%d'))
cols_to_keep=['Country Name','Population ages 0-14, total', 'Population ages 15-64, female','Population ages 15-64, male',
'Population ages 15-64, total','Population ages 65 and above, total',
'Population ages 80 and above, female (% of female population)',
'Population ages 80 and above, male (% of male population)','Population, male',
'Population, total','Rural population (% of total population)']

add_inf=pd.read_csv('/kaggle/input/world-population-and-development-indicators/data.csv')[cols_to_keep]
add_inf.head(3)
add_inf['Country Name'].replace({'bahamasthe':'bahamas','bruneidarussalam':'bahamas','czechrepublic':'czechia',
                   'congodemrep':'congokinshasa','congorep':'congobrazzaville','egyptarabrep':'egypt',
                  'gambiathe':'gambia','iranislamicrep':'iran','korearep':'koreasouth','unitedstates':'us',
                   'kyrgyzrepublic':'kyrgyzstan','russianfederation':'russia','stkittsandnevis':'saintkittsandnevis',
                  'stlucia':'saintlucia','stvincentandthegrenadines':'saintvincentandthegrenadines',
                  'slovakrepublic':'slovakia','syrianarabrepublic':'syria','venezuelarb':'venezuela'},inplace=True)
print('us' in add_inf['Country Name'].unique())

add_inf.drop_duplicates(subset=['Country Name'],inplace=True)
print(add_inf['Country Name'].nunique())
add_inf.describe()
def scale_popultaion_features(data):
    # replace 0 with a more probable value
    data['Population ages 0-14, total'].replace(0,data['Population ages 0-14, total'].mode(),inplace=True)
    data['Population ages 15-64, total'].replace(0,data['Population ages 15-64, total'].mode(),inplace=True)
    data['Population ages 65 and above, total'].replace(0,data['Population ages 65 and above, total'].mode(),inplace=True)
    data['Population, total'].replace(0,data['Population, total'].mode(),inplace=True)
    
    # Normalize
    data['Population ages 15-64, male']=data['Population ages 15-64, male']/data['Population ages 15-64, total']
    data['Population ages 15-64, female']=data['Population ages 15-64, female']/data['Population ages 15-64, total']
    #
    data['Population ages 0-14, total']=data['Population ages 0-14, total']/data['Population, total']
    data['Population ages 15-64, total']=data['Population ages 15-64, total']/data['Population, total']    
    data['Population ages 65 and above, total']=data['Population ages 65 and above, total']/data['Population, total'] 
    #
    data['Population, male']=data['Population, male']/data['Population, total'] 
    
    # rescale % features (bring them back to [0,1] interval)
    pcq_features=[col for col in data.columns if '%' in col]
    print(pcq_features)
    data[pcq_features]=.01*data[pcq_features]
    data.drop(columns=['Population, total'],inplace=True)
    # fillna
    data.fillna(data.median().to_dict(),inplace=True)
    return data
add_inf=scale_popultaion_features(add_inf)
add_inf.describe()
add_inf.isnull().sum()
print('From train:{}  ;  From add_inf:{}  ; intersection:{}'.format(train['Country_Region'].nunique(),add_inf['Country Name'].nunique(),
                                                                    len(set(train['Country_Region']) & set(add_inf['Country Name']))))

print('From test:{}  ;  From add_inf:{}  ; intersection:{}'.format(test['Country_Region'].nunique(),add_inf['Country Name'].nunique(),
                                                                    len(set(test['Country_Region']) & set(add_inf['Country Name']))))
# (1) add world population & developement indicators 
print('train shape: {}  ;  test shape: {}'.format(train.shape,test.shape))
#
train=train.merge(add_inf,left_on='Country_Region',right_on='Country Name',how='left')
test=test.merge(add_inf,left_on='Country_Region',right_on='Country Name',how='left')
#
train.fillna(train.median().to_dict(),inplace=True)
test.fillna(train.median().to_dict(),inplace=True)

print('train shape: {}  ;  test shape: {}'.format(train.shape,test.shape))

train.head()
# concatenate Province_State & Country_Region as "Province_State" ID
train['Province_State']=train['Country_Region']+' '+train['Province_State']
test['Province_State']=test['Country_Region']+' '+test['Province_State']

# reset index with Date
train.index=train['Date']
test.index=test['Date']

# Drop useless columns
train.drop(columns=['Date','Country_Region','Country Name'],inplace=True)
test.drop(columns=['Date','Country_Region','Country Name'],inplace=True)
#
train.head(2)
# Display ConfirmedCases & Fatatilities charts of a random Province state

# Pick one random Province_State
province=np.random.choice(train['Province_State'].unique())
s=train.loc[train['Province_State']==province,['ConfirmedCases','Fatalities']]

plt.style.use('default')
plt.figure(figsize=(10,3))
plt.subplot(121)
s['ConfirmedCases'].plot(kind='area',color='lightskyblue',alpha=.4,label='ConfirmedCases')
s['Fatalities'].plot('area',color='orangered',alpha=.4,label='Fatalities')
plt.legend()
plt.xticks(rotation=80)
plt.title('ConfirmedCases & Fatalities',fontsize=10)
#
plt.subplot(122)
s['Fatalities'].plot('area',color='lightcoral',alpha=.7)
plt.xticks(rotation=80)
plt.title('Fatalities',fontsize=10)

plt.suptitle(province.upper(),fontsize=14)
plt.show()
train.columns
def create_features(df):
    #-----------------------#
    # 1. Global indicators  #
    #-----------------------#
    # add mortality rate
    df['MortalityRate'] = np.where(df['ConfirmedCases']>0,df['Fatalities']/df['ConfirmedCases'],0)
    df['MortalityRate'] = df['MortalityRate'].fillna(0.0)
    #
    # add daily measures
    df['Daily Cases']=df.groupby('Province_State')['ConfirmedCases'].shift(1)
    df['Daily Deaths']=df.groupby('Province_State')['Fatalities'].shift(1)
    # fill na with 0
    df.fillna({'Daily Cases':0,'Daily Deaths':0},inplace=True)

    #
    df['New cases rate']=np.where(df['ConfirmedCases']>0,df['Daily Cases']/df['ConfirmedCases'],np.sign(df['Daily Cases']))
    df['New deaths rate']=np.where(df['Fatalities']>0,df['Daily Deaths']/df['Fatalities'],np.sign(df['Daily Deaths']))

    # fill na with 0
    df.fillna({'New cases rate':0,'New deaths rate':0},inplace=True)
    #
    #-----------------------#
    # 2. Ind by Popultaion  #
    #-----------------------#
    pop_cols=[col for col in df.columns if 'opulation' in col]
    covid_cols=['ConfirmedCases','Fatalities','MortalityRate','Daily Cases','Daily Deaths','New cases rate','New deaths rate']
    for col,cov_col in product(pop_cols,covid_cols):
        df[f'{cov_col}_{col}']=df[col]*df[cov_col]
    #-----------------------#
    # 3. Ind by weekofyear  #
    #-----------------------# 
 
    df=df.merge(df.groupby(['Province_State','weekofyear'],as_index=True).agg(
        week_ConfirmedCases=('ConfirmedCases',sum),
        week_Fatalities=('Fatalities',sum),
        week_Daily_Cases=('Daily Cases','mean'),
        week_Daily_Deaths=('Daily Deaths','mean'),
        week_New_cases_rate=('New cases rate','mean'),
        week_New_deaths_rate=('New deaths rate','mean')),left_on=['Province_State','weekofyear'],right_index=True)
    
    return df
#
print(train.shape)
train=create_features(train)
print(train.shape)

print(train['New cases rate'].describe())
print(train['New deaths rate'].describe())
train.describe()
# log transform highly skewed features
s=train.select_dtypes('number').max()
features_to_transform=s[s>10**4].index.values[1:]
print(features_to_transform)

train[features_to_transform]=np.log(train[features_to_transform]+1)
#
train.describe()
# Show target (ConfirmedCases, Fatalities)
plt.figure(figsize=(10,3))
plt.subplot(121)
train['ConfirmedCases'].plot.hist(bins=50,density=True,color='lightskyblue',edgecolor='k')

plt.title('ConfirmedCases')
#
plt.subplot(122)
train['Fatalities'].plot.hist(bins=50,density=True,color='lightcoral',edgecolor='k')
plt.title('Fatalities')
plt.show()
# Missing values
s=round(100*(train.isnull().sum()/train.shape[0]).to_frame('Nan (%)'),2)
s[s['Nan (%)']>0]
features=train.select_dtypes(include='number').columns.values[1:]
features
# cross-correlation between "Added informtaion" and target
corr={'ConfirmedCases':[],'Fatalities':[]}
for col in features:
    corr['ConfirmedCases'].append(train[[col,'ConfirmedCases']].corr().values[0,1])
    corr['Fatalities'].append(train[[col,'Fatalities']].corr().values[0,1]) 
corr=pd.DataFrame(corr,index=features)
corr['min']=np.abs(corr).min(axis=1)
corr
# drop feature with very low correlation with target
features=corr[corr['min']>.05].index.values
features.shape
def process_seq(df,features):
    # define identifier
    idf=(df['Province_State'].unique()[0],df.index.values)
    # define target 
    tar=df[['ConfirmedCases', 'Fatalities']].values[-1]
    # define sequence 
    seq=df[features].values[:-1,:]
    return idf,seq,tar
def procces_sequnces(train,n=11):
    days_list=train.index.unique()
    identifiers,sequences,targets=[],[],[]
    #
    for i in tqdm(range(days_list.shape[0]-n)):
        dfx=train[(train.index>=days_list[i]) & (train.index<days_list[i+n])].copy()
        #
        #df_sub[features]=preprocessing.MinMaxScaler().fit_transform(df_sub[features])
        out=dfx.groupby(['Province_State']).apply(process_seq,features=features).values
        # add out to 
        for idf,seq,tar in out:
            identifiers.append(idf)
            sequences.append(seq)
            targets.append(tar)
            
    return np.array(identifiers),np.array(sequences),np.array(targets).reshape(-1,2)
identifiers,sequences,targets=procces_sequnces(train)
print(' identifiers: {} \n sequences: {} \n targets: {}'.format(identifiers.shape,sequences.shape,targets.shape))
# Visualize sequences
def visualize_seq():
    idx=np.random.randint(low=0,high=identifiers.shape[0])
    # pick a random subset of features
    feat_idx=np.random.choice(range(len(features)),size=20,replace=True)
    #
    plt.figure(figsize=(16,5))
    plt.pcolor(sequences[idx][:,feat_idx],edgecolors='k', linewidths=1)
    plt.xticks(ticks=.5+np.arange(20),labels=features[feat_idx],rotation=85)        
    plt.yticks(ticks=.5+np.arange(sequences.shape[1]),labels=identifiers[idx][1])
    #
    plt.colorbar()
    plt.title(identifiers[idx][0],fontsize=20,style='italic')
    plt.show()
    
visualize_seq()    
# Inputs
X_train,X_test,y_train,y_test=train_test_split(sequences,targets,test_size=.15,random_state=42)

print('X_train.shape: ',X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train_Conf.shape: ',y_train.shape)
print('y_test_Conf.shape: ',y_test.shape)
model=Sequential()
model.add(LSTM(128,input_shape=X_train.shape[1:],dropout=.1,return_sequences=True))
model.add(Dropout(.2))
model.add(BatchNormalization())

###
model.add(LSTM(128,input_shape=X_train.shape[1:],dropout=.2,return_sequences=True,activation='tanh'))
model.add(BatchNormalization())

###
model.add(LSTM(128,input_shape=X_train.shape[1:],dropout=.15,return_sequences=True,activation='tanh'))
model.add(BatchNormalization())
###




model.add(LSTM(128,input_shape=X_train.shape[1:]))
model.add(Dropout(.2))
model.add(BatchNormalization())


model.add(Dense(128,activation="relu"),)
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(Dense(64,activation="relu"),)
model.add(Dropout(.2))
### 
model.add(Dense(2))
opt=tf.keras.optimizers.Adam(lr=.002,decay=1e-6)
model.summary()
BATCH_SIZE=128
EPOCHS=70
NAME=f'{datetime.datetime.now()}.h5'
NAME
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
model.compile(loss=root_mean_squared_error,
             optimizer=opt,
             metrics=['mae'])
history=model.fit(X_train,y_train,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 validation_data=[X_test,y_test])
history_dict=history.history
history_dict.keys()
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(np.arange(1,EPOCHS+1),history_dict['loss'],color='b',label='train')
plt.plot(np.arange(1,EPOCHS+1),history_dict['val_loss'],color='r',label='validation')
plt.title('loss')
plt.legend()

###
plt.subplot(122)
plt.plot(np.arange(1,EPOCHS+1),history_dict['mae'],color='b',label='train')
plt.plot(np.arange(1,EPOCHS+1),history_dict['val_mae'],color='r',label='validation')
plt.title('accuracy')


plt.legend()
plt.show()
sorted(list(set(test.index) & set(train.index)))
train.reset_index(inplace=True)
test.reset_index(inplace=True)
#
train.head()
print(test.shape)
test=test.merge(train[['Date','Province_State','ConfirmedCases','Fatalities']],on=['Date','Province_State'],how='left')
test.head()
set(train.Date.unique()) & set(test.Date.unique())
(test.shape[0]-test[['ConfirmedCases','Fatalities']].isnull().sum())/test.Province_State.nunique()
n=10
date='2020-04-18'

def delta(date,n):
    return (datetime.datetime.strptime(date,'%Y-%m-%d')-relativedelta(days= n)).strftime('%Y-%m-%d')
delta(date,n)
date_list=np.sort(test.loc[test.ConfirmedCases.isnull(),'Date'].unique())
date_list
def make_predictions(test,n=10):
    out=test.copy()
    # remaining days (with empty targets)
    date_list=np.sort(test.loc[test.ConfirmedCases.isnull(),'Date'].unique())
    
    for date in date_list:
        # pick previuous data -> prediction
        print(out[['ConfirmedCases','Fatalities']].isnull().sum())
        
        test_data=out[(out['Date']<date) & (out['Date']>=delta(date,n))].copy()
        
        print('date min: {} ; date max: {}'.format(test_data.Date.min(),test_data.Date.max()))
        print(test_data['Province_State'].value_counts().unique(),test_data.shape)
        print(test_data[['ConfirmedCases','Fatalities']].isnull().sum())
        
        # create features
        test_data=create_features(test_data)
        # log-transform highly skewed features
        test_data[features_to_transform]=np.log(test_data[features_to_transform]+1)
        # make prediction
        predictions=test_data.groupby(['Province_State']).apply(lambda x:model.predict(x[features].values.reshape(-1,n,len(features)))[0])
        
        # impute predictions
        print(out[['ConfirmedCases','Fatalities']].isnull().sum())
        out.loc[out['Date']==date,['ConfirmedCases','Fatalities']]=predictions.apply(lambda x:pd.Series([x[0],x[1]])).values
        print(out[['ConfirmedCases','Fatalities']].isnull().sum())
    return out
out=make_predictions(test)
out.head()
out[['ForecastId','ConfirmedCases','Fatalities']].isnull().sum()
# apply exp to target (reverse transformation)
out['ConfirmedCases']=np.exp(out['ConfirmedCases'])-1
out['Fatalities']=np.exp(out['Fatalities'])-1
out[['ForecastId','ConfirmedCases','Fatalities']].isnull().sum()
out[['ForecastId','ConfirmedCases','Fatalities']].describe()
out['Fatalities']=np.where(out['Fatalities']>=0,out['Fatalities'],0)
# sumbit predictions
out[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)

def demo():
    idx=np.random.randint(low=0,high=train['Province_State'].nunique()-1)
    #
    idfs=[]
    pred=[]
    true=[]
    #
    for i in range(16):
        pred.append(np.exp(model.predict(sequences[idx+(i*train['Province_State'].nunique())].reshape(-1,10,len(features)))[0])-1)
        true.append(np.exp(targets[idx+(i*train['Province_State'].nunique())])-1)
        #
        idf=identifiers[idx+(i*train['Province_State'].nunique())]
        idfs.append([idf[0],(datetime.datetime.strptime(idf[1][-1],'%Y-%m-%d')+relativedelta(days= 1)).strftime('%Y-%m-%d')])
    #
    idfs,pred,true=np.array(idfs),np.array(pred),np.array(true)
    #
    df_demo=pd.DataFrame({'Province_State':idfs[:,0],
                          'ConfirmedCases_true':true[:,0],
                          'ConfirmedCases_pred':pred[:,0],
                          'Fatalities_true':true[:,1],
                          'Fatalities_pred':pred[:,1]},index=idfs[:,1])
    #
    plt.style.use('default')
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    df_demo['ConfirmedCases_true'].plot(kind='area',color='lightskyblue',alpha=.4,label='True')
    df_demo['ConfirmedCases_pred'].plot(kind='area',color='orangered',alpha=.4,label='Prediction')
    plt.legend()
    plt.xticks(rotation=80)
    plt.title('ConfirmedCases',fontsize=10)
    #
    plt.subplot(122)
    df_demo['Fatalities_true'].plot(kind='area',color='lightskyblue',alpha=.4,label='True')
    df_demo['Fatalities_pred'].plot(kind='area',color='orangered',alpha=.4,label='Prediction')
    plt.xticks(rotation=80)
    plt.title('Fatalities',fontsize=10)
    plt.legend()
    #
    plt.suptitle(idfs[0,0],fontsize=14)
    plt.show()
demo()
