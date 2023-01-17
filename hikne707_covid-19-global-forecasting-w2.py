import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
import numpy as np
from numpy import random
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
## 
import sys


from IPython.display import display
from scipy.stats import skew
import lightgbm as lgb
import os
from xgboost import XGBRegressor
from tqdm import tqdm
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
train.head()
train.info()
catcols=train.select_dtypes(include='object').columns.values.tolist()
catcols.remove('Date')
numcols=train.select_dtypes(include='number').columns.values[1:-1].tolist()
train[catcols].describe().merge(test[catcols].describe(),left_index=True,right_index=True,suffixes=('_train','_test'))
# Missing values
pd.concat([round(100*train[catcols].isnull().sum()/train.shape[0],2).to_frame('train'),
           round(100*test[catcols].isnull().sum()/test.shape[0],2).to_frame('test')],axis=1)
print(f'TRAIN -> date_min= {train["Date"].min()} ; date_max= {train["Date"].max()}')
print(f'TEST -> date_min= {test["Date"].min()} ; date_max= {test["Date"].max()}')
def prepare_features(data):
    # lower Province_State & Country_Region in order to use them to add more information
    data['Province_State']=data['Province_State'].str.lower()
    data['Country_Region']=data['Country_Region'].str.lower()
    
    # Create a new feature = weither the Province_State is known or not
    data['UnkownProvince_State']=data['Province_State'].isnull().astype(int)
    
    # Fill missing Province_State & Country_Region missing values 
    data.fillna({'Province_State':'unknown'},inplace=True)
    data.fillna({'Country_Region':'unknown'},inplace=True)
    
    # Remove non-alpha charachters 
    data['Province_State']=data['Province_State'].apply(lambda x: ''.join([ch for ch in x if ch.isalpha()]))
    data['Country_Region']=data['Country_Region'].apply(lambda x: ''.join([ch for ch in x if ch.isalpha()]))
    
    # Create a new feature = Country_Region frequency
    data['Country_RegionFreq']=data['Country_Region'].map(data['Country_Region'].value_counts(1).to_dict())
    
    return data

catcols.append('UnkownProvince_State')
# clean & transform features
train=prepare_features(train)
test=prepare_features(test)
#
train.head(3)
add_inf=pd.read_csv('/kaggle/input/world-population-and-development-indicators/data.csv')
add_inf.head(3)
# add world population & developement indicators 
train=train.merge(add_inf,left_on='Country_Region',right_on='Country Name',how='left')
test=test.merge(add_inf,left_on='Country_Region',right_on='Country Name',how='left')
# concatenate Province_State & Country_Region as "Province_State" ID
train['Province_State']=train['Country_Region']+'-'+train['Province_State']
test['Province_State']=test['Country_Region']+'-'+test['Province_State']

# reset index with Date
train.index=train['Date']
test.index=test['Date']

# Drop useless columns
train.drop(columns=['Date','Country Name','Country_Region'],inplace=True)
test.drop(columns=['Date','Country Name','Country_Region'],inplace=True)
#
train.head(2)
# Display ConfirmedCases & Fatatilities charts for a random Province state

# Pick one random Province_State
province=np.random.choice(train['Province_State'].unique())
s=train.loc[train['Province_State']==province,['ConfirmedCases','Fatalities']]

plt.style.use('default')
plt.figure(figsize=(8,3))
plt.subplot(121)
s['ConfirmedCases'].plot(kind='area',color='blue',alpha=.4)
plt.xticks(rotation=80)
plt.title('ConfirmedCases',fontsize=10)
#
plt.subplot(122)
s['Fatalities'].plot('area',color='orange',alpha=.4)
plt.xticks(rotation=80)
plt.title('Fatalities',fontsize=10)

plt.suptitle(province.upper(),fontsize=14)
plt.show()
# Log-transform target
train['ConfirmedCases']=np.log(train['ConfirmedCases']+1)
train['Fatalities']=np.log(train['Fatalities']+1)
# Show target (ConfirmedCases, Fatalities)
plt.figure(figsize=(10,3))
plt.subplot(121)
sns.distplot(train['ConfirmedCases'])
plt.title('ConfirmedCases')
#
plt.subplot(122)
train['Fatalities'].plot.hist(bins=100,density=True)
plt.title('Fatalities')
plt.show()
# Missing values
round(100*(train.isnull().sum()/train.shape[0]).to_frame('Nan (%)'),2)
# fill missing values with mean-value
train.fillna(train.mean().to_dict(),inplace=True)
test.fillna(train.mean().to_dict(),inplace=True)
train.head(2)
# cross-correlation between "Added informtaion" and target
corr={'ConfirmedCases':[],'Fatalities':[]}
for col in train.select_dtypes(include='number').columns.values[3:]:
    corr['ConfirmedCases'].append(train[[col,'ConfirmedCases']].corr().values[0,1])
    corr['Fatalities'].append(train[[col,'Fatalities']].corr().values[0,1]) 
corr=pd.DataFrame(corr,index=train.select_dtypes(include='number').columns.values[3:])
corr
# Display added feature distributions
plt.figure(figsize=(18,22))
for i,col in enumerate(train.select_dtypes(include='number').columns.values[4:],start=1):
    plt.subplot(9,3,i)
    sns.distplot(train[col],label='train',color='blue')
    sns.distplot(test[col],label='test',color='red') 
plt.tight_layout()
plt.show()
train.head(2)
cols_before=test.select_dtypes(include='number').columns.values[1:]
print(cols_before.shape)
cols_before
def shifting_features(df):
    data=df.copy()
    for step in [1,2,3,5,12]:
        # shiffting columns
        dfu=data.groupby(['Province_State'],as_index=True)['ConfirmedCases','Fatalities'].shift(step)
        dfu.rename(columns={'ConfirmedCases':f'ConfirmedCases_D-{step}','Fatalities':f'Fatalities_D-{step}'},inplace=True)
        data=pd.concat((data,dfu),axis=1)
        # Rolling columns
        #-----# mean columns
        dfu=data.groupby(['Province_State'],as_index=True).rolling(window=step)['ConfirmedCases','Fatalities'].mean()
        dfu.rename(columns={'ConfirmedCases':f'ConfirmedCases_Mean-{step}','Fatalities':f'Fatalities_Mean-{step}'},inplace=True)
        dfu.index=data.index.copy()
        data=pd.concat((data,dfu),axis=1)
    
        #-----# quantile columns
        dfu=data.groupby(['Province_State'],as_index=True).rolling(window=step)['ConfirmedCases','Fatalities'].quantile(.05)
        dfu.rename(columns={'ConfirmedCases':f'ConfirmedCases_Q05-{step}','Fatalities':f'Fatalities_Q05-{step}'},inplace=True)
        dfu.index=data.index.copy()
        data=pd.concat((data,dfu),axis=1)
        
        dfu=data.groupby(['Province_State'],as_index=True).rolling(window=step)['ConfirmedCases','Fatalities'].quantile(.25)
        dfu.rename(columns={'ConfirmedCases':f'ConfirmedCases_Q25-{step}','Fatalities':f'Fatalities_Q25-{step}'},inplace=True)
        dfu.index=data.index.copy()
        data=pd.concat((data,dfu),axis=1)
        
        dfu=data.groupby(['Province_State'],as_index=True).rolling(window=step)['ConfirmedCases','Fatalities'].quantile(.75)
        dfu.rename(columns={'ConfirmedCases':f'ConfirmedCases_Q75-{step}','Fatalities':f'Fatalities_Q75-{step}'},inplace=True)
        dfu.index=data.index.copy()
        data=pd.concat((data,dfu),axis=1)
        
        dfu=data.groupby(['Province_State'],as_index=True).rolling(window=step)['ConfirmedCases','Fatalities'].quantile(.95)
        dfu.rename(columns={'ConfirmedCases':f'ConfirmedCases_Q95-{step}','Fatalities':f'Fatalities_Q95-{step}'},inplace=True)
        dfu.index=data.index.copy()
        data=pd.concat((data,dfu),axis=1)
        
    return data
data=shifting_features(train)
data.head(3)
data.isnull().sum()
data.dropna(axis=0,inplace=True)
print(data.shape)
print(data.index.unique())
data.index.nunique()
print(np.sort(data.index.unique())[-11])

data['TRAIN_SAMPLE']=(data.index<=np.sort(data.index.unique())[-11])
data['TRAIN_SAMPLE'].value_counts(1)
features=data.select_dtypes(include='number').columns.values[3:]
print(features.shape)
features
# Inputs
X_train,X_test=data.loc[data['TRAIN_SAMPLE'],features],data.loc[~data['TRAIN_SAMPLE'],features]

# ConfirmedCases outputs
y_train_Conf,y_test_Conf=data.loc[data['TRAIN_SAMPLE'],'ConfirmedCases'],data.loc[~data['TRAIN_SAMPLE'],'ConfirmedCases']

# Fatalities outputs
y_train_Fat,y_test_Fat=data.loc[data['TRAIN_SAMPLE'],'Fatalities'],data.loc[~data['TRAIN_SAMPLE'],'Fatalities']

print('X_train.shape: ',X_train.shape)
print('X_test.shape: ',X_test.shape)
print('y_train_Conf.shape: ',y_train_Conf.shape)
print('y_test_Conf.shape: ',y_test_Conf.shape)
print('y_train_Fat.shape: ',y_train_Fat.shape)
print('y_test_Fat.shape: ',y_test_Fat.shape)
clf_Conf=XGBRegressor(**{'learning_rate': 0.2, 'max_depth': 5,'n_estimators': 120, 'objective': 'reg:squarederror'})
%time clf_Conf.fit(X_train,y_train_Conf)
print('train score: {}'.format(np.sqrt(metrics.mean_squared_error(y_train_Conf,clf_Conf.predict(X_train)))))
print('test score: {}'.format(np.sqrt(metrics.mean_squared_error(y_test_Conf,clf_Conf.predict(X_test)))))
##

print('train score: {}'.format(metrics.r2_score(y_train_Conf,clf_Conf.predict(X_train))))
print('test score: {}'.format(metrics.r2_score(y_test_Conf,clf_Conf.predict(X_test))))
clf_Fat=XGBRegressor(**{'learning_rate': 0.07, 'max_depth': 5,'n_estimators': 120, 'objective': 'reg:squarederror'})
%time clf_Fat.fit(X_train,y_train_Fat)
print('train score: {}'.format(np.sqrt(metrics.mean_squared_error(y_train_Fat,clf_Fat.predict(X_train)))))
print('test score: {}'.format(np.sqrt(metrics.mean_squared_error(y_test_Fat,clf_Fat.predict(X_test)))))
##

print('train score: {}'.format(metrics.r2_score(y_train_Fat,clf_Fat.predict(X_train))))
print('test score: {}'.format(metrics.r2_score(y_test_Fat,clf_Fat.predict(X_test))))
sorted(list(set(test.index) & set(train.index)))
train.reset_index(inplace=True)
test.reset_index(inplace=True)
#
train.head()
print(test.shape)
test=test.merge(train[['Date','Province_State','ConfirmedCases','Fatalities']],on=['Date','Province_State'],how='left')
test.head()
(test.shape[0]-test[['ConfirmedCases','Fatalities']].isnull().sum())/test.Province_State.nunique()
def make_predictions(test):
    for i in range(test.Date.nunique()-13):
        day_list=np.sort(test.Date.unique())[i:i+14]
        print(len(day_list),day_list)
        test_data=shifting_features(test[np.isin(test['Date'],day_list)])
        test_data=test_data[test_data['Date']==day_list[-2]]
        print(test_data[test_data[features].isnull().sum(axis=1)==0].shape)

        # make predictions
        print(test[['ConfirmedCases','Fatalities']].isnull().sum())
        test.loc[test['Date']==day_list[-1],'ConfirmedCases']=clf_Conf.predict(test_data[features])
        test.loc[test['Date']==day_list[-1],'Fatalities']=clf_Fat.predict(test_data[features])
        print(test[['ConfirmedCases','Fatalities']].isnull().sum())
    return test

test=make_predictions(test)
test.head()
test[['ForecastId','ConfirmedCases','Fatalities']].isnull().sum()
# apply exp to target (reverse transformation)
test['ConfirmedCases']=np.exp(test['ConfirmedCases'])-1
test['Fatalities']=np.exp(test['Fatalities'])-1
# sumbit predictions
test[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)
test[['ForecastId','ConfirmedCases','Fatalities']].head()
