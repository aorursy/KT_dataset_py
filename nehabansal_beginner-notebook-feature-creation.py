# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ipywidgets import interact,widgets
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dir_ = "/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/"

train_data = pd.read_csv(f"{dir_}train_sessions.csv")
test_data = pd.read_csv(f"{dir_}test_sessions.csv")
def basic_info(data):
    print(data.shape)
    print(data.isnull().sum())
    print(data.info())
basic_info(train_data)
basic_info(test_data)
site_dic = pd.read_pickle(f"{dir_}site_dic.pkl")
print(min(list(site_dic.values())),max(list(site_dic.values())))
from sklearn.preprocessing import MinMaxScaler
def missing_impute(data):
    data= data.fillna(0)
    return data

def feature_creation(data):
    data['num_sites'] = data[[f"site{i}" for i in range(1,11)]].count(axis=1)
    
    for col in [f"time{i}" for i in range(1,11)]:
        data[col] = pd.to_datetime(data[col])
    
    data['session_strt'] = data[[f"time{i}" for i in range(1,11)]].min(axis=1)
    data['session_end'] = data[[f"time{i}" for i in range(1,11)]].max(axis=1)
    data['session_len'] = (data['session_end'] - data['session_strt']).dt.seconds
    
    data['session_year'] = data['session_strt'].dt.year
    data['session_day'] = data['session_strt'].dt.day
    data['session_month'] = data['session_strt'].dt.month

    data['weekday'] = data['session_strt'].dt.dayofweek
    data['weekend'] = data['weekday'].apply(lambda x: 1 if x>=5 else 0)
    data['workweek'] = data['weekday'].apply(lambda x: 1 if x<5 else 0)
    
    data['unique_sites'] = data[[f'site{i}' for i in range(1,11)]].apply(lambda x: len(set(x)),axis=1)

    data = data.drop(columns=[f'site{i}' for i in range(1,11)] + [f'time{i}' for i in range(1,11)])
    
    return data

def feature_transform(data,cols,scaler):
    if not scaler:
        scaler = MinMaxScaler()
        scaler.fit(data[cols])
        
    scaled_df = pd.DataFrame(scaler.transform(data[cols]),columns =cols)
    for col in cols:
        data[f'{col}_std'] = scaled_df[col]
        
    data = data.drop(columns=cols)
    
    return data,scaler

def dummy_var(data,cols):
    for col in cols:
        data[col] = data[col].astype(str)
    dummy_df = pd.get_dummies(data[cols])
    dummy_df['session_id'] = data['session_id']
    data = data.drop(columns=cols)    
    data = pd.merge(data,dummy_df,on=['session_id'])

    return data
@interact
def eda(col=['num_sites','session_year','session_month','weekday','weekend','workweek'
                           ,'target']):
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    ((train_data_clean[col].value_counts(normalize=True))*100).plot(kind='bar')
    plt.xlabel(col)
    plt.ylabel("%records")
    
    if col!='target':
        tempdf = train_data_clean.groupby([col],as_index=False).agg({'target':['sum','count']})
        tempdf.columns = tempdf.columns.droplevel(0)
        tempdf.rename(columns={'':col},inplace=True)
        tempdf['alice%'] = tempdf['sum']*100/tempdf['count']

        plt.subplot(1,2,2)
        plt.plot(col,"alice%",data=tempdf,marker='o')
        plt.grid()
        plt.xlabel(col)
        plt.ylabel("% Alice Sessions")
        plt.title(col)
train_data_clean = feature_creation(train_data)
test_data_clean = feature_creation(test_data)

train_data_clean,scaler = feature_transform(train_data_clean,['num_sites','session_len'],None)
test_data_clean,scaler = feature_transform(test_data_clean,['num_sites','session_len'],scaler)

train_data_clean = dummy_var(train_data_clean,['session_year','session_month'])
test_data_clean = dummy_var(test_data_clean,['session_year','session_month'])

for col in train_data_clean.columns:
    if col not in test_data_clean.columns :
        test_data_clean[col] = 0

train_data_clean.columns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate,cross_val_score
cols_model  = ['num_sites_std', 'session_len_std','weekend', 'workweek',
               'session_year_2013', 'session_year_2014', 'session_month_1',
               'session_month_10', 'session_month_11', 'session_month_12',
               'session_month_2', 'session_month_3', 'session_month_4',
               'session_month_5', 'session_month_6', 'session_month_7',
               'session_month_8', 'session_month_9']

lr_model = LogisticRegression()
lr_model.fit(train_data_clean[cols_model],train_data_clean['target'])
cv_scores = cross_val_score(lr_model,train_data_clean[cols_model],train_data_clean['target'],scoring='roc_auc',cv=5)
print(cv_scores.mean())

# rf_model = RandomForestClassifier()
# rf_model.fit(train_data_clean[cols_model],train_data_clean['target'])
# cv_scores = cross_val_score(rf_model,train_data_clean[cols_model],train_data_clean['target'],scoring='roc_auc',cv=5)
# print(cv_scores.mean())
submission = test_data_clean[['session_id']]
submission['target'] = lr_model.predict_proba(test_data_clean[cols_model])[:,1]
submission.to_csv("submission2.csv",index=False)
