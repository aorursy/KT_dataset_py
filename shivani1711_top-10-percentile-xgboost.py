import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPRegressor
%matplotlib inline

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_df = pd.read_csv('../input/sampleSubmission.csv')
print(train_df.shape)
print(test_df.shape)
plt.hist(train_df['humidity'],bins=20)
### Checking distribution of windspeed.
plt.hist(train_df['windspeed'],bins=20)
### Normalizing windspeed as it is skewed
def normalize(df,col) :
    df[col] = df[col].apply(lambda x : x+1)
    df[col] = np.log(df[col])
    return df
train_df = normalize(train_df,'windspeed')
test_df = normalize(test_df,'windspeed')    
train_df.corr()['casual']
### Parse dates for useful information
def get_dates(df) :
    #print df
    datetime = df['datetime'].values
    date_arr = []
    time_arr = []
    year = []
    month = []
    d = []
    for item in datetime :
        dates = item.split(" ")
        date = dates[0]
        time = dates[1].split(':')[0]
        ym = date.split('-')
        year.append(int(ym[0]))
        month.append(int(ym[1]))
        d.append(int(ym[2]))
        date_arr.append(date)
        time_arr.append(int(time))
    

    df['date'] = date_arr
    df['time'] = time_arr
    df['year'] = year
    df['month'] = month
    df['day'] = d
    return df

train_df = get_dates(train_df)
test_df = get_dates(test_df)
train_df['day_of_week'] = train_df['date'].apply(lambda x : parse(x).weekday())
test_df['day_of_week'] = test_df['date'].apply(lambda x : parse(x).weekday())
### Drop columns not required

train_df.drop(['datetime','date','day'],axis=1,inplace=True)
test_datetime = test_df['datetime']
test_df.drop(['datetime','date','day'],axis=1,inplace=True)
### Creating new features 
train_df['ftemp'] = train_df['temp'] + train_df['atemp']
test_df['ftemp'] = test_df['temp'] + test_df['atemp']
### rel_temp penalizes temperature above and below normal temperature range of 18-33
train_df['rel_temp'] = 0
train_df.loc[train_df['atemp'] <8,'rel_temp'] = 3
train_df.loc[train_df['atemp'] >43,'rel_temp'] = 3
train_df.loc[(train_df['atemp'] >=34) & (train_df['atemp'] <=38),'rel_temp'] = 1
train_df.loc[(train_df['atemp'] >=13) & (train_df['atemp'] <=18),'rel_temp'] = 1
train_df.loc[(train_df['atemp'] >=39) & (train_df['atemp'] <=43),'rel_temp'] = 2
train_df.loc[(train_df['atemp'] >=8) & (train_df['atemp'] <=12),'rel_temp'] = 2
test_df['rel_temp'] = 0
test_df.loc[train_df['atemp'] <8,'rel_temp'] = 3
test_df.loc[train_df['atemp'] >43,'rel_temp'] = 3
test_df.loc[(train_df['atemp'] >=34) & (test_df['atemp'] <=38),'rel_temp'] = 1
test_df.loc[(train_df['atemp'] >=13) & (test_df['atemp'] <=18),'rel_temp'] = 1
test_df.loc[(train_df['atemp'] >=39) & (test_df['atemp'] <=43),'rel_temp'] = 2
test_df.loc[(train_df['atemp'] >=8) & (test_df['atemp'] <=12),'rel_temp'] = 2
### Drop columns not required
train_df.drop(['temp','atemp','count'],axis=1,inplace=True)
test_df.drop(['temp','atemp'],axis=1,inplace=True)
### Separate train and test data for casual and registered models
train_df_casual_y = train_df['casual']
train_df_casual_x = train_df.drop(['registered','casual'],axis=1)
train_df_registered_y = train_df['registered']
train_df_registered_x = train_df.drop(['registered'],axis=1)
### Hyperparameter tuning for casual model
train_data = xgb.DMatrix(train_df_casual_x,label=train_df_casual_y)
res1 = xgb.cv({'n_estimators':100,'max_depth' : 100,'subsample':0.8,'min_child_weight':3,'gamma':0.3,'eta':0.1,'seed':42},train_data,num_boost_round=100,nfold=3,)
print(res1)
### Model for casual riders using xgboost
reg = XGBRegressor(max_depth=100,subsample=0.8,eta=0.05,gamma=0.3,seed=42,n_estimators=100)
reg.fit(train_df_casual_x,train_df_casual_y)
test_df_casual = reg.predict(test_df)

### Add causal as a feature to guess registered riders
test_df['casual'] = test_df_casual
test_df = test_df[train_df_registered_x.columns.values]

### Parameter tuning for registered model
train_data = xgb.DMatrix(train_df_registered_x,label=train_df_registered_y)
res2 = xgb.cv({'n_estimators':100,'max_depth' : 30,'subsample':0.8,'min_child_weight':3,'gamma':0.1,'eta':0.1,'seed':42},train_data,early_stopping_rounds=10, verbose_eval=True,num_boost_round=500,nfold=3)
print(res2)
### Predicting registered riders using xgboost
reg = XGBRegressor(max_depth=100,subsample=0.8,gamma=0.1,eta=0.1,seed=42,min_child_weight=3,n_estimators=100)
reg.fit(train_df_registered_x,train_df_registered_y)
test_df_registered = reg.predict(test_df)
print(cross_val_score(reg,train_df_casual_x,train_df_casual_y))
### Getting final count data
count = test_df_casual + test_df_registered
for i in range(0,len(count)) :
    if count[i] < 0 :
        count[i] = 0
count = np.floor(count)
### Submit file in required format
submission = pd.DataFrame({
        "datetime": test_datetime,
        "count": count
   
    })
submission = submission[['datetime','count']]
submission.head()
submission.to_csv('submission.csv', index=False)
