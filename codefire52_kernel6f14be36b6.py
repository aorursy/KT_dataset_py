import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sample=pd.read_csv('../input/sampleSubmission.csv')

train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')
sample.head()
train_df.head()
train_df.info()
train_df.nunique()
test_df.nunique()
train_df.isnull().sum()
train_df.describe()
season_df=train_df.groupby('season')
season_df.head()
cat_feat=['season','weather','holiday','workingday']

cont_feat=['temp','atemp','humidity','windspeed']

count_feats=['casual','registered','count']
hour=[]

day=[]

month=[]

year=[]

for row in train_df['datetime']:

    date_hour=row.split()

    date=date_hour[0]

    hour_row=date_hour[1]

    hour.append(hour_row.split(':')[0])

    date=date.split('-')

    day.append(date[2])

    month.append(date[1])

    year.append(date[0])
train_df['hour']=hour

train_df['day']=day

train_df['month']=month

train_df['year']=year
train_df.head()
test_df.head()
hour=[]

day=[]

month=[]

year=[]

for row in test_df['datetime']:

    date_hour=row.split()

    date=date_hour[0]

    hour_row=date_hour[1]

    hour.append(hour_row.split(':')[0])

    date=date.split('-')

    day.append(date[2])

    month.append(date[1])

    year.append(date[0])

test_df['hour']=hour

test_df['day']=day

test_df['month']=month

test_df['year']=year
datetime=['hour','day','month','year']

for time in datetime:

    train_df[time]=train_df[time].astype(int)

    test_df[time]=test_df[time].astype(int)
#Continous Features Analysis

for i in range(len(cont_feat)-1):

    for j in range(i+1,len(cont_feat)):

        sns.jointplot(cont_feat[i],cont_feat[j],data=train_df)

        plt.title('{} relation with {}'.format(cont_feat[i],cont_feat[j]))

        plt.show()

        
#Categorical feature analysis

for cat in cat_feat:

    sns.barplot(x=cat,y='count',data=train_df,estimator=sum)

    plt.title('{} vs total_rent'.format(cat))

    plt.show()
climate=['temp','humidity','windspeed']

for clim in climate:

    sns.swarmplot(x='hour',y=clim,hue='season',data=train_df)

    plt.title('{} vs {}'.format('hour',clim))

    plt.show()
sns.distplot(train_df['count'])

train_df['count']=train_df['count'].apply(lambda x:np.log(x))
sns.heatmap(train_df.corr())
train_df.info()
train_df=pd.DataFrame(train_df)
train_df.info()
train_df.head()
train_df.set_index('datetime',inplace=True)
test_df.set_index('datetime',inplace=True)
train_df.drop(columns=['casual','registered'],axis=1,inplace=True)
train_df.info()
weather_df=pd.get_dummies(train_df['weather'],prefix='weather')

yr_df=pd.get_dummies(train_df['year'],prefix='year')

month_df=pd.get_dummies(train_df['month'],prefix='month')

hour_df=pd.get_dummies(train_df['hour'],prefix='hour')

season_df=pd.get_dummies(train_df['season'],prefix='season')

train_df=train_df.join(weather_df)

train_df=train_df.join(yr_df)

train_df=train_df.join(month_df)                     

train_df=train_df.join(hour_df)

train_df=train_df.join(season_df)

                     

weather_df=pd.get_dummies(test_df['weather'],prefix='weather')

yr_df=pd.get_dummies(test_df['year'],prefix='year')

month_df=pd.get_dummies(test_df['month'],prefix='month')

hour_df=pd.get_dummies(test_df['hour'],prefix='hour')

season_df=pd.get_dummies(test_df['season'],prefix='season')

test_df=test_df.join(weather_df)

test_df=test_df.join(yr_df)

test_df=test_df.join(month_df)                     

test_df=test_df.join(hour_df)

test_df=test_df.join(season_df)
train_df.head()
train_df.drop(columns=['season','hour','month','year','weather'],axis=1,inplace=True)

test_df.drop(columns=['season','hour','month','year','weather'],axis=1,inplace=True)
def rmlse(predicted,actual):

    sum_val=0

    for i in range(len(predicted)):

        sum_val+=(np.log(predicted[i]+1)-np.log(actual[i]+1))**2

    return (sum_val/len(predicted))**(0.5)
X=train_df.drop(columns='count',axis=1)

y=train_df['count']
X.info()
y.shape
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

'''

param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

'''

'''

param_test2={

    'gamma':[0,0.125,0.25,0.5,0.75,1]

}

'''

'''

param_test3={

    'min_child_weight':[1,2,3,4,5,6,7,8,9]

}

'''

'''

param_test4={

    'learning_rate':[0.1,0.01,0.001]

}

'''

param_test5={

    'subsample':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.125, n_estimators=1000, max_depth=9,

 min_child_weight=4, gamma=0.125, subsample=0.8, colsample_bytree=0.8,

 random_state=42), 

param_grid = param_test5,n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train,y_train)

predicted=gsearch1.predict(X_test)

print('Model Score: {}'.format(rmlse(np.exp(predicted),np.exp(y_test))))

print(gsearch1.best_params_)
y_test
import xgboost as xgb

xgr=xgb.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=9,

 min_child_weight=4, gamma=0.125, subsample=1, colsample_bytree=0.8)

xgr.fit(train_df.drop(columns='count',axis=1),train_df['count'])

y_predict=xgr.predict(test_df)
test_df['count']=np.exp(y_predict)
test_df=test_df.reset_index()
result=pd.DataFrame()
result['datetime']=test_df['datetime']

result['count']=test_df['count']
result.head()
result
result.to_csv('output.csv',index=False)