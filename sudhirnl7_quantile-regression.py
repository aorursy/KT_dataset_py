import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import statsmodels.api as sm

path = 'covid19-global-forecasting-week-5'

path = '../input/covid19-global-forecasting-week-5'

train = pd.read_csv(path+'/train.csv')

test = pd.read_csv(path+'/test.csv')
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])



drop = ['County','Province_State']

train = train.drop(drop, axis=1)

test = test.drop(drop, axis=1)



test = test.rename(columns= {'ForecastId':'Id'})



train_len = train.shape[0]

df = pd.concat([train,test])
train.shape, test.shape, df.shape
train.isnull().sum()
train.head()
train.describe()
#plt.plot_date(x=train['Date'],y=train['TargetValue'])
test.isnull().sum()/test.shape[0]
test['Date'].describe()
train['Date'].describe()
train.tail()
train['Target'].value_counts()
test['Target'].value_counts()
def date_time_feature(df,col):

    df[col+'_month'] = df['Date'].dt.month

    df[col+'_day'] = df['Date'].dt.day

    df[col+'_week'] = df['Date'].dt.week

    df[col+'_weekofyear'] = df['Date'].dt.weekofyear

    df['Date'] = df['Date'].dt.strftime('%Y%m%d').astype(int)

    return df
col= 'Date'

df = date_time_feature(df,col)
from sklearn.preprocessing import LabelEncoder
def encode(df,col):

    

    le = LabelEncoder()

    for c in col:

        df[c] = le.fit_transform(df[c])

    return df
col =['Country_Region','Target']

df = encode(df, col)

col = ['Country_Region','Date_month','Date_day','Date_week','Date_weekofyear']

df = pd.get_dummies(data=df,columns=col,drop_first=True)

df.head()
df1 = df.drop(['Date','Id','TargetValue'],axis=1)
X_train = df1[:train_len]

X_test = df1[train_len:]

y_train = df.iloc[:train_len]['TargetValue']
X_train = X_train.reset_index(drop = True)

X_test = X_test.reset_index(drop = True)

y_train = y_train.reset_index(drop = True)
from sklearn.model_selection import RandomizedSearchCV
# param = {

#     'learning_rate':np.linspace(0.001,0.5),

#     'n_estimators':np.arange(10,500),

# }
# rsCV  = RandomizedSearchCV(model,param_distributions=param, n_iter=3,n_jobs= -1)

# rsCV.fit(X_train,y_train)
import lightgbm as lgb
def lgb_model(q):

    model = lgb.LGBMRegressor(

        objective = 'quantile',

        alpha = q,

        learning_rate = 0.05,

        n_estimators = 1000,

        min_data_in_leaf=5,

        num_leaves = 100000,

        bagging_fraction=0.95,

        feature_fraction = 0.95,

        max_depth = 10,

        random_state = 12,

        num_threads = -1

    )

    model.fit(X_train,y_train)

    score = model.score(X_train,y_train)

    print(f'quantile {q} score: {round(score,3)}')

    y_pred = model.predict(X_test)

    return y_pred
sub = pd.DataFrame()

sub['Id']  = test.Id

sub['q0.05'] = lgb_model(0.05)

sub['q0.5'] = lgb_model(0.5)

sub['q0.95'] = lgb_model(0.95)
sub=pd.melt(sub, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()