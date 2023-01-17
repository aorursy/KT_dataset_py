# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/restaurant-revenue-prediction-data/train.csv/train.csv')

print(train_data.head(10))

test_data=pd.read_csv('/kaggle/input/restaurant-revenue-prediction-data/test.csv/test.csv')

# print(test_data.head(10))

sample_sub=pd.read_csv('/kaggle/input/restaurant-revenue-prediction-data/sampleSubmission.csv')

print(sample_sub.head(10))

print(train_data.shape)

print(test_data.shape)
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import SGDRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
train_data.isnull().any()
train_data.hist(figsize=(12,12))
test_data.hist(figsize=(12,12))
x=test_data['Id']

train_data.drop('Id',axis=1,inplace=True)

train_data.head(10)
df = pd.concat([train_data,test_data],axis=0)

df['date']=df['Open Date']

df['Open Date']=pd.to_datetime(df.date)

df['year']=pd.DatetimeIndex(df.date).year

df['month']=pd.DatetimeIndex(df.date).month

df.drop(['Id','date','Open Date'],axis=1,inplace=True)

target='revenue'
df.head(5)
sns.countplot(df['month'])
plt.figure(figsize=(12,12))

sns.countplot(df['year'])
df['Type'] = df['Type'].map({'FC':0,'IL':1,'DT':2,'MB':3})



from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

df['City Group'] = encoder.fit_transform(np.array(df['City Group']).reshape(-1,1))

df['City Group'] = df['City Group'].apply(int)
df.head(10)
df.drop('City',axis=1,inplace=True)

df.shape
df.groupby('year')['revenue'].mean()
df.groupby('month')['revenue'].mean()
train_data = df.dropna(axis=0)

test_data=df[137:].drop('revenue',axis=1)

test_data

print(train_data.shape)

print(test_data.shape)

train_data


regressors={'random forest':RandomForestRegressor(n_estimators=50,max_depth=5,random_state=0,max_features=0.5,verbose=True),

            'k nearest neighbors':KNeighborsRegressor(),

            'svr':SVR(verbose=True),

            'decision tree':DecisionTreeRegressor(random_state=0,max_depth=5),

            'elastic':ElasticNet(random_state=0),

            'lasso':LassoCV(alphas = [1, 0.1, 0.001, 0.0005], verbose=True, random_state=0),

            'xgb reg':XGBRegressor(random_state=0),

            'catboost':CatBoostRegressor(verbose=0, random_state=0),

            'gradient boost':GradientBoostingRegressor(random_state=0,n_estimators=100)

    }
X=train_data.drop('revenue',axis=1)

y=train_data['revenue']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

y
for name,func in regressors.items():

    func.fit(X_train,y_train)

    pred=func.predict(X_test)

    print(name)

    print('r2_score: ',r2_score(y_test,pred))

    print('mae: ',mean_absolute_error(y_test,pred))

    print('mse: ',mean_squared_error(y_test,pred)/10e+10)
test_data1=pd.read_csv('/kaggle/input/restaurant-revenue-prediction-data/test.csv/test.csv')
gbr=RandomForestRegressor(n_estimators=50,max_depth=5,random_state=0,max_features=0.5,verbose=True)

gbr.fit(X,y)

final_pred=gbr.predict(test_data)

sub1=pd.DataFrame({'Id':x,'Prediction':final_pred})

sub1.to_csv('submission_els.csv',index=False)

print('Done')

sub1

score=cross_val_score(gbr,X,y,cv=8)

score.mean()
sub1