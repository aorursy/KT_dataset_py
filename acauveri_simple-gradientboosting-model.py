%%time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
%%time
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.shape
df_test.shape
df_train.head(2)
df_test.head(2)
df_train.isnull().sum()
df_test.isnull().sum()


df_train.date  = pd.to_datetime(df_train.date, format='%Y-%m-%d')
df_train['year'] = df_train.date.dt.year
df_train['month']=df_train.date.dt.month
df_train['day']=df_train.date.dt.day
df_train.head(2)

df_test.date  = pd.to_datetime(df_test.date, format='%Y-%m-%d')
df_test['year'] = df_test.date.dt.year
df_test['month']=df_test.date.dt.month
df_test['day']=df_test.date.dt.day
df_test.head(2)
y=pd.DataFrame()
y['sales']=df_train['sales']
df_train=df_train.drop(columns='date',axis=1)
df_train=df_train.drop(columns='sales',axis=1)
df_train.dtypes
df_test=df_test.drop(columns='id',axis=1)

df_test=df_test.drop(columns='date',axis=1)
df_test.dtypes
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
x=df_train

clf=clf.fit(x,y)
%%time
output=clf.predict(df_test)
result=pd.DataFrame(output)
result
test=pd.read_csv('../input/test.csv',usecols=['id'])
fin=pd.DataFrame(test)
fin['sales']=result
fin.to_csv('Sales_lightGBM.csv',index=False)
 