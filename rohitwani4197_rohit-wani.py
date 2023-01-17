import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/into-the-future/train.csv')

train
train.head()
test = pd.read_csv('../input/into-the-future/train.csv')

test
test.head()
train.isnull().sum()
test.isnull().sum()
train.describe()
test.describe()
str(train)
str(test)
train.info()
test.info()
train['time']=pd.to_datetime(train['time'])

train['time']
import seaborn as sns 
sns.boxplot(x='feature_1',data=train)
sns.boxplot(y='feature_2',data=train)
sns.boxplot(y='feature_1',data=test)
sns.jointplot(x='feature_1',y='feature_2',data=train)
uv=np.percentile(train.feature_1,[99])[0]

uv
train[(train.feature_1)>uv]
train.feature_1[(train.feature_1)>3*uv]=3*uv
train[(train.feature_1)>uv]
train.corr()
del train['id']
import statsmodels.api as sm
X=sm.add_constant(train['feature_1'])
X.head()
lm=sm.OLS(train['feature_2'],X).fit()
lm.summary()
from sklearn.linear_model import LinearRegression
X=train['feature_1']
X_1=pd.DataFrame(X)
y=train['feature_2']
y_1=pd.DataFrame(y)
lr=LinearRegression()
lr.fit(X_1,y_1)
print(lr.intercept_)
y_pred=lr.predict(X_1)
##From the graph we can say that one point is leaverage point 



sns.jointplot(x=train['feature_1'],y=train['feature_2'],data=train,kind='reg')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_1,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(y_train.shape)
lr2=LinearRegression()
lr2.fit(X_1,y)
y_test_a=lr2.predict(X_test)
y_train_a=lr2.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_test,y_test_a)
###Higher the value of r2_score better the model 

r2_score(y_train,y_train_a)
y_pred