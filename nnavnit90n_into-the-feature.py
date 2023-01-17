import numpy as np
import pandas as pd
train=pd.read_csv('../input/into-the-future/train.csv')
train.head(10)
test=pd.read_csv('../input/into-the-future/test.csv')
test.head(10)
train.isnull().sum()
test.isnull().sum()
train['time']=pd.to_datetime(train['time'])
train.info()
test['time']=pd.to_datetime(test['time'])
test.info()
train.shape
test.shape
train.describe()
import seaborn as sns
sns.boxplot(x='feature_1',data=train)
# There are outliers in the feature 1 of the train data
sns.boxplot(x='feature_2',data=train)
# There are outliers in the feature_2 in the train data
sns.boxplot(x='feature_1',data=test)
# There are no outlier in the test data set
sns.jointplot(x='feature_1',y='feature_2',data=train)
# Accorfing ti the dependencie we ca see that there is one outlier so it is not oin to affect our data
# treating thte outlier
np.percentile(train.feature_1,[99])
np.percentile(train.feature_1,[99])[0]
uv=np.percentile(train.feature_1,[99])[0]
train[(train.feature_1)>uv]
train.feature_1[(train.feature_1)>3*uv]=3*uv
train[(train.feature_1)>uv]
# Correlation 
train.corr()
del train['id']
train.head()
import statsmodels.api as sm
X=sm.add_constant(train['feature_1'])
X.head()
lm=sm.OLS(train['feature_2'],X).fit()
lm.summary()
from sklearn.linear_model import LinearRegression
X=train['feature_1']
X.head()
X_1=pd.DataFrame(X)
y=train['feature_2']
y_1=pd.DataFrame(y)
y.head()
lr=LinearRegression()
lr.fit(X_1,y_1)
print(lr.intercept_)# Beta nod
print(lr.coef_)
lr.predict(X_1)
sns.jointplot(x=train['feature_1'],y=train['feature_2'],data=train,kind='reg')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_1,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(y_train.shape)
lr2=LinearRegression()
lr2.fit(X_1,y)
#Now let get the predicted value of test set
y_test_a=lr2.predict(X_test)
##Now let get the predicted value of train set
y_train_a=lr2.predict(X_train)
from sklearn.metrics import r2_score

r2_score(y_test,y_test_a)
r2_score(y_train,y_train_a)
