# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler





from sklearn import metrics

from sklearn.linear_model import LinearRegression as lm

from sklearn import tree

from sklearn.tree import DecisionTreeRegressor   

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/course-material-walmart-challenge/train.csv')

test=pd.read_csv('/kaggle/input/course-material-walmart-challenge/test.csv')

sample_submission=pd.read_csv('/kaggle/input/course-material-walmart-challenge/sample_submission.csv')
train.head()
test.head()
train.describe()
test.describe()
train.Weekly_Sales[train.Weekly_Sales < 0].count()
train.drop(train.loc[train['Weekly_Sales']<0].index, inplace=True)
train.isnull().sum()
test.isnull().sum()
train['MarkDown1'] = train['MarkDown1'].replace('?',np.nan).astype(float)

train['MarkDown1'] = train['MarkDown1'].fillna((train['MarkDown1'].median()))

test['MarkDown1'] = test['MarkDown1'].replace('?',np.nan).astype(float)

test['MarkDown1'] = test['MarkDown1'].fillna((train['MarkDown1'].median()))





train['MarkDown2'] = train['MarkDown2'].replace('?',np.nan).astype(float)

train['MarkDown2'] = train['MarkDown2'].fillna((train['MarkDown2'].median()))

test['MarkDown2'] = test['MarkDown2'].replace('?',np.nan).astype(float)

test['MarkDown2'] = test['MarkDown2'].fillna((train['MarkDown2'].median()))



train['MarkDown3'] = train['MarkDown3'].replace('?',np.nan).astype(float)

train['MarkDown3'] = train['MarkDown3'].fillna((train['MarkDown3'].median()))

test['MarkDown3'] = test['MarkDown3'].replace('?',np.nan).astype(float)

test['MarkDown3'] = test['MarkDown3'].fillna((train['MarkDown3'].median()))



train['MarkDown4'] = train['MarkDown4'].replace('?',np.nan).astype(float)

train['MarkDown4'] = train['MarkDown4'].fillna((train['MarkDown4'].median()))

test['MarkDown4'] = test['MarkDown4'].replace('?',np.nan).astype(float)

test['MarkDown4'] = test['MarkDown4'].fillna((train['MarkDown4'].median()))



train['MarkDown5'] = train['MarkDown5'].replace('?',np.nan).astype(float)

train['MarkDown5'] = train['MarkDown5'].fillna((train['MarkDown5'].median()))

test['MarkDown5'] = test['MarkDown5'].replace('?',np.nan).astype(float)

test['MarkDown5'] = test['MarkDown5'].fillna((train['MarkDown5'].median()))
plt.figure(figsize=(10, 10))

sns.heatmap(train.corr(), annot=True)

plt.show()
train.dtypes
test.dtypes
sns.boxplot(train.Weekly_Sales)

plt.show()
print(train.Store.nunique())

print(train.Dept.nunique())

print(train.IsHoliday.nunique())

print(train.Unemployment.nunique())

print(train.Type.nunique())

print(train.Size.nunique())
for cols in ['Store','Dept','Size','IsHoliday','Unemployment','Type']:

  train[cols]=train[cols].astype('category')
for cols in ['Store','Dept','Size','IsHoliday','Unemployment','Type']:

  test[cols]=test[cols].astype('category')
#ID_train= train['Store'].astype(str)+'_'+train['Dept'].astype(str)+'_'+train['Date'].astype(str)

ID = test['Store'].astype(str)+'_'+test['Dept'].astype(str)+'_'+test['Date'].astype(str)
train.head()
test.head()
train.drop(['Date'],axis=1,inplace=True)
test.drop(['Date'],axis=1,inplace=True)
x = train.loc[:,train.columns.difference(['Weekly_Sales'])]

y = pd.DataFrame(train['Weekly_Sales'])
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2, random_state = 345)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
x_train = pd.get_dummies(x_train,prefix=['IsHoliday','Type'], columns = ['IsHoliday','Type'],drop_first=True)

x_val = pd.get_dummies(x_val,prefix=['IsHoliday','Type'], columns = ['IsHoliday','Type'],drop_first=True)

test = pd.get_dummies(test,prefix=['IsHoliday','Type'], columns = ['IsHoliday','Type'],drop_first=True)
x_train.dtypes
for cols in ['Dept','Size','Store','Unemployment']:

  x_train[cols] = x_train[cols].cat.codes



for cols in ['Dept','Size','Store','Unemployment']:

  x_val[cols] = x_val[cols].cat.codes



for cols in ['Dept','Size','Store','Unemployment']:

  test[cols] = test[cols].cat.codes
scaler = MinMaxScaler()

scaler.fit(y_train)

y_train.Weekly_Sales=scaler.transform(y_train)

y_val.Weekly_Sales=scaler.transform(y_val)
y_train.head(2)
x_train.head(2)
x_train.shape,x_val.shape,y_train.shape,y_val.shape
linearmod=lm().fit(x_train,y_train)

linearmod

train_pred_lr=linearmod.predict(x_train)

val_pred_lr=linearmod.predict(x_val)





print('MAE:',metrics.mean_absolute_error((y_train),train_pred_lr))

print('MSE:',metrics.mean_squared_error((y_train),train_pred_lr))

print('RMSE:',np.sqrt(metrics.mean_squared_error((y_train),train_pred_lr)))

print('R2:',metrics.r2_score(y_pred=train_pred_lr,y_true=y_train))



print('MAE:',metrics.mean_absolute_error((y_val),val_pred_lr))

print('MSE:',metrics.mean_squared_error((y_val),val_pred_lr))

print('RMSE:',np.sqrt(metrics.mean_squared_error((y_val),val_pred_lr)))

print('R2:',metrics.r2_score(y_pred=val_pred_lr,y_true=y_val))
clf = tree.DecisionTreeRegressor(random_state=1234, max_depth=10)

clf = clf.fit(x_train, y_train)

train_pred_dt=clf.predict(x_train)

val_pred_dt=clf.predict(x_val)





print('MAE:',metrics.mean_absolute_error((y_train),train_pred_dt))

print('MSE:',metrics.mean_squared_error((y_train),train_pred_dt))

print('RMSE:',np.sqrt(metrics.mean_squared_error((y_train),train_pred_dt)))

print('R2:',metrics.r2_score(y_pred=train_pred_dt,y_true=y_train))



print('MAE:',metrics.mean_absolute_error((y_val),val_pred_dt))

print('MSE:',metrics.mean_squared_error((y_val),val_pred_dt))

print('RMSE:',np.sqrt(metrics.mean_squared_error((y_val),val_pred_dt)))

print('R2:',metrics.r2_score(y_pred=val_pred_dt,y_true=y_val))
rf = RandomForestRegressor(n_estimators = 100, random_state = 40)

rf = rf.fit(x_train, y_train)

train_pred_random=rf.predict(x_train)

val_pred_random=rf.predict(x_val)







print('MAE:',metrics.mean_absolute_error((y_train),train_pred_random))

print('MSE:',metrics.mean_squared_error((y_train),train_pred_random))

print('RMSE:',np.sqrt(metrics.mean_squared_error((y_train),train_pred_random)))

print('R2:',metrics.r2_score(y_pred=train_pred_random,y_true=y_train))



print('MAE:',metrics.mean_absolute_error((y_val),val_pred_random))

print('MSE:',metrics.mean_squared_error((y_val),val_pred_random))

print('RMSE:',np.sqrt(metrics.mean_squared_error((y_val),val_pred_random)))

print('R2:',metrics.r2_score(y_pred=val_pred_random,y_true=y_val))

test_predictions=rf.predict(test)
test_predictions
random_forest=pd.DataFrame({'Weekly_Sales':test_predictions,'ID':ID})

random_forest.head()

#random=pd.DataFrame(data_rf)

#random.to_csv('/kaggle/working/results_on_rf.csv',index=False)
random_forest.to_csv('random_forest.csv',index=False)