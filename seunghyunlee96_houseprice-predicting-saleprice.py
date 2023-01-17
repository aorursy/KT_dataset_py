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
pd.options.display.max_columns=300

pd.options.display.max_rows=100
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv.gz')

train.head()
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

test.head()
alldata = pd.concat([train,test], sort= False)

alldata.head()
alldata.dtypes
len(alldata.columns)
alldata.corr()['SalePrice'].sort_values(ascending=False).head(50)
alldata['YearRemodAdd'].unique()
alldata.corr()['SalePrice'].sort_values().head(50)
alldata.columns[alldata.dtypes==object]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for i in alldata.columns[alldata.dtypes == object] : 

    alldata[i] = le.fit_transform(list(alldata[i]))
alldata.head()
alldata.head()
train.columns[train.dtypes!=object] 
train[train.columns[train.dtypes!=object]].head()
alldata['above_basement_area'] = alldata['GrLivArea'] + alldata['TotalBsmtSF'] # 랜덤 포레스트로 도움이 되지 않는다. 15929.26170
alldata['total_and_garage'] = alldata['GarageArea'] + alldata['above_basement_area'] # 15303.90315
#alldata = alldata.drop(['above_basement_area'],axis=1) # 15564.43625
alldata[:][['YearBuilt','YearRemodAdd','MoSold','YrSold']].head()
# alldata['interval_remod_built'] = alldata['YearRemodAdd'] - alldata['YearBuilt'] # 15399.22275
# alldata['interval_sold_built'] = alldata['YrSold'] - alldata['YearBuilt'] # 15446.76499
# both : 15410.28201

# Got worse.
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,10))

sns.scatterplot(alldata['GrLivArea'],alldata['SalePrice'])
plt.figure(figsize=(15,10))

sns.scatterplot(train['GrLivArea'],train['SalePrice'],hue=train['GarageQual'])
train.loc[(train['GarageQual']=='Ex')]
test.loc[(test['GarageQual']=='Ex')]
plt.figure(figsize=(15,10))

sns.scatterplot(alldata['above_basement_area'],alldata['SalePrice'])
# total_and_garage

plt.figure(figsize=(15,10))

sns.scatterplot(alldata['total_and_garage'],alldata['SalePrice'])
alldata['Neighborhood'].unique()
plt.figure(figsize=(20,12))

sns.boxplot(train['Neighborhood'],train['SalePrice'])
train['Neighborhood'].value_counts() # 1500개의 데이터, 25개의 카테고리인 점을 고려하면 각 변수별로 충분히 들어가 있다고 판단. 
train.shape
plt.figure(figsize=(20,12))

sns.stripplot(train['Neighborhood'],train['SalePrice'])
plt.figure(figsize=(20,12))

sns.swarmplot(train['Neighborhood'],train['SalePrice']) 
plt.figure(figsize=(15,10))

sns.scatterplot(train['GrLivArea'],train['SalePrice'])
alldata2 = alldata.drop(['Id','SalePrice'],axis=1)
alldata2.isnull().any().sum()
alldata2 = alldata2.fillna(-1)
train2 = alldata2[:len(train)]

test2 = alldata2[len(train):]
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=0,n_jobs=-1)

rf.fit(train2,train['SalePrice']) 

result = rf.predict(test2)
importance = pd.Series(rf.feature_importances_, index = train2.columns).sort_values(ascending=False)
importance.sort_values(ascending=False)
sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')

sub.head()
sub['SalePrice'] = result

sub.head() # check whether the result had correctly plugged in
sub.to_csv('submission_tree.csv',index=False)
alldata_r = pd.concat([train,test], sort= False)

alldata_r.head()
alldata_r = alldata_r.fillna(-1)
alldata_r = alldata_r.drop(['Id','SalePrice'],axis=1)
alldata_r2 = pd.get_dummies(alldata_r)

alldata_r2.shape
alldata_r2.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



alldata_r3 = ss.fit_transform(alldata_r2)

alldata_r3
alldata_r3[0] #length is same as number of columns
train_r2 = alldata_r3[:len(train)]

test_r2 = alldata_r3[len(train):]
from sklearn.linear_model import Ridge

rid = Ridge(alpha=150)

rid.fit(train_r2,train['SalePrice'])
result2 = rid.predict(test_r2)
sub2 = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')

sub2.head()
sub2['SalePrice'] = result2

sub2.head() # check whether the result had correctly plugged in
sub2.to_csv('submission_linear.csv',index=False)
display(sub.head(), sub2.head())
sub3 = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
sub3['SalePrice'] = sub['SalePrice'] * 0.6 + sub2['SalePrice'] * 0.4
sub3.to_csv('ensemble.csv',index=False)