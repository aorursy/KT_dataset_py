# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt #data visualization

import seaborn as sns



from datetime import datetime # date type



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')  # training dataframe

test = pd.read_csv('../input/test.csv')  # testing dataframe
print("train.csv. Shape: ",train.shape)

print("test.csv. Shape: ",test.shape)
pd.DataFrame({'null_count' : train.isnull().sum()})
pd.DataFrame({'null_count' : test.isnull().sum()})
train.head()
train['price'].describe()
f, ax = plt.subplots(figsize = (8,6))

sns.distplot(train['price'])

print("%s -> Skewness: %f, Kurtosis: %f" %  ('price',train['price'].skew(), 

                                                     train['price'].kurt()))
#id는 예측하는데 필요 없으므로 삭제하겠습니다.

train_id = train['id']

del train['id']
#상관관계 확인

k=20 #히트맵 변수 갯수

corrmat = train.corr() #변수간의 상관관계

cols = corrmat.nlargest(k, 'price')['price'].index #price기준으로 제일 큰순서대로 20개를 뽑아냄

cm = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(data = cm, annot=True, square=True, fmt = '.2f', linewidths=.5, cmap='Reds', 

            yticklabels = cols.values, xticklabels = cols.values)
train['date'].describe()
f, ax = plt.subplots(3, 2, figsize = (20,20))



columns = ['sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15','sqft_above', 'sqft_basement']



count = 0

for row in range(3):

    for col in range(2):

        sns.distplot(train[columns[count]], ax=ax[row][col])

        print("%s -> Skewness: %f, Kurtosis: %f" %  (columns[count],train[columns[count]].skew(), 

                                                     train[columns[count]].kurt()))

        count += 1
train[train['sqft_living']>=6000]
f, ax = plt.subplots(figsize=(8,6))

sns.regplot(x='sqft_living', y='sqft_living15', data=train[['sqft_living','sqft_living15']])
# sqft_living15에 비해 상당히 주거공간이 큰경우는 대저택일 것같습니다.

# 이런 대저택들이 대부분의 주택들의 가격을 대변할 수 있을까...?

train[(train['sqft_living']>10000) ]
f, ax = plt.subplots(1, 2, figsize=(20,6))

sns.regplot(x='sqft_living', y='price', data=train[['sqft_living','price']], ax=ax[0])

sns.regplot(x='sqft_living15', y='price', data=train[['sqft_living15','price']], ax=ax[1])
train[train['sqft_living']>7500].sort_values('sqft_living', ascending = False)
train[train['sqft_lot']>=900000]
f, ax = plt.subplots(1, 3, figsize=(20,6))

sns.regplot(x='sqft_lot', y='price', data=train[['sqft_lot','price']], ax=ax[0])

sns.regplot(x='sqft_lot15', y='price', data=train[['sqft_lot15','price']], ax=ax[1])

sns.regplot(x='sqft_lot', y='sqft_lot15', data=train[['sqft_lot','sqft_lot15']], ax=ax[2])
train[train['price']>4000000].sort_values(['sqft_lot','sqft_lot15'])
len(train[train['sqft_living']!=(train['sqft_above']+train['sqft_basement'])])
base_zero = len(train[train['sqft_basement']==0])

base_zero_not = len(train[train['sqft_basement']!=0])

total_base = len(train['sqft_basement'])



basement_count = pd.Series([base_zero, base_zero_not])

basement_pnt = basement_count/total_base*100

basement = pd.concat([basement_count, basement_pnt], axis=1, keys = ['Count', '%'])

basement.index = ['0의값', '0을 제외한 값']

basement
f, ax = plt.subplots(figsize=(8,6))

sns.regplot(x='sqft_basement', y='price', data=train[train['sqft_basement']!=0][['sqft_basement','price']])
train[train['price']>4000000]
data = pd.concat([train['price'], train['grade']], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x='grade', y='price', data=data)
train[(train['grade'] == 11) & (train['price']>3000000)]
train[(train['price']>7000000)].sort_values('grade')
train[(train['grade']== 13)].sort_values('price')
f, ax = plt.subplots(figsize = (10,6))

sns.boxplot(x='bathrooms', y='price', data=train[['bathrooms', 'price']])
train[(train['bathrooms'] == 4.5) & (train['price']>7000000)]
train[(train['bathrooms']>=6.25) & (train['bathrooms']<=7.5)].sort_values('bathrooms')
train[(train['grade'] == 13) & (train['price']>= 2000000)].sort_values('price')
train[(train['sqft_living']>=7000) & (train['grade'] >= 11)].sort_values('price')
train[(train['sqft_living']>=3500) & (train['grade'] == 7)].sort_values('price')
f, ax = plt.subplots(figsize = (8,6))

sns.boxplot(x='bedrooms', y='price', data=train[['bedrooms', 'price']])
train[((train['bedrooms']==5) | (train['bedrooms'] == 6)) & (train['price']>=4000000)]
f, ax = plt.subplots(figsize=(20,6))

sns.boxplot(x = train['zipcode'], y= train['price'])
train = train.drop((train[train['sqft_living']>13000]).index, axis=0)

train = train.drop((train[train['grade']== 3]).index, axis=0)

train = train.drop((train[(train['bathrooms']== 6.75)]).index, axis=0)

train = train.drop((train[(train['bathrooms']== 7.5)]).index, axis=0)
train_test_data = [train, test]
for dataset in train_test_data:

    #date -> 년, 월, 일 단위로 새로운 칼럼 만듦

    dataset['year'] = dataset['date'].str[:4]

    dataset['year'] = dataset['year'].astype(int)

    dataset['month'] = dataset['date'].str[4:6]

    dataset['month'] = dataset['month'].astype(int)

    dataset['day'] = dataset['date'].str[6:8]

    dataset['day'] = dataset['day'].astype(int)

    

    # 부지 면적 대비 실제 사용 면적 비율

    dataset['sqft_ratio'] = dataset['sqft_living'] / dataset['sqft_lot']

        

    # 15개의 부지 면적 대비 실제 사용 면적 평균 비율

    dataset['sqft_ratio15'] = dataset['sqft_living15'] / dataset['sqft_lot15'] 

    

    #지하실 유무

    dataset['is_basement'] = dataset['sqft_basement'].map(lambda x: 1 if x != 0 else 0)

    

    #재건축 여부

    dataset['is_renovated'] = dataset['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)

    

    #재건축 년도 수정

    dataset['yr_renovated'] = dataset['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    dataset['yr_renovated'] = dataset['yr_renovated'].fillna(dataset['yr_built'])



    #구매했을 때 집의 년식 (리모델링한경우는 리모델링 한 날 부터)

    dataset['age'] = dataset['year']-dataset['yr_renovated']

    

    #구매당시 집을 지었는지 여부

    dataset['new_built'] = dataset['year']-dataset['yr_built']

    dataset['new_built'] = dataset['new_built'].apply(lambda x : 1 if x==0 else 0)

    

    #year는 데이터가 두개라서 0과 1로 바꿔줌

    dataset['year'] = dataset['year'].apply(lambda x : 0 if x == 2014 else 1)
#zipcode 수정

train_zipcode = train[['zipcode','price']].groupby('zipcode', as_index = False).mean().sort_values('price')

train_zipcode.head()
zipcode_num = {}

for i in range(0, 70):

    zipcode = train_zipcode['zipcode'].iloc[i]

    zipcode_num[zipcode] = i

    

zipcode_num
train_test_data = [train, test]
for dataset in train_test_data:

    dataset['zipcode_num'] = dataset['zipcode'].map(zipcode_num)
train.head()
#above는 삭제 시 성능 더 좋아짐

drop_columns = ['date', 'sqft_above']

train = train.drop(drop_columns, axis = 1)

test = test.drop(drop_columns, axis = 1)

train.head()
#상관관계 확인

k=30 #히트맵 변수 갯수

corrmat = train.corr()

cols = corrmat.nlargest(k, 'price')['price'].index

cm = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(data = cm, annot=True, square=True, fmt = '.2f', linewidths=.5, cmap='Reds', 

            yticklabels = cols.values, xticklabels = cols.values)
train_columns = []

for column in train.columns:

    if train[column].skew() >= 1:

        print("%s -> Skewness: %f, Kurtosis: %f" %  (column,train[column].skew(), 

                                                 train[column].kurt()))

        train_columns.append(column)

    elif train[column].kurt() >= 3:

        print("%s -> Skewness: %f, Kurtosis: %f" %  (column,train[column].skew(), 

                                                 train[column].kurt()))

        train_columns.append(column)
#정규분포모형을 가질 수 있도록 첨도와 왜도를 조정

#조정하는 방법에는 square root, quarter root, log 등이 있다.

#log에서 0의 값이 들어왔을 때 무한으로 가는 것을 방지하도록 1 더해주는 log1p를 사용



# columns = ['sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'sqft_above']



train['price'] = np.log1p(train['price'])

print("%s -> Skewness: %f, Kurtosis: %f" % ('price', train['price'].skew(), train['price'].kurt()))



for column in train_columns[1:]:

    train[column] = np.log1p(train[column])

    test[column] = np.log1p(test[column])

    print("%s -> Skewness: %f, Kurtosis: %f" %  (column,train[column].skew(), 

                                                 train[column].kurt()))
#단순선형회귀모형



from sklearn.linear_model import LinearRegression

import statsmodels.api as sm



train_columns = [c for c in train.columns if c not in ['price']]



model = sm.OLS(train['price'], train[train_columns])

result = model.fit()

print(result.summary())
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state=631))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=631))

forest = RandomForestRegressor(n_estimators = 150, n_jobs = -1, random_state=42)

gboost = GradientBoostingRegressor(n_estimators = 350, learning_rate = 0.1, max_depth = 4, random_state=42)

xgboost = xgb.XGBRegressor(min_child_weight = 8, max_depth = 8, gamma = 0, random_state=42)

lightgbm = lgb.LGBMRegressor(num_leaves = 50, max_depth = -1, min_data_in_leaf = 30, random_state=42)



models = [{'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},

          {'model':lightgbm, 'name':'LightGBM'}, {'model' : lasso, 'name' : 'LASSO Regression'}, 

          {'model' : ENet, 'name' : 'Elastic Net Regression'}, {'model' : forest, 'name' : 'RandomForset'}]
target = train['price']

del train['price']
#cross validation score

n_folds = 5



def cv_score(models):

    kfold = KFold(n_splits=n_folds, shuffle=True ,random_state=42).get_n_splits(train.values)

    for m in models:

        print("Model {} CV score : {:.4f}".format(m['name'], 

                                                  np.mean(cross_val_score(m['model'], 

                                                                          train.values, target, cv=kfold))))
cv_score(models)
#x.values 는 배열로 데이터를 뽑아옴

#3개의 모델로 만들어진 predict 데이터들의 평균을 구한다.



models = [{'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},

          {'model':lightgbm, 'name':'LightGBM'}]



def AveragingBlending(models, x, y, sub_x):

    for m in models : 

        m['model'].fit(x.values, y)

    

    predictions = np.column_stack([m['model'].predict(sub_x.values) for m in models])

    return np.mean(predictions, axis=1)
test_id = test['id']

del test['id']
y_pred = AveragingBlending(models, train, target, test)
sub = pd.DataFrame(data={'id':test_id,'price':np.expm1(y_pred)})
sub.head()
sub.to_csv('submission.csv', index=False)