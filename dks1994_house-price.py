# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
house_price = pd.read_csv('../input/train.csv')
#Total number of features given to us

print(len(house_price.columns))
#Total number of data instances

print(len(house_price))
#What are the data type of each feature

#print(house_price.dtypes)
#Getting detailed description for each feature

house_price.describe()
#importing graph and visualisation libraries

import matplotlib.pyplot as plt

import seaborn as sns
#Distribution of SalePrice

%matplotlib inline

plt.scatter(range(house_price.shape[0]),np.sort(house_price['SalePrice']))
#Distribution of SalePrice

sns.distplot(house_price['SalePrice'],kde=True)
#Distribution is skewed towards left

#Distribution for Log(SalePrice)

sns.distplot(np.log(house_price['SalePrice']))
#Checking the mean price of house built in particular year

plt.subplots(figsize=(18,10))

sns.barplot(house_price['YearBuilt'],house_price['SalePrice'])

plt.xticks(rotation='vertical')

plt.show()
#We can see a trend upwards in the long run. But it can be inflation or due to number of houses built

#particular year. Not a great insight
#Count of each type of feature

dftypes_df = house_price.dtypes.reset_index()

dftypes_df.columns = ['count','column_type']

dftypes_df = dftypes_df.groupby('column_type').aggregate('count').reset_index()

dftypes_df
#Plotting the number of missing value for each feature

missing_type = house_price.isnull().sum().reset_index()

missing_type.columns = ['Feature_Name','Number of Missing Value']
plt.subplots(figsize=(18,18))

plt.ylim(0,1500)

sns.barplot(missing_type['Feature_Name'],missing_type['Number of Missing Value'])

plt.xticks(rotation='vertical')

plt.ylabel('Number of missing value')

plt.xlabel('Features')

plt.show()

#Plot above gives us an idea about which features could be dropped
#Plot the correlation matrix for given data

corrmat = house_price.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,square=True)

plt.show()
# Using xgboost to find feature importance

import xgboost as xgb

from sklearn import preprocessing,model_selection
for f in house_price.columns:

  if house_price[f].dtype == 'object':

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(house_price[f].values))

    house_price[f] = lbl.transform(list(house_price[f].values))
train_y = house_price.SalePrice.values

train_X = house_price.drop(['SalePrice','Id'],axis=1)
xgb_params = {

    'eta':0.05,

    'max_depth':8,

    'subsample':0.7,

    'colsample_bytree':0.7,

    'objective':'reg:linear',

    'eval_metric':'rmse',

    'silent':1

}



dtrain = xgb.DMatrix(train_X,train_y,feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=100)



fig,ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model,max_num_features=50,height=0.8,ax=ax)

plt.show()
house_price_grouped = house_price.groupby('YearBuilt').count()
house_price_grouped['Id'].tail()
plt.subplots(figsize=(15,10))

plt.axis([1870,2012,0,70])

plt.plot(house_price_grouped['Id'])

plt.show()
#Number of houses sold in early 2000's is exceptionally much higher than previous year or decaded.  

# 2008 recession anyone???