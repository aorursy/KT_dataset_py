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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
%matplotlib inline
data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.shape
features_with_null=[features for features in data.columns if data[features].isnull().sum()>0]
for feature in features_with_null:
    print(feature,'{} Null values'.format(data[feature].isnull().sum()))
dataset=data.copy()
for feature in features_with_null:
    dataset[feature]=np.where(data[feature].isnull(),1,0)
    dataset.groupby(feature)['SalePrice'].median().plot.bar()
    plt.show()
data.head()
categorical_features=[features for features in data.columns if data[features].dtype=='O']
print(len(categorical_features),'categorical features')

numerical_features=[features for features in data.columns if data[features].dtype!='O' and features !='Id']
print(len(numerical_features),'Numerical features')

date_feature=[feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]
date_feature
for feature in date_feature:
    print(len(data[feature].unique()),'Unique value in {}'.format(feature))

data.groupby('YrSold')['SalePrice'].median().plot()
plt.show()
for feature in date_feature:
    dataset=data.copy()
    if feature!='YrSold':
        dataset[feature]=dataset['YrSold']- dataset[feature]
        dataset.groupby(feature)['SalePrice'].median().plot()
        plt.show()
discreate_features=[feature for feature in numerical_features if len(data[feature].unique())<25 and feature!='YrSold']
discreate_features
continuous_features=[]
for feature in numerical_features:
    if feature not in discreate_features:
        if feature not in date_feature:
            continuous_features.append(feature)
continuous_features
dataset=data.copy()
for feature in discreate_features:
    dataset.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.show()
dataset=data.copy()
for feature in continuous_features:
    dataset[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()
dataset=data.copy()
dataset['SalePrice']=np.log(dataset['SalePrice'])
for feature in continuous_features:
    if 0 in dataset[feature].unique():
        pass
    else:
        dataset[feature]=np.log(dataset[feature])
        plt.scatter(dataset[feature],dataset['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Sale price')
        plt.show()
        
    
dataset=data.copy()
for feature in continuous_features:
    plt.scatter(dataset[feature],dataset['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('saleprice')
    plt.show()
from scipy import stats
z=np.abs(stats.zscore(dataset[continuous_features]))
z
print(np.where((z>3)))
print(len(np.where((z>3)[0])))
dataset[continuous_features][(z < 3).all(axis=1)]
dataset.head()

dataset[continuous_features].head()
dataset[continuous_features].shape
dataset.shape


dataset=data.copy()
for feature in continuous_features:
    dataset.boxplot(feature)
    plt.ylabel(feature)
    plt.show()
data[categorical_features].head()
for feature in categorical_features:
    print(feature,'has {} different unique values'.format(len(data[feature].unique())))
dataset=data.copy()
for feature in categorical_features:
    dataset.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.show()
    
for feature in categorical_features:
    print(feature,data[feature].isnull().sum())
data.head()
data.drop(['Alley','PoolQC','Fence','MiscFeature'],inplace=True,axis=1)
data.drop('FireplaceQu',axis=1,inplace=True)
cat_feat_with_null=[feature for feature in data.columns if data[feature].isnull().sum()>0 and data[feature].dtype =='O']
for feature in cat_feat_with_null:
        
        data[feature]=data[feature].fillna('Missing')
print(cat_feat_with_null)
for feature in cat_feat_with_null:
    data.groupby(feature)[feature].count().sort_values(ascending=False).plot.bar()
    plt.show()
for feature in cat_feat_with_null:
    frequent_occuring=data[feature].value_counts().sort_values(ascending=False).index[0]
    data[feature]=data[feature].replace('Missing',frequent_occuring)
    data.groupby(feature)[feature].count().sort_values(ascending=False).plot.bar()
    plt.show()
    
data[cat_feat_with_null].head()
data[cat_feat_with_null].isnull().sum()
data[numerical_features].isnull().sum()
numer_feat_with_null=[feature for feature in numerical_features if data[feature].isnull().sum()>0 and data[feature].dtype!='O']
print(numer_feat_with_null)
for feature in numer_feat_with_null:
    median=data[feature].median()
    data[feature]=data[feature].fillna(median)
data[numer_feat_with_null].isnull().sum()
data.isnull().sum()
transform_feature=[]
for feature in continuous_features:
    if feature not in date_feature and 0 not in data[feature].unique():
        transform_feature.append(feature)
        print(feature)
data.head()
for feature in transform_feature[1:len(transform_feature)]:
    dataset[feature]=np.log(dataset[feature])
dataset.head()
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    dataset[feature]=dataset['YrSold']- dataset[feature]
dataset[date_feature].head()
for feature in transform_feature:
    dataset=data.copy()
    dataset[feature].hist(bins=10)
    plt.xlabel(feature)
    plt.show()
categorical_features=[feature for feature in dataset.columns if data[feature].dtype=='O']
for feature in categorical_features:
    print(data[feature].value_counts())
categorical_features=[feature for feature in dataset.columns if data[feature].dtype=='O']
for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)
dataset.head()
scaled_features=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]
print(scaled_features)
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
fit=mms.fit(dataset[scaled_features])
dataset[scaled_features]=fit.transform(dataset[scaled_features])
dataset[scaled_features].head()
dataset=pd.concat([dataset[scaled_features],dataset['SalePrice']],axis=1)
dataset.head()
dataset.shape
dataset.head()
dataset.drop('YrSold',axis=1,inplace=True)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
x=dataset.iloc[:,:-1]
y=dataset['SalePrice']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
ls=Lasso(alpha=0.005,normalize=True)
ls.fit(x_train,y_train)
y_pred=ls.predict(x_test)
ls.score(x_test,y_test)
ls.get_params()
x_test.shape
feature_sel_model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(x_train,y_train)
feature_sel_model.get_support()
selected_features=x_train.columns[feature_sel_model.get_support()]
selected_features
print('total no of features {}'.format(x_train.shape[1]))
print('no of selected features {}'.format(len(selected_features)))

plt.figure(figsize=(15,9))
from sklearn.linear_model import LinearRegression
lg=LinearRegression()
predictors=x_train.columns 
lg.fit(x_train,y_train)
coef=pd.Series(lg.coef_,predictors).sort_values()
coef.plot.bar()
y_pred=lg.predict(x_test)
mse=np.mean((y_test-y_pred)**2)
mse
lg_score=lg.score(x_test,y_test)
lg_score
from sklearn.linear_model import Ridge
alpha=[0.05,0.005,1,4,5,10,15]
mse=[]
ridge_score=[]
for i in alpha:
    ridge=Ridge(alpha=i,normalize=True)
    ridge.fit(x_train,y_train)
    y_pred=ridge.predict(x_test)
    mse.append(np.mean((y_pred-y_test)**2))
    ridge_score.append(ridge.score(x_test,y_test))
plt.plot(alpha,ridge_score,linestyle='dashed',color='blue',marker='o',markersize='10',markerfacecolor='red')

plt.xlabel('alpha')
plt.ylabel('ridge score')
plt.show()


plt.plot(alpha,mse,linestyle='dashed',color='blue',marker='o',markersize='10',markerfacecolor='red')
plt.xlabel('alpha')
plt.ylabel('mean squared error')
plt.show()

alpha=[0.05,0.005,1,4,5,10,15]
predictors=x_train.columns
for i in alpha:
    plt.figure(figsize=(15,9))
    ridge=Ridge(alpha=i)
    ridge.fit(x_train,y_train)
    ridge_coeff=pd.Series(ridge.coef_,predictors).sort_values()
    ridge_coeff.plot.bar()
    plt.show()
