# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
target=train['SalePrice']

print(type(train))
#train.info()
#train.info()

sns.pairplot(train.iloc[:,:3])
#train.info()
#train.describe()
#train['YearBuilt']
train['Age_of_house']=train['YrSold']-train['YearBuilt']
#train['Age_of_house']
sns.pairplot(train.iloc[:,:3])
train=train.drop(['YearBuilt','YrSold'],axis=1)
#for i in train.columns:
#    print(train[i].unique())
#train['YrSold'].unique()
train=train.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley','SalePrice'],axis=1)
#train.shape



#for i,labels in enumerate(train.isnull().sum()):
#    print(i,labels)
#a=[28,29,30,31,33,40,55,56,57,60,61]
#for i in a:
#    print(train.iloc[:,i])

from sklearn.impute import SimpleImputer
si=SimpleImputer(strategy='mean')
train.iloc[:,3:4]=si.fit_transform(train.iloc[:,3:4])

a=train.isnull().sum()
null_columns=[]
for i in train.columns:
    if a[i]!=0:
        null_columns.append(i)
        
for i in null_columns:
    train[i].fillna(train[i].value_counts().index[0],inplace=True)


#train['PoolQC'].fillna(train['PoolQC'].value_counts().index[0],inplace=True)
#train['PoolQC']

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.shape
test['Age_of_house']=test['YrSold']-test['YearBuilt']
test=test.drop(['YearBuilt','YrSold'],axis=1)
test=test.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley'],axis=1)
test.shape
#test['Age_of_house']=test['YrSold']-test['YearBuilt']
#test=test.drop(['YrSold','YearBuilt'],axis=1)
b=test.isnull().sum()
null_columns_test=[]
for i in test.columns:
    if b[i]!=0:
        null_columns_test.append(i)
        
for i in null_columns_test:
    test[i].fillna(test[i].value_counts().index[0],inplace=True)




a=train.columns
categorical=[i for i in a if train[i].dtype == 'O']


b=test.columns
categorical_test=[i for i in b if test[i].dtype == 'O']

numerical=[i for i in a if i not in categorical ]


numerical_test=[i for i in a if i not in categorical_test ]

#for i in categorical:
#    print(train[i])
train=train.drop(['Id'],axis=1)
ids=test['Id']
test=test.drop(['Id'],axis=1)
for i in train.columns:
    print(train[i].unique())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in categorical:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])
#train=pd.get_dummies(train,drop_first=True)
#test=pd.get_dummies(test,drop_first=True)
#train['New_year']=2020
#train['n_year']=train['New_year']-train['YrSold']
#train=train.drop(['New_year'],axis=1)
#train.iloc[:,:19]
#for i in train.columns:
#    print(train[i].unique())
#from sklearn.decomposition import PCA
#train=train.drop(['SalePrice'],axis=1)


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train)
train=ss.transform(train)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(train)
train=pca.transform(train)

test=ss.transform(test)
test=pca.transform(test)
#pca=PCA(n_components=7)
#X_train=pca.fit_transform(X_train)
#X_test=pca.fit_transform(X_test)
#test_new=pca.fit_transform(test_new)
#pca.explained_variance_ratio_
train.shape
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#train=sc.fit_transform(train)
#test=sc.transform(test)
#test=sc.transform(test)

import matplotlib.pyplot as plt 
plt.figure(figsize=(8,10))
plt.scatter(train[:,0],train[:,1],c=target)


from sklearn.linear_model import LinearRegression
lg=LinearRegression()
lg.fit(train,target)
pr=lg.predict(test)

output = pd.DataFrame({'Id': ids, 'SalePrice': pr})
output.to_csv('my_submission2.csv', index=False)
from xgboost import XGBRegressor
#from sklearn.ensemble import RandomForestRegressor 
#from sklearn.metrics import mean_squared_log_error
lr=XGBRegressor()
lr.fit(train,target)
#predic=lr.predict(X_test)
prediction=lr.predict(test)
#print(math.sqrt(mean_squared_log_error(predic,test)))
output = pd.DataFrame({'Id': ids, 'SalePrice': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.01, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(train,
         target)
prediction=xgb_grid.predict(test)
print(math.sqrt(mean_squared_log_error(predic,test)))

output = pd.DataFrame({'Id': ids, 'SalePrice': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
