# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.info()
null_train=train.columns[train.isna().any()==True].tolist()

test.drop(['PoolQC',
 'Fence',
 'MiscFeature','Alley',"FireplaceQu"],inplace=True,axis=1)
train.drop(['PoolQC',
 'Fence',
 'MiscFeature','Alley',"FireplaceQu"],inplace=True,axis=1)


train.info()
(null_train)
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrType']=train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])

train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['BsmtExposure']=train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType1']=train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])
train['BsmtFinType2']=train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train['Electrical']=train['Electrical'].fillna(train['Electrical'].mode()[0])
train['GarageType']=train['GarageType'].fillna(train['GarageType'].mode()[0])
train['GarageFinish']=train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train['GarageCond']=train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train['GarageQual']=train['GarageQual'].fillna(train['GarageQual'].mode()[0])
sns.heatmap(train.isnull())

li=train.dtypes
li=train.columns[train.dtypes=='object'].tolist()
li
le = preprocessing.LabelEncoder()
z=['LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'SaleType',
 'SaleCondition']
train['MSZoning']=le.fit_transform(train['MSZoning'])
train['Street']=le.fit_transform(train['Street'])
train['LotShape']=le.fit_transform(train['LotShape'])
train['LandContour']=le.fit_transform(train['LandContour'])
train['Utilities']=le.fit_transform(train['Utilities'])



for x in z:
    train[x]=le.fit_transform(train[x])

x=train.drop(['SalePrice','Id'],axis=1)
y=train['SalePrice']
x_train,x_test,y_train, y_test = train_test_split(x, y, random_state=101,test_size=0.2)
#model=(RandomForestRegressor(n_estimators=100,random_state=101))
#model.fit(x_train,y_train)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#pred=model.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
test.info()
null_test=test.columns[test.isna().any()==True].tolist()
null_test
for x in null_test:
    if x not in null_train:
        print(x)
li_obj=test.columns[test.dtypes=='object'].tolist()
li_float=test.columns[test.dtypes=='float64'].tolist()
li_int=test.columns[test.dtypes=='int64'].tolist()
for x in li_obj:
    test[x]=test[x].fillna(test[x].mode()[0])
for x in li_int:
    test[x]=test[x].fillna(test[x].mean())
for x in li_float:
    test[x]=test[x].fillna(test[x].mean())
for x in li_obj:
    test[x]=le.fit_transform(test[x])
    
train.info()
test.info()
test_=test.drop('Id',axis=1)
test_=scaler.transform(test_)
test.to_csv('clean_test.csv',index=False)
#my_submission = pd.DataFrame({'Id':test['Id'],'SalePrice': pred_test})
train.to_csv('clean_train.csv',index=False)
#my_submission.to_csv('submission.csv', index=False)
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100)


parameters = {
    "n_estimators":[5,50,250,500],
    "max_depth":[1,3,5,7,9],
    "learning_rate":[0.01,0.1,1,10,100]
}
GBR.fit(x_train, y_train)
pred=GBR.predict(test_)

#_train.isnull().sum()
from sklearn.model_selection import  GridSearchCV


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
# define dataset

# define the base models
level0 = list()
level0.append(('knn', KNeighborsRegressor()))
level0.append(('cart', DecisionTreeRegressor()))
level0.append(('svm', SVR()))
level0.append(('GB', GradientBoostingRegressor(n_estimators=100)))
level0.append(('RNN',RandomForestRegressor(n_estimators=100)))
# define meta learner model
#level1 = LinearRegression()
# define the stacking ensemble
#model = StackingRegressor(estimators=level0, cv=5)
# fit the model on all available data
#model.fit(x_train, y_train)
# make a prediction for one example
#data = [[0.59332206,-0.56637507,1.34808718,-0.57054047,-0.72480487,1.05648449,0.77744852,0.07361796,0.88398267,2.02843157,1.01902732,0.11227799,0.94218853,0.26741783,0.91458143,-0.72759572,1.08842814,-0.61450942,-0.69387293,1.69169009] 
#pred2=model.predict(test_)

my_submission = pd.DataFrame({'Id':test['Id'],'SalePrice': pred})
my_submission.to_csv('lastbjbhhb2.csv', index=False)

from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model)
sns.distplot(train['SalePrice'],bins=80)
visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
visualizer.show()       



