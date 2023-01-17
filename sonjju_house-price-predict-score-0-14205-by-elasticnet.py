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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
housing_train = pd.read_csv("../input/train.csv")

housing_test = pd.read_csv("../input/test.csv")

print('housing_train shape',housing_train.shape)

print('housing_test shape', housing_test.shape)
housing_train.head()
#check NAs

NAs = pd.concat([housing_train.isnull().sum(), housing_test.isnull().sum()], axis=1, keys=['Train','Test'])

NAs[NAs.sum(axis=1)>0]
housing_train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature',

                   'GarageCond','GarageFinish','GarageType','BsmtFinType1',

                   'BsmtFinType2'], axis=1, inplace=True)

housing_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature',

                   'GarageCond','GarageFinish','GarageType','BsmtFinType1',

                   'BsmtFinType2'], axis=1, inplace=True)
# Now handle numeric data
housing_train.corr()['SalePrice'].sort_values(ascending=False)
#Delete features that corr is under 0.4

housing_train.drop(['BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF',

                   'OpenPorchSF','HalfBath','LotArea','BsmtFullBath',

                   'BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea',

                   'MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',

                   'MiscVal','LowQualFinSF','YrSold','OverallCond',

                   'MSSubClass','EnclosedPorch','KitchenAbvGr'], axis=1,inplace=True)



housing_test.drop(['BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF',

                   'OpenPorchSF','HalfBath','LotArea','BsmtFullBath',

                   'BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea',

                   'MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',

                   'MiscVal','LowQualFinSF','YrSold','OverallCond',

                   'MSSubClass','EnclosedPorch','KitchenAbvGr'], axis=1,inplace=True)
sns.regplot(data=housing_train, x='GrLivArea',y='SalePrice',color='orange')
from pandas.tools.plotting import scatter_matrix

attributes=['SalePrice','OverallQual','GrLivArea','GarageCars',

            'GarageArea','TotalBsmtSF','1stFlrSF','FullBath',

            'TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt',

            'MasVnrArea','Fireplaces']
cat_attributes=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig',

         'LandSlope','Neighborhood','Condition1','Condition2','BldgType',

         'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',

         'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',

         'BsmtExposure','Heating','HeatingQC',

         'CentralAir','Electrical','KitchenQual','Functional',

         'GarageQual','PavedDrive',

         'SaleType','SaleCondition']
sns.set_style('whitegrid')

sns.barplot(data=housing_train, x='Neighborhood',y='SalePrice',color='orange')
#Fill NAs

NAs = pd.concat([housing_train.isnull().sum(), housing_test.isnull().sum()], axis=1, keys=['Train','Test'])

NAs[NAs.sum(axis=1)>0]
for i in housing_train:

    if housing_train[i].isnull().sum() > 0:

        if i in cat_attributes:

            housing_train[i].fillna(housing_train[i].mode()[0], inplace= True)

        else:

            housing_train[i].fillna(housing_train[i].mean(), inplace= True)



for i in housing_test:

    if housing_test[i].isnull().sum() > 0:

        if i in cat_attributes:

            housing_test[i].fillna(housing_test[i].mode()[0], inplace= True)

        else:

            housing_test[i].fillna(housing_test[i].mean(), inplace= True)
#Fill NAs is done

NAs = pd.concat([housing_train.isnull().sum(), housing_test.isnull().sum()], axis=1, keys=['Train','Test'])

NAs[NAs.sum(axis=1)>0]
#split features and label

train_label = housing_train.pop('SalePrice')

train_feature = housing_train
#standardize numeric value

for i in train_feature:

    if i != 'Id':

        if train_feature[i].dtype == 'int64' or train_feature[i].dtype == 'float64':

            train_feature[i]= (train_feature[i]-train_feature[i].mean())/train_feature[i].std()

            

for i in housing_test:

    if i != 'Id':

        if housing_test[i].dtype == 'int64' or housing_test[i].dtype == 'float64':

            housing_test[i]= (housing_test[i]-housing_test[i].mean())/housing_test[i].std()
#Log transform label

train_label = np.log(train_label)

plt.hist(train_label)

plt.show()
#Encoding categorical data

#dummies version

dummies = pd.get_dummies(train_feature[cat_attributes])

train_feature.drop(cat_attributes, axis =1, inplace=True)

train_feature = pd.concat([train_feature, dummies], axis=1)



dummies = pd.get_dummies(housing_test[cat_attributes])

housing_test.drop(cat_attributes, axis =1, inplace=True)

housing_test = pd.concat([housing_test, dummies], axis=1)
housing_test.head()
#Delete features that are in train set but not in test set

excep_list=[]

for i in train_feature.keys():

    if i not in housing_test.keys():

        excep_list.append(i)



train_feature.drop(excep_list, axis=1, inplace=True)
#Split train, validation dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_feature, train_label,test_size=0.2, random_state=400)
from sklearn import linear_model

from sklearn.metrics import r2_score, mean_squared_error

reg=linear_model.LinearRegression()

reg.fit(x_train,y_train)

res = reg.predict(x_test)

#score

print('r2 score: {}'.format(r2_score(res,y_test)))

print('RMSE: {}'.format(np.sqrt(mean_squared_error(res,y_test))))

residual = y_test - res

residuals = pd.DataFrame({'Id':x_test['Id'], 'residual':residual})

sns.regplot(data=residuals, x='Id', y='residual')
#this is best

# Elastic net

reg = linear_model.ElasticNetCV(n_alphas=100, normalize=True, cv=5, l1_ratio=1.2)

reg.fit(x_train, y_train)

res = reg.predict(x_test)

#score

print('r2 score: {}'.format(r2_score(res,y_test)))

print('RMSE: {}'.format(np.sqrt(mean_squared_error(res,y_test))))

residual = y_test - res

residuals = pd.DataFrame({'Id':x_test['Id'], 'residual':residual})

sns.regplot(data=residuals, x='Id', y='residual',fit_reg=False)