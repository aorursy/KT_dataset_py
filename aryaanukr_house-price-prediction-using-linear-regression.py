# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train.isnull().sum().values
test.info()
test.isnull().sum().values
for col in train.columns:

    a = train.isnull().sum()[col] > 400

    if a == True:

        print(col,train.isnull().sum()[col])
train = train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis = 1)
for col in test.columns:

    a = test.isnull().sum()[col] > 400

    if a == True:

        print(col,test.isnull().sum()[col])
test = test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis = 1)
ax, fig = plt.subplots(figsize = (15,15))

sns.heatmap(train.corr())
train.corr().nlargest(25,'SalePrice')['SalePrice']
train.shape, test.shape
## Find coumns having missing values in train dataset

b = []

for col in train.columns:

    a = train.isnull().sum()[col] > 0

    if a == True:

        b.append(col)

        print(col,train.isnull().sum()[col],train[col].dtype)
## Fill missing values in train dataset



for col in b:   

    if train[col].dtype == 'float64':

        train[col].fillna(train[col].mean(),inplace = True)

    else:

        train[col] = train[col].fillna(train[col].mode()[0])
# Find coumns having missing values in test dataset



c = []

for col in test.columns:

    f = test.isnull().sum()[col] > 0

    if f == True:

        c.append(col)

        print(col,test.isnull().sum()[col],test[col].dtype)
## Fill missing values in test dataset



for col in c:   

    if test[col].dtype == 'float64':

        test[col].fillna(test[col].mean(),inplace = True)

    else:

        test[col] = test[col].fillna(test[col].mode()[0])
train.isnull().sum().values
test.isnull().sum().values
train.shape, test.shape
n = []

for col in train.columns:

    if train[col].dtypes == 'object':

        n.append(col)

        print(col)

print("total number of columns having categorical data is %s"%len(n))
m = []

for col in train.columns:

    if train[col].dtypes == 'object':

        m.append(col)

        print(col)

print("total number of columns having categorical data is %s"%len(m))
# check for the columns which have different number of types in test and train dataset and drop that
for col in m:

    if len(train[col].value_counts()) != len(test[col].value_counts()):

        print(col,len(train[col].value_counts()),len(test[col].value_counts()))
train1 = train



train1 = train1.drop(['Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating',

                      'Electrical','GarageQual'],axis = 1)
test1= test



test1 = test1.drop(['Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating',

                      'Electrical','GarageQual'],axis = 1)
train1.shape , test1.shape
n_new = []

for col in train1.columns:

    if train1[col].dtypes == 'object':

        n_new.append(col)

        print(col)
len(n_new)
# Create dummies



for col in n_new:

    if len(train1[col].value_counts()) <= 8:

        train1[col] = train1[col].astype('category')

        train1[col] = train1[col].cat.codes

        train1 = pd.get_dummies(train1, columns=[col], drop_first=True)

    else:

        train1[col] = train1[col].astype('category')

        train1[col] = train1[col].cat.codes
m_new = []

for col in test1.columns:

    if test1[col].dtypes == 'object':

        m_new.append(col)

        print(col)
len(m_new)
for col in m_new:

    if len(test1[col].value_counts()) <= 8:

        test1[col] = test1[col].astype('category')

        test1[col] = test1[col].cat.codes

        test1 = pd.get_dummies(test1, columns=[col], drop_first=True)

    else:

        test1[col] = test1[col].astype('category')

        test1[col] = test1[col].cat.codes
train1.shape , test1.shape
y = train1['SalePrice']

x = train1.drop(['SalePrice'],axis = 1)

x_test = test1
from sklearn.model_selection import  train_test_split

x_train,x_val,y_train,y_val = train_test_split(x,y, test_size = 0.2,random_state = 100)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm = lm.fit(x_train,y_train)
y_pred = lm.predict(x_val)
y_error = y_val - y_pred
y_error.head()
from sklearn.metrics import r2_score

r2_score(y_val,y_pred)
import statsmodels.api as sma

x_train1 = sma.add_constant(x_train) ## let's add an intercept (beta_0) to our model

x_val1 =sma.add_constant(x_val)
import statsmodels.formula.api as sm

sm1 = sm.OLS(y_train,x_train1).fit()

sm1.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

[variance_inflation_factor(x_train1.values,j) for j in range(x_train1.shape[1])]
# drop columns having high p-value(p>|t|), (here, ie greater than 0.4), lower the p value better it is



x_train1 = x_train1.drop(['Id','Condition1','BsmtUnfSF','LowQualFinSF','BsmtHalfBath','HalfBath','GarageYrBlt','GarageArea',

                    'ScreenPorch','PoolArea','MiscVal','MoSold','LotShape_1','LotConfig_3','LotConfig_4','LandSlope_2',

                    'BldgType_4','RoofStyle_1','RoofStyle_2','RoofStyle_3','RoofStyle_4','RoofStyle_5','MasVnrType_1',

                    'MasVnrType_3','ExterQual_1','ExterCond_2','Foundation_3','Foundation_4','BsmtCond_1','BsmtFinType1_1',

                    'BsmtFinType1_3','BsmtFinType1_4','BsmtFinType2_2','BsmtFinType2_3','BsmtFinType2_4','BsmtFinType2_5',

                    'HeatingQC_1','Functional_1','Functional_4','GarageType_2','GarageType_3','GarageType_4','GarageType_5',

                    'GarageFinish_2','GarageCond_1','GarageCond_2','GarageCond_3','GarageCond_4','PavedDrive_1','PavedDrive_2',

                    'SaleCondition_1','SaleCondition_2','SaleCondition_3','SaleCondition_4'],axis =1)
# drop columns having high p-value, (here, ie greater than 0.4), lower the p value better it is



x_val1 = x_val1.drop(['Id','Condition1','BsmtUnfSF','LowQualFinSF','BsmtHalfBath','HalfBath','GarageYrBlt','GarageArea',

                    'ScreenPorch','PoolArea','MiscVal','MoSold','LotShape_1','LotConfig_3','LotConfig_4','LandSlope_2',

                    'BldgType_4','RoofStyle_1','RoofStyle_2','RoofStyle_3','RoofStyle_4','RoofStyle_5','MasVnrType_1',

                    'MasVnrType_3','ExterQual_1','ExterCond_2','Foundation_3','Foundation_4','BsmtCond_1','BsmtFinType1_1',

                    'BsmtFinType1_3','BsmtFinType1_4','BsmtFinType2_2','BsmtFinType2_3','BsmtFinType2_4','BsmtFinType2_5',

                    'HeatingQC_1','Functional_1','Functional_4','GarageType_2','GarageType_3','GarageType_4','GarageType_5',

                    'GarageFinish_2','GarageCond_1','GarageCond_2','GarageCond_3','GarageCond_4','PavedDrive_1','PavedDrive_2',

                    'SaleCondition_1','SaleCondition_2','SaleCondition_3','SaleCondition_4'],axis =1)
# drop columns having high p-value, (here, ie greater than 0.4), lower the p value better it is



x_test = x_test.drop(['Id','Condition1','BsmtUnfSF','LowQualFinSF','BsmtHalfBath','HalfBath','GarageYrBlt','GarageArea',

                    'ScreenPorch','PoolArea','MiscVal','MoSold','LotShape_1','LotConfig_3','LotConfig_4','LandSlope_2',

                    'BldgType_4','RoofStyle_1','RoofStyle_2','RoofStyle_3','RoofStyle_4','RoofStyle_5','MasVnrType_1',

                    'MasVnrType_3','ExterQual_1','ExterCond_2','Foundation_3','Foundation_4','BsmtCond_1','BsmtFinType1_1',

                    'BsmtFinType1_3','BsmtFinType1_4','BsmtFinType2_2','BsmtFinType2_3','BsmtFinType2_4','BsmtFinType2_5',

                    'HeatingQC_1','Functional_1','Functional_4','GarageType_2','GarageType_3','GarageType_4','GarageType_5',

                    'GarageFinish_2','GarageCond_1','GarageCond_2','GarageCond_3','GarageCond_4','PavedDrive_1','PavedDrive_2',

                    'SaleCondition_1','SaleCondition_2','SaleCondition_3','SaleCondition_4'],axis =1)
x_train1 = x_train1.drop(['const'],axis =1)

x_val1 = x_val1.drop(['const'],axis = 1)
## create model2



lm2 = lm.fit(x_train1,y_train)
y_pred2 = lm2.predict(x_val1)
y_error1 = y_pred2 - y_val
lm2.intercept_
x_train2 = sma.add_constant(x_train1) ## let's add an intercept (beta_0) to our model

x_val2 =sma.add_constant(x_val1)
sm2 = sm.OLS(y_train,x_train2).fit()

sm2.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

[variance_inflation_factor(x_train2.values, j) for j in range(x_train2.shape[1])]
## Drop columns having varience greater than 5



x_train2 = x_train2.drop([ 'const','YearBuilt', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'MSZoning_1', 'MSZoning_3',

                      'MSZoning_4', 'ExterQual_2', 'ExterQual_3', 'Foundation_2', 'BsmtQual_2', 'BsmtQual_3', 'KitchenQual_2',

                      'KitchenQual_3'],axis=1)
x_val2 = x_val2.drop(['const','YearBuilt', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'MSZoning_1', 'MSZoning_3',

                      'MSZoning_4', 'ExterQual_2', 'ExterQual_3', 'Foundation_2', 'BsmtQual_2', 'BsmtQual_3', 'KitchenQual_2',

                      'KitchenQual_3'],axis=1)
x_test = x_test.drop([ 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'MSZoning_1', 'MSZoning_3',

                      'MSZoning_4', 'ExterQual_2', 'ExterQual_3', 'Foundation_2', 'BsmtQual_2', 'BsmtQual_3', 'KitchenQual_2',

                      'KitchenQual_3'],axis=1)
## create model3

lm3 = lm.fit(x_train2,y_train)
y_pred2 = lm3.predict(x_val2)

y_error2 = y_pred2 - y_val
from sklearn.metrics import r2_score

r2_score(y_val,y_pred2)
x_train3 = sma.add_constant(x_train2) ## let's add an intercept (beta_0) to our model

x_val3 =sma.add_constant(x_val2)



sm3 = sm.OLS(y_train,x_train3).fit()

sm3.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

[variance_inflation_factor(x_train3.values, j) for j in range(x_train3.shape[1])]
y_salesprice_pred = lm3.predict(x_test)
list(y_salesprice_pred)
id = test['Id']

ID = list(id)

y=list(y_salesprice_pred)

output = pd.DataFrame({'Id': ID, 'SalePrice': y})

output.to_csv('Predicted HousePrice.csv', index=False)
out = pd.read_csv('Predicted HousePrice.csv')
out.head()