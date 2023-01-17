# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
dataset_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
dataset_train.head()
#Checking for the Null Value
dataset_train.isnull().sum()

dataset_null = (dataset_train.isnull().sum() / len(dataset_train)) * 100
dataset_null = dataset_null.drop(dataset_null[dataset_null == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' : dataset_null})
missing_data
def missing_percentage(df):   
    
    missing_total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([missing_total, percent], axis=1, keys=['Missing_Total','Percent'])

missing_percentage(dataset_train)
#dataset_train = dataset_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis= 1)
def show_missing_list():
    list_missing = dataset_train.columns[dataset_train.isnull().any()].tolist()
    return list_missing

def cat_exploration(column):
    return dataset_train[column].value_counts()

def cat_imputation(colums, value):
    dataset_train.loc[dataset_train[colums].isnull(), colums] = value
dataset_train[show_missing_list()].isnull().mean()
cat_exploration('Alley')
cat_imputation('Alley', 'None')
dataset_train[['MasVnrType' , 'MasVnrArea']][dataset_train['MasVnrType'].isnull() == True]
cat_exploration('MasVnrType')
cat_imputation('MasVnrType', 'None')
cat_imputation('MasVnrArea', 0.0)
cat_exploration('Electrical')
cat_imputation('Electrical', 'SBrkr')
cat_exploration('FireplaceQu')
dataset_train['Fireplaces'][dataset_train['FireplaceQu'].isnull() == True].describe()
cat_imputation('FireplaceQu', 'None')
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
dataset_train[garage_cols][dataset_train['GarageType'].isnull()==True]
for cols in garage_cols:
    if dataset_train[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)
cat_exploration('PoolQC')
dataset_train['PoolArea'][dataset_train['PoolQC'].isnull() ==True].describe()
cat_imputation('PoolQC', 'None')
cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')
dataset_train[show_missing_list()].isnull().sum()
dataset_train['LotFrontage'].corr(dataset_train['LotArea'])
dataset_train['SqrtLotArea'] = np.sqrt(dataset_train['LotArea'])
dataset_train['LotFrontage'].corr(dataset_train['SqrtLotArea'])
cond = dataset_train['LotFrontage'].isnull()
dataset_train.LotFrontage[cond]=dataset_train.SqrtLotArea[cond]
del dataset_train['SqrtLotArea']
dataset_train[show_missing_list()].isnull().sum()
dataset_train.head(10)
#Get all Categorical Variable
def getCategorical(dataset):
    cat_var = [key for key in dict(dataset.dtypes)
          if dict(dataset.dtypes)[key] in ['object']]
    
    return cat_var


getCategorical(dataset_train)
#Get all numerical variable
def getNumericData(dataset):
    numeric_var = [key for key in dict(dataset.dtypes)
                   if dict(dataset.dtypes)[key]
                       in ['float64','float32','int32','int64']]
    return numeric_var
     

getNumericData(dataset_train)

x = dataset_train.drop(['SalePrice'], axis = 1)

X = pd.get_dummies(x, drop_first=True)
X_inscaled = pd.get_dummies(x, drop_first=True)

from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
X = scaler1.fit_transform(X)
y_ = scaler1.fit_transform(dataset_train[['SalePrice']])

from sklearn.feature_selection import SelectKBest,  \
                                      f_regression


selector_n = SelectKBest(score_func=f_regression, k = 50)
X_k = selector_n.fit_transform(X, y_)

f_score = selector_n.scores_
p_value = selector_n.pvalues_

#columns = list(x.columns)
#for i in range(0, len(columns)):
#    print("f1 score %4.2f"% f_score[i])
#    print("p1 score %2.6f"% p_value[i])
#    print("_____________________________________")
    
    

selected_col = []
for i in range (0, len(x.columns)):
    if p_value[i] < 0.5:
        selected_col.append(p_value[i])

cols = selector_n.get_support(indices=True)
selected_col = X_inscaled.columns[cols].tolist()
print("Selected Cols ", selected_col)

#This are the top 50 feature 
coloumns_sel = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 
                'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'MSZoning_RL', 'MSZoning_RM', 
                'LotShape_Reg', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 
                'Exterior1st_VinylSd', 'Exterior2nd_VinylSd', 'MasVnrType_None', 
                'MasVnrType_Stone', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 
                'Foundation_PConc', 'BsmtQual_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 
                'BsmtFinType1_GLQ', 'HeatingQC_TA', 'CentralAir_Y', 'KitchenQual_Gd', 
                'KitchenQual_TA', 'FireplaceQu_Gd', 'FireplaceQu_None', 'GarageType_Attchd', 
                'GarageType_Detchd', 'GarageFinish_Unf', 'GarageQual_TA', 'GarageCond_TA',
                'SaleType_New', 'SaleCondition_Partial']

data_select = X_inscaled[coloumns_sel]


X = data_select

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, y_, test_size = 0.3, random_state = 1234)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)

print("R Score ", lr.score(X_train, Y_train))
print("R Score Test", lr.score(X_test, Y_test))
print("RMSE ", np.sqrt(mean_squared_error(Y_test, Y_predict)))

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, Y_test, Y_predict, cv=10)

fig, ax = plt.subplots()
ax.scatter(Y_test, predicted)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

y_result = scaler1.inverse_transform(Y_predict).tolist()
y_resul_actual = scaler1.inverse_transform(Y_test).tolist()