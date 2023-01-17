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
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 200)
applData = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", encoding= 'unicode_escape')
testData = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", encoding= 'unicode_escape')
applData.head()
print(applData.shape)
print(applData.info())
pctDF = (applData.isnull().sum()/len(applData) * 100).reset_index()
pctDF.columns = ['Columns','Missing Value Percentage']
pctDF = pctDF[pctDF['Missing Value Percentage'] > 45]
pctDF
columns = list(map(lambda x: x, pctDF['Columns']))
applData.drop(columns = columns, inplace=True)
applData.head()
#Remove the id Columns
applData.drop(columns = ['Id'], inplace=True)
from scipy.special import boxcox1p

transformedColumns = []
droppedColumns = []
for i in enumerate(applData.columns):
    column = i[1]
    valueDF = ((applData[column].value_counts() / len(applData[column]))*100).reset_index()
    valueDF.columns = ['column','Value Percentage']
    skewDF = valueDF[valueDF['Value Percentage'] >= 75]
    if(len(skewDF) > 0):
        applData.drop(columns = [column], inplace=True)
        droppedColumns.append(column)
            
print("Dropped Highly skewed columns:", droppedColumns)
applData.head()
# 44 Features are remaining after removing the highly skewed columns.
applData.shape
pctDF = (applData.isnull().sum()/len(applData) * 100).reset_index()
pctDF.columns = ['Columns','Missing Value Percentage']
pctDF = pctDF[pctDF['Missing Value Percentage'] > 0]
pctDF
print(applData['LotFrontage'].value_counts())
print("Median: ", applData["LotFrontage"].median())
print("Mean: ", applData["LotFrontage"].mean())
plt.figure(figsize  = (20, 5))
plt.subplot(1, 2, 1)
sns.distplot(applData['LotFrontage'].fillna(applData["LotFrontage"].median()))
plt.subplot(1, 2, 2)
sns.distplot(applData['LotFrontage'].fillna(applData["LotFrontage"].mean()))
print("Missing Values Before: ", applData['LotFrontage'].isnull().sum())
applData['LotFrontage'].fillna(applData["LotFrontage"].median(), inplace=True)
print("Missing Values After: ", applData['LotFrontage'].isnull().sum())
print(applData['MasVnrType'].value_counts())
print("Mode: ", applData["MasVnrType"].mode())
print("Missing Values Before: ", applData['MasVnrType'].isnull().sum())
applData['MasVnrType'].fillna('None', inplace=True)
print("Missing Values After: ", applData['MasVnrType'].isnull().sum())
print(applData['MasVnrArea'].value_counts())
print("Missing Values Before: ", applData['MasVnrArea'].isnull().sum())
applData['MasVnrArea'].fillna(0, inplace=True)
print("Missing Values After: ", applData['MasVnrArea'].isnull().sum())
print(applData['BsmtQual'].value_counts())
print("Missing Values Before: ", applData['BsmtQual'].isnull().sum())
applData['BsmtQual'].fillna('NA', inplace=True)
print("Missing Values After: ", applData['BsmtQual'].isnull().sum())

#For BsmtExposure and BsmtFinType1 apply the same rule as BsmtQual
applData['BsmtExposure'].fillna('NA', inplace=True)
applData['BsmtFinType1'].fillna('NA', inplace=True)

print(applData['GarageType'].value_counts())
print("Missing Values Before: ", applData['GarageType'].isnull().sum())
applData['GarageType'].fillna('NA', inplace=True)
applData['GarageFinish'].fillna('NA', inplace=True)
print("Missing Values After: ", applData['GarageType'].isnull().sum())


print(applData['GarageYrBlt'].value_counts())
import datetime

today = datetime.datetime.now()
print("Missing Values Before: ", applData['GarageYrBlt'].isnull().sum())
applData['GarageYrBlt'].fillna(today.year+1, inplace=True) #Set to future date
print("Missing Values After: ", applData['GarageYrBlt'].isnull().sum())

pctDF = (applData.isnull().sum()/len(applData) * 100).reset_index()
pctDF.columns = ['Columns','Missing Value Percentage']
pctDF = pctDF[pctDF['Missing Value Percentage'] > 0]
pctDF
applData.head()
columns = ['LotArea','LotFrontage','MasVnrArea','WoodDeckSF','OpenPorchSF','GrLivArea','TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1']

plt.figure(figsize  = (15,30))
for i in enumerate(columns):
    plt.subplot(10, 3, i[0]+1)
    sns.boxplot(data=applData, x=i[1])
def treatOutliers(col, df):
    q4 = df[col].quantile(0.99)
    df[col][df[col] >=  q4] = q4
    
    q1 = df[col].quantile(0.01)
    df[col][df[col] <=  q1] = q1
    
    return df
columns = ['LotArea','LotFrontage','MasVnrArea','WoodDeckSF','OpenPorchSF','GrLivArea','TotalBsmtSF', 'BsmtFinSF1', '1stFlrSF']
for col in columns:
    applData = treatOutliers(col, applData)
plt.figure(figsize  = (15,30))
for i in enumerate(columns):
    plt.subplot(10, 3, i[0]+1)
    sns.boxplot(data=applData, x=i[1])
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numdf = applData.select_dtypes(include=numerics)

plt.figure(figsize  = (15,30))
for i in enumerate(numdf.columns.drop('SalePrice')):
    plt.subplot(15, 3, i[0]+1)
    sns.distplot(applData[i[1]])
applData.head()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
plt.figure(figsize  = (15,30))
sns.pairplot(applData[cols], size = 2.5)
plt.show();
plt.figure(figsize  = (15,50))
for i in enumerate(numdf.columns.drop('SalePrice')):
    plt.subplot(15, 3, i[0]+1)
    sns.scatterplot(x = applData[i[1]], y = applData['SalePrice'])


from scipy.stats import probplot

plt.figure(figsize  = (15, 10))
plt.subplot(2, 2, 1)
sns.distplot(applData['SalePrice'])
plt.subplot(2, 2, 2)
res = probplot(applData['SalePrice'], plot=plt)
plt.figure(figsize  = (15, 10))
plt.subplot(2, 2, 1)
sns.distplot(applData['SalePrice'])
plt.subplot(2, 2, 2)
res = probplot(np.log(applData['SalePrice']), plot=plt)
def makeValuesAsOther(df, col, percent):
    print('Before')
    print(df[col].value_counts()/len(df)*100)
    
    values = (df[col].value_counts()/len(df)*100).reset_index()
    values = values[values[col] < percent]["index"]

    for i in values:
        df[col].replace(i, 'Other', inplace=True)
        
    print('After')
    print(df[col].value_counts()/len(df)*100)    
makeValuesAsOther(applData, "Neighborhood", 2)
makeValuesAsOther(applData, "HouseStyle", 10)
makeValuesAsOther(applData, "Exterior1st", 6)
makeValuesAsOther(applData, "Exterior2nd", 6)
applData.head()
today = datetime.datetime.now()

columns = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']

for col in columns:
    applData[col+'Age'] = today.year - applData[col]

applData.drop(columns = columns, inplace=True)
applData.head()
valMap1 = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
applData['ExterQual'] = applData['ExterQual'].map(valMap1)
applData['BsmtQual'] = applData['BsmtQual'].map(valMap1)
applData['HeatingQC'] = applData['HeatingQC'].map(valMap1)
applData['KitchenQual'] = applData['KitchenQual'].map(valMap1)

valMap2 = {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
applData['BsmtExposure'] = applData['BsmtExposure'].map(valMap2)
applData.head()
corr = applData.corr()
corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corr_df = corr.unstack().reset_index()
corr_df.columns = ['Variable1', 'Variable2', 'Correlation']
corr_df.dropna(subset = ['Correlation'], inplace=True)
corr_df['Correlation'] = round(corr_df['Correlation'].abs(), 2)
corr_df.sort_values(by = 'Correlation', ascending=False).head(10)
plt.figure(figsize  = (30,20))
sns.heatmap(applData.corr(), annot=True, cmap='RdYlGn')

columns = ['LotShape', 'LotConfig', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'GarageType', 'GarageFinish']

applData = pd.get_dummies(data=applData, columns=columns, drop_first=True)

applData.head()
stdScaler = StandardScaler()

columns = ['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF',
           'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF', 
           'MoSold','YearBuiltAge','YearRemodAddAge','GarageYrBltAge','YrSoldAge','OverallQual','OverallCond',
           'ExterQual','BsmtQual','BsmtExposure','HeatingQC','FullBath','BedroomAbvGr','KitchenQual','TotRmsAbvGrd',
           'BsmtFullBath','HalfBath','Fireplaces','GarageCars']
applData[columns] = stdScaler.fit_transform(applData[columns])

applData.head()
# Futher divide the dataset in X_train and y_train
y_train = applData.pop('SalePrice')
X_train = applData
linreg = LinearRegression()
rfe = RFE(linreg, n_features_to_select=25 )
rfe = rfe.fit(X_train, y_train)
#useful columns according to rfe
useful_cols = X_train.columns[rfe.support_]
useful_cols
# Not useful columns according to rfe
X_train.columns[~rfe.support_]
X_train_rfe = X_train[useful_cols]
X_train_rfe.head()

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0,9.0,10.0,20.0,30.0,50.0,100.0 ]}

ridge = Ridge()

folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train_rfe, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
model_cv.best_estimator_
alpha = model_cv.best_estimator_.alpha
ridge = Ridge(alpha=alpha)

ridge.fit(X_train_rfe, y_train)
ridge.coef_
def score(y_pred, y_true):
    error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
    score = 1 - error
    return score
y_pred = ridge.predict(X_train_rfe)
print("Ridge Score: ", round(score(y_pred, y_train)*100, 2), "%")
ridgeCoefDF = pd.DataFrame()
ridgeCoefDF['Column'] = X_train_rfe.columns
ridgeCoefDF['Coef'] = ridge.coef_
ridgeCoefDF['Coef_Absolute'] = abs(ridgeCoefDF['Coef'])
ridgeCoefDF = ridgeCoefDF.sort_values(by = 'Coef_Absolute', ascending=False)
ridgeCoefDF.head(10)

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200 ]}

lasso = Lasso()

model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train_rfe, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha'] <= 200]
cv_results.head()
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()
model_cv.best_estimator_
alpha = model_cv.best_estimator_.alpha
lasso = Lasso(alpha=alpha)

lasso.fit(X_train_rfe, y_train)
lasso.coef_
y_pred = lasso.predict(X_train_rfe)
print("Lasso Score: ", round(score(y_pred, y_train)*100, 2), "%")
lassoCoefDF = pd.DataFrame()
lassoCoefDF['Column'] = X_train_rfe.columns
lassoCoefDF['Coef'] = lasso.coef_
lassoCoefDF['Coef_Abs'] = abs(lassoCoefDF['Coef'])
lassoCoefDF = lassoCoefDF.sort_values(by = 'Coef_Abs', ascending=False)
lassoCoefDF.head(10)
testData.head()
#Apply same rules in Test Data
testData['LotFrontage'].fillna(testData["LotFrontage"].median(), inplace=True)

testData['MasVnrType'].fillna('None', inplace=True)

testData['MasVnrArea'].fillna(0, inplace=True)

testData['BsmtQual'].fillna('NA', inplace=True)
testData['BsmtExposure'].fillna('NA', inplace=True)
testData['BsmtFinType1'].fillna('NA', inplace=True)

testData['GarageType'].fillna('NA', inplace=True)
testData['GarageFinish'].fillna('NA', inplace=True)

testData['GarageYrBlt'].fillna(today.year+1, inplace=True) #Set to future date
makeValuesAsOther(testData, "Neighborhood", 2)
makeValuesAsOther(testData, "HouseStyle", 10)
makeValuesAsOther(testData, "Exterior1st", 6)
makeValuesAsOther(testData, "Exterior2nd", 6)
today = datetime.datetime.now()

columns = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']

for col in columns:
    testData[col+'Age'] = today.year - testData[col]

testData.drop(columns = columns, inplace=True)
valMap1 = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
testData['ExterQual'] = testData['ExterQual'].map(valMap1)
testData['BsmtQual'] = testData['BsmtQual'].map(valMap1)
testData['HeatingQC'] = testData['HeatingQC'].map(valMap1)
testData['KitchenQual'] = testData['KitchenQual'].map(valMap1)

valMap2 = {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
testData['BsmtExposure'] = applData['BsmtExposure'].map(valMap2)
columns = ['LotShape', 'LotConfig', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'GarageType', 'GarageFinish']

testData = pd.get_dummies(data=testData, columns=columns, drop_first=True)
columns = ['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF',
           'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF', 
           'MoSold','YearBuiltAge','YearRemodAddAge','GarageYrBltAge','YrSoldAge','OverallQual','OverallCond',
           'ExterQual','BsmtQual','BsmtExposure','HeatingQC','FullBath','BedroomAbvGr','KitchenQual','TotRmsAbvGrd',
           'BsmtFullBath','HalfBath','Fireplaces','GarageCars']
testData[columns] = stdScaler.fit_transform(testData[columns])

X_test = testData[X_train_rfe.columns]
X_test.head()
X_test['KitchenQual'].fillna(X_test['KitchenQual'].mean(), inplace=True)
X_test['BsmtFinSF1'].fillna(X_test['BsmtFinSF1'].median(), inplace=True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].median(), inplace=True)
y_test_pred_ridge = ridge.predict(X_test)
testData['Predicted House Price Ridge'] = y_test_pred_ridge
y_test_pred_lasso = lasso.predict(X_test)
testData['Predicted House Price Lasso'] = y_test_pred_lasso
testData.head()
