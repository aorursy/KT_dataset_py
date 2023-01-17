import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import math 

np.random.seed(2019)

from scipy.stats import skew

from scipy import stats

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier

from xgboost.sklearn import XGBRegressor



import statsmodels



#!pip install ml_metrics

from ml_metrics import rmsle



import matplotlib.pyplot as plt

%matplotlib inline

print("done")
def read_and_concat_dataset(training_path, test_path):

    train = pd.read_csv(training_path)

    train['train'] = 1

    test = pd.read_csv(test_path)

    test['train'] = 0

    data = train.append(test, ignore_index=True)

    return train, test, data



train, test, data = read_and_concat_dataset('../input/train.csv', '../input/test.csv')

data = data.set_index('Id')
data.columns[data.isnull().sum()>0]
def filling_missing_values(data,variable, new_value):

    data[variable] = data[variable].fillna(new_value)
filling_missing_values(data,'GarageCond','None')

filling_missing_values(data,'GarageQual','None')

filling_missing_values(data,'FireplaceQu','None')

filling_missing_values(data,'BsmtCond','None')

filling_missing_values(data,'BsmtQual','None')

filling_missing_values(data,'PoolQC','None')

filling_missing_values(data,'MiscFeature','None')
data['MSSubClass'][data['MSSubClass'] == 20] = '1-STORY 1946 & NEWER ALL STYLES'

data['MSSubClass'][data['MSSubClass'] == 30] = '1-STORY 1945 & OLDER'

data['MSSubClass'][data['MSSubClass'] == 40] = '1-STORY W/FINISHED ATTIC ALL AGES'

data['MSSubClass'][data['MSSubClass'] == 45] = '1-1/2 STORY - UNFINISHED ALL AGES'

data['MSSubClass'][data['MSSubClass'] == 50] = '1-1/2 STORY FINISHED ALL AGES'

data['MSSubClass'][data['MSSubClass'] == 60] = '2-STORY 1946 & NEWER'

data['MSSubClass'][data['MSSubClass'] == 70] = '2-STORY 1945 & OLDER'

data['MSSubClass'][data['MSSubClass'] == 75] = '2-1/2 STORY ALL AGES'

data['MSSubClass'][data['MSSubClass'] == 80] = 'SPLIT OR MULTI-LEVEL'

data['MSSubClass'][data['MSSubClass'] == 85] = 'SPLIT FOYER'

data['MSSubClass'][data['MSSubClass'] == 90] = 'DUPLEX - ALL STYLES AND AGES'

data['MSSubClass'][data['MSSubClass'] == 120] = '1-STORY PUD (Planned Unit Development) - 1946 & NEWER'

data['MSSubClass'][data['MSSubClass'] == 150] = '1-1/2 STORY PUD - ALL AGES'

data['MSSubClass'][data['MSSubClass'] == 160] = '2-STORY PUD - 1946 & NEWER'

data['MSSubClass'][data['MSSubClass'] == 180] = 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER'

data['MSSubClass'][data['MSSubClass'] == 190] = '2 FAMILY CONVERSION - ALL STYLES AND AGES'
def fixing_ordinal_variables(data, variable):

    data[variable][data[variable] == 'Ex'] = 5

    data[variable][data[variable] == 'Gd'] = 4

    data[variable][data[variable] == 'TA'] = 3

    data[variable][data[variable] == 'Fa'] = 2

    data[variable][data[variable] == 'Po'] = 1

    data[variable][data[variable] == 'None'] = 0
fixing_ordinal_variables(data,'ExterQual')

fixing_ordinal_variables(data,'ExterCond')

fixing_ordinal_variables(data,'BsmtCond')

fixing_ordinal_variables(data,'BsmtQual')

fixing_ordinal_variables(data,'HeatingQC')

fixing_ordinal_variables(data,'KitchenQual')

fixing_ordinal_variables(data,'FireplaceQu')

fixing_ordinal_variables(data,'GarageQual')

fixing_ordinal_variables(data,'GarageCond')

fixing_ordinal_variables(data,'PoolQC')
data['PavedDrive'][data['PavedDrive'] == 'Y'] = 3

data['PavedDrive'][data['PavedDrive'] == 'P'] = 2

data['PavedDrive'][data['PavedDrive'] == 'N'] = 1
colu = data.columns[(data.isnull().sum()<50) & (data.isnull().sum()>0)]

for i in colu:

    print(data[colu].isnull().sum())
colu = data.columns[data.isnull().sum()>=50]

for i in colu:

    print(data[colu].isnull().sum())
filling_missing_values(data, 'GarageArea',0)

filling_missing_values(data, 'GarageCars',0)

data['GarageFinish'][(data.GarageFinish.isnull()==True) & (data.GarageCond==0)] =0

data['GarageType'][(data.GarageType.isnull()==True) & (data.GarageCond==0)] =0

data['GarageYrBlt'][(data.GarageYrBlt.isnull()==True) & (data.GarageCond==0)] =0
print(data[['MiscFeature','MiscVal']][(data.MiscFeature=='None') & (data.MiscVal>0)])

data.MiscVal.loc[2550] = 0



print(data[['MiscFeature','MiscVal']][(data.MiscVal==0) & (data.MiscFeature!='None')])

c=data[['MiscFeature','MiscVal']][(data.MiscVal==0) & (data.MiscFeature!='None')].index

data.MiscFeature.loc[c] = 'None'
def inputing(variab):

    y = data[variab]

    data2 = data.drop([variab],axis=1)

    col = data2.columns[data2.isnull().sum()==0]

    data2 = data2[col]

    data2 = pd.get_dummies(data2)

    c_train = y[y.notnull()==True].index

    y_train = y[c_train]

    columny = data2.columns

    X_train = data2[columny].loc[c_train]

    c_test = y[y.notnull()!=True].index

    y_test = y[c_test]

    X_test = data2[columny].loc[c_test]

    #Model

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #Filling missing data

    y_pred = pd.Series(y_pred, index=c_test)

    data[variab].loc[c_test] = y_pred.loc[c_test]

    

def inputingnum(variab):

    y = data[variab]

    data2 = data.drop([variab],axis=1)

    col = data2.columns[data2.isnull().sum()==0]

    data2 = data2[col]

    data2 = pd.get_dummies(data2)

    c_train = y[y.notnull()==True].index

    y_train = y[c_train]

    columny = data2.columns

    X_train = data2[columny].loc[c_train]

    c_test = y[y.notnull()!=True].index

    y_test = y[c_test]

    X_test = data2[columny].loc[c_test]

    #Model

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #Filling missing data

    y_pred = pd.Series(y_pred, index=c_test)

    data[variab].loc[c_test] = y_pred.loc[c_test]
inputing(variab='Electrical')

inputing(variab='Exterior2nd')

inputing(variab='Exterior1st')

inputing(variab='MasVnrType')

inputing(variab='Functional')

inputing(variab='MSZoning')

inputing(variab='SaleType')

inputing(variab='Alley')

inputing(variab='BsmtExposure')

inputing(variab='BsmtFinType1')

inputing(variab='BsmtFinType2')

inputing(variab='Fence')



inputingnum(variab='KitchenQual')

data['KitchenQual'] = data.KitchenQual.astype(int)

inputingnum(variab='BsmtFullBath')

data['BsmtFullBath'] = data.BsmtFullBath.astype(int)

inputingnum(variab='BsmtHalfBath')

data['BsmtHalfBath'] = data.BsmtHalfBath.astype(int)



inputingnum(variab='TotalBsmtSF')

inputingnum(variab='BsmtFinSF1')

inputingnum(variab='BsmtFinSF2')

inputingnum(variab='MasVnrArea')

inputingnum(variab='BsmtUnfSF')

inputingnum(variab='LotFrontage')
print(data['Utilities'].value_counts())

data  = data.drop(['Utilities'],axis=1)
data.columns[data.isnull().sum()>0]
data.describe()
from scipy.stats import norm

plt.figure(figsize=(15,8))

sns.distplot(data['SalePrice'][data.SalePrice.isnull()==False], fit= norm,kde=True)

plt.show()
print(data.plot.scatter(x='LotFrontage',y='SalePrice'))
def dropping_outliers(data, condition):

    #put condition with with reference to the data table, use brackets and (& |) operators, remember about you can drop observation only from train dataset

    condition_to_drop = data[condition].index

    data = data.drop(condition_to_drop)
dropping_outliers(data, (data.SalePrice<100000) & (data.train==1) & (data.LotFrontage>150))

dropping_outliers(data, (data.LotFrontage>200) & (data.train==1))

dropping_outliers(data, (data.SalePrice>700000) & (data.train==1))

dropping_outliers(data, (data.SalePrice>700000) & (data.train==1))

dropping_outliers(data, (data.LotArea>60000) & (data.train==1))

dropping_outliers(data, (data.MasVnrArea>1450) & (data.train==1))

dropping_outliers(data, (data.BedroomAbvGr==8) & (data.train==1))

dropping_outliers(data, (data.KitchenAbvGr==3) & (data.train==1))

dropping_outliers(data, (data['3SsnPorch']>400) & (data.train==1))

dropping_outliers(data, (data.LotArea>100000) & (data.train==1))

dropping_outliers(data, (data.MasVnrArea>1300) & (data.train==1))

dropping_outliers(data, (data.BsmtFinSF1>2000) & (data.train==1) & (data.SalePrice<300000))

dropping_outliers(data, (data.BsmtFinSF2>200) & (data.SalePrice>350000)  & (data.train==1))

dropping_outliers(data, (data.BedroomAbvGr==8) & (data.train==1))

dropping_outliers(data, (data.KitchenAbvGr==3) & (data.train==1))

dropping_outliers(data, (data.TotRmsAbvGrd==2) & (data.train==1))
# c=data[(data['SalePrice']<100000) & (data.train==1) & (data['LotFrontage']>150)].index

# data = data.drop(c)

# c=data[(data['LotFrontage']>200) & (data.train==1)].index

# data = data.drop(c)

# c=data[(data['SalePrice']>700000) & (data.train==1)].index

# data = data.drop(c)

# c = data[(data['SalePrice']>700000) & (data.train==1)].index

# data = data.drop(c)

# c = data[(data['LotArea']>60000) & (data.train==1)].index

# data = data.drop(c)

# c = data[(data['MasVnrArea']>1450) & (data.train==1)].index

# data = data.drop(c)

# c = data[(data['BedroomAbvGr']==8) & (data.train==1)].index

# data = data.drop(c)

# c = data[(data['KitchenAbvGr']==3) & (data.train==1)].index

# data = data.drop(c)

# c = data[(data['3SsnPorch']>400) & (data.train==1)].index

# data = data.drop(c)

# c=data[(data.LotArea>100000) & (data.train==1)].index

# data = data.drop(c)

# c=data[(data.MasVnrArea>1300) & (data.train==1)].index

# data = data.drop(c)

# c=data[(data.BsmtFinSF1>2000) & (data.train==1) & (data.SalePrice<300000)].index

# data = data.drop(c)

# c=data[(data.BsmtFinSF2>200) & (data.SalePrice>350000)  & (data.train==1)].index

# data = data.drop(c)

# c=data[(data.BedroomAbvGr==8) & (data.train==1)].index

# data = data.drop(c)

# c=data[(data.KitchenAbvGr==3) & (data.train==1)].index

# data = data.drop(c)

# c=data[(data.TotRmsAbvGrd==2) & (data.train==1)].index

# data = data.drop(c)
#CentralAir

print(data['CentralAir'].value_counts())

data['CentralAir'] = pd.Series(np.where(data['CentralAir'].values == 'Y', 1, 0),

          data.index)
data['2ndFloor'] = pd.Series(np.where(data['2ndFlrSF'].values == 0, 0, 1),data.index)

data['Floors'] = data['1stFlrSF'] + data['2ndFlrSF']

data = data.drop(['1stFlrSF'],axis=1)

data = data.drop(['2ndFlrSF'],axis=1)

data['TotBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])

data['Porch'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch']

data['TotalSF'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['Floors'] 

data['Pool'] = pd.Series(np.where(data['PoolArea'].values == 0, 0, 1),data.index)

data['Bsmt'] = pd.Series(np.where(data['TotalBsmtSF'].values == 0, 0, 1),data.index)

data['Garage'] = pd.Series(np.where(data['GarageArea'].values == 0, 0, 1),data.index)

data['Fireplace'] = pd.Series(np.where(data['Fireplaces'].values == 0, 0, 1),data.index)

data['Remod'] = pd.Series(np.where(data['YearBuilt'].values == data['YearRemodAdd'].values, 0, 1),data.index)

data['NewHouse'] = pd.Series(np.where(data['YearBuilt'].values == data['YrSold'].values, 1, 0),data.index)

data['Age'] = data['YrSold'] - data['YearRemodAdd']
c = data[(data['Floors']>4000) & (data.train==1)].index

data = data.drop(c)

c = data[(data['SalePrice']>500000) & (data['TotalSF']<3500) & (data.train==1)].index

data = data.drop(c)
data = data.drop(['PoolQC'],axis=1)

data = data.drop(['GrLivArea'],axis=1)

data = data.drop(['Street'],axis=1)

data = data.drop(['GarageYrBlt'],axis=1)

data = data.drop(['PoolArea'],axis=1)

data = data.drop(['MiscFeature'],axis=1)
Results = pd.DataFrame({'Model': [],'Accuracy Score': []})
data = pd.get_dummies(data)
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



model = XGBRegressor(learning_rate=0.001,n_estimators=4600,

                                max_depth=7, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

model.fit(trainX,trainY)

y_pred = model.predict(testX)

y_pred = np.exp(y_pred)



res = pd.DataFrame({"Model":['XGBoost'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



model = DecisionTreeRegressor(max_depth=6)

model.fit(trainX,trainY)

y_pred = model.predict(testX)

y_pred = np.exp(y_pred)



print(rmsle(testY, y_pred))



res = pd.DataFrame({"Model":['Decision Tree'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



model = RandomForestRegressor(n_estimators=1500,

                                max_depth=6)

model.fit(trainX,trainY)

y_pred = model.predict(testX)

y_pred = np.exp(y_pred)

print(rmsle(testY, y_pred))



res = pd.DataFrame({"Model":['Random Forest'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



model = Lasso(alpha=0.0005)



model.fit(trainX,trainY)

y_pred = model.predict(testX)

y_pred = np.exp(y_pred)

print(rmsle(testY, y_pred))



res = pd.DataFrame({"Model":['LASSO'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
import statsmodels.api as sm

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



X2 = sm.add_constant(trainX)

o=0

for i in X2.columns:

    o+=1

    print(o)

    model = sm.OLS(trainY, X2.astype(float))

    model = model.fit()

    p_values = pd.DataFrame(model.pvalues)

    p_values = p_values.sort_values(by=0, ascending=False)

    if float(p_values.loc[p_values.index[0]])>=0.05:

        X2=X2.drop(p_values.index[0],axis=1)

    else:

        break



kolumny = X2.columns

testX = sm.add_constant(testX)

testX = testX[kolumny]



y_pred = model.predict(testX)

y_pred = np.exp(y_pred)





res = pd.DataFrame({"Model":['Stepwise Regression'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



model = Ridge(alpha=0.0005)



model.fit(trainX,trainY)

y_pred = model.predict(testX)

y_pred = np.exp(y_pred)

print(rmsle(testY, y_pred))



res = pd.DataFrame({"Model":['Ridge'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)

trainY = np.log(trainY)



model = Lasso(alpha=0)



model.fit(trainX,trainY)

y_pred = model.predict(testX)

y_pred = np.exp(y_pred)

print(rmsle(testY, y_pred))



res = pd.DataFrame({"Model":['Linear Regression'],

                    "Accuracy Score": [rmsle(testY, y_pred)]})

Results = Results.append(res)
Results
trainX = data[data.SalePrice.isnull()==False].drop(['SalePrice','train'],axis=1)

trainY = data.SalePrice[data.SalePrice.isnull()==False]

testX = data[data.SalePrice.isnull()==True].drop(['SalePrice','train'],axis=1)

trainY = np.log(trainY)

model = Lasso(alpha=0.0005)

model.fit(trainX, trainY)

test = data[data.train==0]

test['SalePrice'] = model.predict(testX)

test['SalePrice'] = np.exp(test['SalePrice'] )

test = test.reset_index()

test[['Id','SalePrice']].to_csv("submissionLASSO.csv",index=False)

print("done1")