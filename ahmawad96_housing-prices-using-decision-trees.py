import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

import seaborn as sns

from matplotlib import pyplot as plt

print("Libraries imported!")

train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
targets=train_data["SalePrice"]

targets.head()
corrmat = train_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
train_data.drop(['Id','SalePrice'],inplace=True, axis=1)

test_ids=test_data['Id']

test_data.drop(['Id'],inplace=True, axis=1)

total_data=pd.concat((train_data, test_data)).reset_index(drop=True)

null_columns=total_data.columns[total_data.isnull().any()]

total_data[null_columns].isnull().sum()
total_data.info()
sns.countplot(total_data.MSZoning)
total_data=total_data.fillna({'MSZoning':'RL'})

print(total_data['MSZoning'].isnull().sum())
total_data['LotFrontage']=total_data.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.median()))

print(total_data['LotFrontage'].isnull().sum())
total_data.head(10)
total_data=total_data.fillna({"Alley":"NoAlley"})

sns.countplot(total_data.Alley)
sns.countplot(total_data.Utilities)
total_data=total_data.fillna({"Utilities":"AllPub"})

print(total_data['LotFrontage'].isnull().sum())
total_data['Exterior1st']=total_data['Exterior1st'].fillna(total_data['Exterior1st'].mode()[0])

total_data['Exterior2nd']=total_data['Exterior2nd'].fillna(total_data['Exterior2nd'].mode()[0])

print(total_data['Exterior1st'].isnull().sum())

temp_data=total_data['MasVnrArea'][total_data.MasVnrType.isnull()]

temp_data.head(26)
total_data.MasVnrType=total_data.MasVnrType.fillna("None")

total_data.MasVnrArea=total_data.MasVnrArea.fillna(0)

print(total_data['MasVnrArea'].isnull().sum())

cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

for col in cols:

    total_data[col]=total_data[col].fillna("NoBsmt")

    print(total_data[col].isnull().sum())
temp_data=total_data[total_data.BsmtFinSF1.isnull()]

#temp_data=temp_data['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

temp_data.head()
null_columns=total_data.columns[total_data.isnull().any()]

total_data[null_columns].isnull().sum()
cols=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']



for col in cols:

    total_data[col]=total_data[col].fillna(0)

    print(total_data[col].isnull().sum())
total_data["Electrical"]=total_data["Electrical"].fillna(total_data["Electrical"].mode()[0])

total_data['Electrical'].isnull().sum()
temp_data=total_data[total_data.BsmtFullBath.isnull()]

temp_data.head()
cols=['BsmtFullBath','BsmtHalfBath']



for col in cols:

    total_data[col]=total_data[col].fillna(0)

    print(total_data[col].isnull().sum())
cols=['KitchenQual','Functional']

for col in cols:

    total_data[col]=total_data[col].fillna(total_data[col].mode()[0])

    print(total_data[col].isnull().sum())

cols=['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']

for col in cols:

    total_data[col]=total_data[col].fillna("None")
null_columns=total_data.columns[total_data.isnull().any()]

total_data[null_columns].isnull().sum()
for col in ['GarageCars', 'GarageArea']:

    total_data[col]=total_data[col].fillna(0)
for col in ['PoolQC','Fence','MiscFeature']:

    total_data[col]=total_data[col].fillna("None")

    

total_data['SaleType']=total_data['SaleType'].fillna(total_data['SaleType'].mode()[0])
null_columns=total_data.columns[total_data.isnull().any()]

total_data[null_columns].isnull().sum()
total_data=pd.get_dummies(total_data)
train_indx=train_data.shape[0]

train_data=total_data[:train_indx]

test_data=total_data[train_indx:]

train_data.shape
from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from math import sqrt

def rmse(y,predicts):

    score=sqrt(mean_squared_error(np.log(y),np.log(predicts)))

    return score



def fit_model(X, y):

   

    cv_sets = KFold(n_splits=10, random_state=42)

    regressor = DecisionTreeRegressor(random_state=0)

    params = {"max_depth":[4,8,10,11,12,14,16,20],'min_samples_split':[20,40,60,80,100,200],'min_samples_leaf':[10,15,20,25,40]}

    scoring_fnc = make_scorer(rmse,greater_is_better=False)

    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)

    results=grid.cv_results_

    return results, grid.best_estimator_
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(train_data,targets,test_size=0.2,random_state=42, shuffle=True)
result, reg = fit_model(X_train, y_train)

print(reg.get_params())
report=pd.DataFrame(result)

report.head(30)
y_train_pred=reg.predict(X_train)

y_test_pred=reg.predict(X_test)

test_loss=rmse(y_test,y_test_pred)

train_loss=rmse(y_train,y_train_pred)

print('The training loss is', train_loss)

print('The test loss is', test_loss)
submission=pd.DataFrame({'Id':[],'SalePrice':[]})

sub_predicts=reg.predict(test_data)

submission.Id=test_ids

submission.SalePrice=sub_predicts

submission.head()


submission.to_csv("sub3.csv",index=False)
submission.shape