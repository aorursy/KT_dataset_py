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
# loading dataset

data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print(data.head(10))

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')





target=data['SalePrice']

print(target)

train=pd.DataFrame(data)
# importing requirements

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import sklearn.linear_model as linear_model

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
print(train.shape)

print(test.shape)

# EDA

x=train['LotArea']

z=train['MSSubClass']



y=train['SalePrice']



plt.scatter(x,y,cmap='r')

plt.xlabel('LotArea')

plt.ylabel('SalePrice')

plt.xlim(0, 20000)

plt.show()

sns.distplot(x)


# To find the correlation among 

# the columns using pearson method 

corr_matrix = train.corr(method ='pearson').abs()

print(corr_matrix)


# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.90

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



train.drop(to_drop,inplace=True,axis=1)
# heatmap

import seaborn as sns

plt.figure(figsize=(12,12), dpi= 80)

sns.heatmap(train.corr(), cmap='RdYlGn')



# Decorations

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
# manual highlighting imp corr features wrt sales price

'''OverallQual

1stFlrSF

TotalBsmtSF

YearBuilt

GarageArea

GarageCars

LotArea

MSSubClass'''
# multiple scatter plots

y_columns=['LotArea','OverallQual',

'TotalBsmtSF',

'YearBuilt',

'GarageCars','MSSubClass']

for y_col in y_columns:



    figure = plt.figure

    ax = plt.gca()

    ax.scatter(train['SalePrice'], train[y_col])

    ax.set_xlabel('SalePrice')

    ax.set_ylabel(y_col)

    ax.set_title("SalePrice vs {}".format( y_col))



    plt.legend()

    plt.show()

# displayong total null values

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# columns where NaN values have meaning e.g. no pool etc.

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2'

               ,'PoolQC','MiscFeature']



# replace 'NaN' with 0 in these columns

for col in cols_fillna:

    train[col].fillna(0,inplace=True)

    test[col].fillna(0,inplace=True)

    #train[col] = train[col].apply(lambda x: 1 if not pd.isnull(x) else np.nan)


# handling null values , replacing with mean

x=train.isna().sum()

print(x.head(20))

train['LotFrontage'].fillna(train['LotFrontage'].mean(),inplace=True)

train['Alley']=train['Alley'].replace(np.nan,0)

train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(),inplace=True)

train['MasVnrArea'].fillna(train['MasVnrArea'].mean(),inplace=True)

print(train['Alley'])

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
# total null values are zero

print(train.isna().sum().sum())

# No of observations

print(train.shape[0])

# No of features

print(train.shape[1])

print(data.columns)

print(data.dtypes)

print(data.head(10))
# finding categorical columns

categorical_data = [var for var in train.columns if train[var].dtype=='O']

categorical_features=pd.DataFrame(categorical_data)

print("Number of Categorical features: ", len(categorical_features))

print(categorical_features.head(40))


# process columns, apply LabelEncoder to categorical features

for c in categorical_data:

    lbl = LabelEncoder() 

    lbl.fit(list(train[c].values)) 

    train[c] = lbl.transform(list(train[c].values))

    lbl.fit(list(test[c].values)) 

    test[c] = lbl.transform(list(test[c].values))
# standardizing data

sc=StandardScaler()

sc.fit_transform(train)
y = np.log(train.SalePrice)

X = train.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Lasso,LassoCV

import xgboost as xgb

from sklearn.model_selection import cross_val_score,GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



#  linear Regression 

lr=LinearRegression()

lr.fit(X_train,y_train)



y_pred=lr.predict(X_test)

x=lr.score(X_train,y_train)

print("Linear Reg score is ",x*100)

#*** 90.6% accuracy ***



# LassoRegression

params={"eps":[0.001,0.01,0.0001],"cv":[3,5,7],"n_alphas":[0.001,0.005,0.01]}



lasreg=LassoCV( eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, max_iter=1000, tol=0.0001, cv=5, verbose=True, selection='cyclic')

lasreg.fit(X_train,y_train)

#lasreg.score(X_train,y_train)

scores = cross_val_score(lasreg, X_train, y_train, cv=5)

print(scores)

lasreg.get_params()
# label encoding labels

from sklearn import utils

lab_enc = LabelEncoder()

training_scores_encoded = lab_enc.fit_transform(y_train)

print(training_scores_encoded)

print(utils.multiclass.type_of_target(y_train))

print(utils.multiclass.type_of_target(y_train.astype('int')))

print(utils.multiclass.type_of_target(training_scores_encoded))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

# logistic Regression

# Instantiate a logistic regression 

las=Lasso()



'''

                                    ***for logreg ***

logreg =LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=1000)

c_grid=np.logspace(0, 4, 10)

logreg.fit(X_train,training_scores_encoded)

preds = logreg.predict(X_test) 

mean_squared_error(y_test,preds)'''



param_grid={"max_iter":[100,200,1000],"alpha":[1,0.8,0.5]}

logreg_cv = GridSearchCV(las,param_grid, cv=3)

    

# Fit it to the data

logreg_cv.fit(X_train,training_scores_encoded)





print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 

print("Best score is {}".format(logreg_cv.best_score_))

# using gradient boosting 



import xgboost as xgb

from sklearn.metrics import r2_score,mean_squared_error

xg_reg = xgb.XGBRegressor(learning_rate =0.01, n_estimators=3460, max_depth=3,min_child_weight=0 , gamma=0, subsample=0.7, colsample_bytree=0.7, objective= 'reg:linear', reg_alpha=0.00006)

xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test) 

print('The accuracy of the xgboost is',r2_score(y_test,preds)*100) 

print ('RMSE is: \n', mean_squared_error(y_test,preds))

print(preds)

#*** 91% accuracy ***

# *** RMSE 0.013 ***
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha':[1e-10,1e-8,1e-6,1e-4,1e-2,1,5,10,20,30,40,45,50,55,60]}

reg_r=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv = 5)

reg_r.fit(X_train,y_train)

# showing best parametrs and estimator

print(reg_r.best_estimator_)

print(reg_r.best_score_)



# using best estimator obtained from gridsearch and predicting

best=reg_r.best_estimator_

y_pred=best.predict(X_test)



x=best.score(X_test,y_test)

print(x)

#submitting XGB predictions

sub = pd.DataFrame()

my_submission = pd.DataFrame({ 'SalePrice': predicted_prices})

sub['SalePrice'] = np.expm1(preds_xg)

sub.to_csv('Kaggel_submission.csv',index=False)

print('DONE')