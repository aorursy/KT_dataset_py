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
from scipy import stats

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid') # just optional!

%matplotlib inline



#Setting display format to retina in matplotlib to see better quality images.

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')



# Lines below are just to ignore warnings

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.head()
df_train.describe()
df_test.describe()
sns.distplot(df_train['SalePrice'])
df_train.info()
df_test.info()
df =pd.DataFrame()

df['null'] =  pd.Series(df_train.isnull().sum())

df = df[df.null != 0]

df.sort_values(by= 'null',ascending= False )
f, ax = plt.subplots(figsize=(8,5))

plt.xticks(rotation='90')

sns.barplot(x=df.index, y=df['null'])

plt.xlabel('Features')

plt.ylabel('Number of missing values')

plt.title('Number missing data by feature')
# Replacing missing values with None 

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType','PoolQC','MiscFeature','FireplaceQu'

            ,'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass','Alley','Fence'):

    df_train[col] = df_train[col].fillna('None')

#

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType','PoolQC','MiscFeature','FireplaceQu'

            ,'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass','Alley','Fence'):

    df_test[col] = df_test[col].fillna('None')
#Replacing missing values with 0 

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'MasVnrArea',

            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_train[col] = df_train[col].fillna(0)

#

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1','MasVnrArea', 

            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_test[col] = df_test[col].fillna(0)
#Replacing missing values with median 

df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#

df_test["LotFrontage"] = df_test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#Functional

df_train["Functional"] = df_train["Functional"].fillna("Typ")

#

df_test["Functional"] = df_test["Functional"].fillna("Typ")
print (df_train['Utilities'].value_counts())

print (df_test['Utilities'].value_counts())
#Drop utilities

df_train = df_train.drop(['Utilities'], axis=1)

#

df_test = df_test.drop(['Utilities'], axis=1)
# Replacing missing values with mode 

for col in ('SaleType','Exterior2nd','Exterior1st','KitchenQual','Electrical','MSZoning'):

    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

    

for col in ('SaleType','Exterior2nd','Exterior1st','KitchenQual','Electrical','MSZoning'):

    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
#Recheck remaining missing values if any

df =pd.DataFrame()

df['null'] =  pd.Series(df_train.isnull().sum())

df = df[df.null != 0]

df
print("The train data size before dropping Id feature is : {} ".format(df_train.shape))

print("The test data size before dropping Id feature is : {} ".format(df_test.shape))



#Save the 'Id' column

train_ID = df_train['Id']

test_ID = df_test['Id']



#dropping Id 

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(df_train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(df_test.shape))
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True,cmap='GnBu')
#saleprice correlation matrix

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap='GnBu')

plt.show()
categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

li_cat_feats = list(categorical_feats)

nr_rows = 15

nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):

    for c in range(0,nr_cols):

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y='SalePrice', data=df_train, ax = axs[r][c])

plt.tight_layout()

plt.show()
atg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',

                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType','FireplaceQu', 'GarageQual']

num_Strong_corr = [cols]
for df in [df_train, df_test]:

    df['MSZ_num'] = 1

    df.loc[(df['MSZoning']=='RH' ), 'MSZ_num'] = 2

    df.loc[(df['MSZoning']== 'RM' ), 'MSZ_num'] = 3

    df.loc[(df['MSZoning']== 'RL' ), 'MSZ_num'] = 4

    df.loc[(df['MSZoning']=='FV' ), 'MSZ_num'] = 5



    

for df in [df_train, df_test]:

    df['NbHd_num'] = 1

    df.loc[(df['Neighborhood']== 'Blmngtn'), 'NbHd_num'] = 2

    df.loc[(df['Neighborhood']==  'ClearCr'), 'NbHd_num'] = 3

    df.loc[(df['Neighborhood']==  'CollgCr'), 'NbHd_num'] = 4

    df.loc[(df['Neighborhood']==  'Crawfor'), 'NbHd_num'] = 5

    df.loc[(df['Neighborhood']==   'Gilbert'), 'NbHd_num'] = 6

    df.loc[(df['Neighborhood']==  'NWAmes'), 'NbHd_num'] = 7

    df.loc[(df['Neighborhood']==  'Somerst'), 'NbHd_num'] = 8

    df.loc[(df['Neighborhood']==  'Timber'), 'NbHd_num'] = 9

    df.loc[(df['Neighborhood']==  'Veenker'), 'NbHd_num'] = 10

    df.loc[(df['Neighborhood']== 'NoRidge' ), 'NbHd_num'] = 11

    df.loc[(df['Neighborhood']== 'NridgHt' ), 'NbHd_num'] = 12

    df.loc[(df['Neighborhood']== 'StoneBr' ), 'NbHd_num'] = 13





for df in [df_train, df_test]:

    df['Cond2_num'] = 1

    df.loc[(df['Condition2']=='Norm' ), 'Cond2_num'] = 2

    df.loc[(df['Condition2']=='RRAe' ), 'Cond2_num'] = 3

    df.loc[(df['Condition2']=='PosA'), 'Cond2_num'] = 4

    df.loc[(df['Condition2']=='PosN'), 'Cond2_num'] = 5



for df in [df_train, df_test]:

    df['Mas_num'] = 1

    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2

    

for df in [df_train, df_test]:

    df['ExtQ_num'] = 1

    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2

    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3

    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4

    

for df in [df_train, df_test]:

    df['BsQ_num'] = 1

    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2

    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3

    

for df in [df_train, df_test]:

    df['CA_num'] = 0

    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1

    

for df in [df_train, df_test]:

    df['Elc_num'] = 1

    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2

    

for df in [df_train, df_test]:

    df['KiQ_num'] = 1

    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2

    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3

    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4

    

for df in [df_train, df_test]:

    df['SlTy_num'] = 1

    df.loc[(df['SaleType']=='Oth'), 'SlTy_num'] = 2

    df.loc[(df['SaleType']=='CWD'), 'SlTy_num'] = 3

    df.loc[(df['SaleType']=='Con' ), 'SlTy_num'] = 4

    df.loc[(df['SaleType']=='New' ), 'SlTy_num'] = 5

    



for df in [df_train, df_test]:

    df['FireplaceQu_num'] = 1

    df.loc[(df['FireplaceQu']=='Gd'), 'FireplaceQu_num'] = 2

    df.loc[(df['FireplaceQu']=='TA'), 'FireplaceQu_num'] = 3

    df.loc[(df['FireplaceQu']=='Fa' ), 'FireplaceQu_num'] = 4

    df.loc[(df['FireplaceQu']=='Ex' ), 'FireplaceQu_num'] = 5

    df.loc[(df['FireplaceQu']=='Po' ), 'FireplaceQu_num'] = 6

    

for df in [df_train, df_test]:

    df['GarageQual_num'] = 1

    df.loc[(df['GarageQual']=='None'), 'GarageQual_num'] = 2

    df.loc[(df['GarageQual']=='Fa'), 'GarageQual_num'] = 3

    df.loc[(df['GarageQual']=='Gd' ), 'GarageQual_num'] = 4

    df.loc[(df['GarageQual']=='Po' ), 'GarageQual_num'] = 5

    df.loc[(df['GarageQual']=='Ex' ), 'GarageQual_num'] = 6
isNumeric_train = df_train._get_numeric_data()

isNumeric_test = df_test._get_numeric_data()
corrmat = isNumeric_train.corr()

f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(corrmat, vmax=.8,square=True,cmap='GnBu')

# cbar=True, annot=True, square=True


k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(isNumeric_train[cols].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap='GnBu')

plt.show()
#defind outliers

fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.title('')

plt.show()
#Deleting outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#function to create csv file

def create_csv_predictions(predictions, file_name):

    

    predictions=pd.DataFrame(predictions)

    predictions['Id']= [i for i in range(1461, 2920)]

    predictions.set_index('Id',inplace=True)

    predictions.rename(columns={0:'SalePrice'},inplace = True)

    predictions.to_csv(file_name)
from sklearn.preprocessing import StandardScaler
X_train= isNumeric_train[['OverallQual','GrLivArea','ExtQ_num','NbHd_num','KiQ_num','BsQ_num','TotalBsmtSF','GarageCars','1stFlrSF','GarageArea','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]

y_train= isNumeric_train['SalePrice']
X_test= isNumeric_test[['OverallQual','GrLivArea','ExtQ_num','NbHd_num','KiQ_num','BsQ_num','TotalBsmtSF','GarageCars','1stFlrSF','GarageArea','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
Model= []

R_s_training = []

mean_cv_s_5 = []
#find the score of beasline



# create a DummyRegressor model instance

dummy_constant = DummyRegressor(strategy='constant', constant=y_train.mean())



# "Train" dummy regressor

dummy_constant.fit(X_train, y_train)

dummy_constant.score(X_train, y_train)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV

from sklearn.model_selection import train_test_split, cross_val_score



# create a linear regression model instance

model = LinearRegression()



# get cross validated scores

scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



#fit and evaluate the data on the whole training set

model.fit(X_train, y_train)



print("Training Score:", model.score(X_train, y_train))
predictions= model.predict(X_test)

create_csv_predictions(predictions, 'predictions.csv')
df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')


# Replacing missing values with None 

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType','PoolQC','MiscFeature','FireplaceQu'

            ,'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass','Alley','Fence'):

    df_train[col] = df_train[col].fillna('None')

#

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType','PoolQC','MiscFeature','FireplaceQu'

            ,'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass','Alley','Fence'):

    df_test[col] = df_test[col].fillna('None')



#Replacing missing values with 0 

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'MasVnrArea',

            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_train[col] = df_train[col].fillna(0)

#

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1','MasVnrArea', 

            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_test[col] = df_test[col].fillna(0)



#Replacing missing values with median 

df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#

df_test["LotFrontage"] = df_test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))





#Functional

df_train["Functional"] = df_train["Functional"].fillna("Typ")

#

df_test["Functional"] = df_test["Functional"].fillna("Typ")





#Drop utilities

df_train = df_train.drop(['Utilities'], axis=1)

#

df_test = df_test.drop(['Utilities'], axis=1)

#

for col in ('SaleType','Exterior2nd','Exterior1st','KitchenQual','Electrical','MSZoning'):

    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

    

for col in ('SaleType','Exterior2nd','Exterior1st','KitchenQual','Electrical','MSZoning'):

    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
isNumeric_train = df_train._get_numeric_data()

isNumeric_test = df_test._get_numeric_data()
#creat dummies for categorical feats

categorical_feats = list(df_train.dtypes[df_train.dtypes == "object"].index)

categorical_train= df_train[categorical_feats]

categorical_test= df_test[categorical_feats]

dummies_train = pd.get_dummies(categorical_train)

dummies_test = pd.get_dummies(categorical_test)
new_df_train= pd.concat([isNumeric_train,dummies_train], axis=1)

new_df_test= pd.concat([isNumeric_test,dummies_test], axis=1)
X_train= new_df_train[[c for c in new_df_train if c != 'SalePrice']]

y_train= new_df_train['SalePrice']

X_test= new_df_test[[c for c in new_df_test]]

missing=[]

for mis in X_train.columns:

    if mis not in X_test.columns:

        missing.append(mis)



missing
for col in missing:

    X_test[col]= np.nan

    

X_test.fillna(0, inplace=True)

X_test=X_test[X_train.columns]
#models evaluation df

evaluation = pd.DataFrame({'Model': [],

                          'R-squared (training)':[],

                          '5-Fold Cross Validation':[]})
from sklearn.preprocessing import MinMaxScaler

from sklearn.dummy import DummyRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV







scaler = MinMaxScaler(feature_range=(0, 1))



x_train_scaled = scaler.fit_transform(X_train)

X_train = pd.DataFrame(x_train_scaled)



x_test_scaled = scaler.fit_transform(X_test)

X_test = pd.DataFrame(x_test_scaled)
#gridsearch params

dtc_params = {

    'max_depth': range(1,30),

    'max_features': [None, 'log2', 'sqrt'],

    'min_samples_split': range(5,20),

    'max_leaf_nodes': [None],

    'min_samples_leaf': range(1,10)

}





# set the gridsearch

model_dtc = DecisionTreeRegressor()

model_dtc_gs = GridSearchCV(estimator=model_dtc,param_grid=dtc_params,cv=5, error_score='raise-deprecating', n_jobs=-1, verbose=1)



# fit the model

model_dtc_gs=model_dtc_gs.fit(X_train,y_train)
predictions= model_dtc_gs.best_estimator_.predict(X_test)

create_csv_predictions(predictions, 'predictions_dtc_gs.csv')
# evaluate on the training set

print('Training score:', model_dtc_gs.best_score_)



evaluation = evaluation.append({'Model' : 'dtc' ,

                                'R-squared (training)' : model_dtc_gs.best_score_, 

                                '5-Fold Cross Validation':''} ,ignore_index=True)
#import required packages

from sklearn import neighbors

from sklearn.metrics import mean_squared_error 

from math import sqrt

import matplotlib.pyplot as plt

%matplotlib inline



# gridsearch params

tuned_parameters = [{'weights': ['uniform', 'distance'],

                    'n_neighbors': range(2,100)}]

# create a GridSearchCV model instance

model_KNN = GridSearchCV(neighbors.KNeighborsRegressor(), tuned_parameters, cv=5)
# fit the model

model_KNN=model_KNN.fit(X_train,y_train)
# evaluate on the training set

print('Training score:', model_KNN.best_score_)



evaluation = evaluation.append({'Model' : 'KNN' ,

                                'R-squared (training)' : model_KNN.best_score_, 

                                '5-Fold Cross Validation':''} ,ignore_index=True)
predictions= model_KNN.best_estimator_.predict(X_test)

create_csv_predictions(predictions, 'predictions_KNN.csv')
# create a LassoCV model instance

model_LassoCV = LassoCV(alphas=np.logspace(-50,100, 100), cv=5)



# fit the model

model_LassoCV.fit(X_train, y_train)



# get the best alpha

print('Best alpha:', model_LassoCV.alpha_)



# evaluate on the training set

print('Training score:', model_LassoCV.score(X_train, y_train))



evaluation = evaluation.append({'Model' : 'LassoCV' ,

                                'R-squared (training)' : model_LassoCV.score(X_train, y_train), 

                                '5-Fold Cross Validation':''} ,ignore_index=True)
# create a Lasso model instance

model_lasso = Lasso(alpha=80)



# get cross validated scores

scores = cross_val_score(model_lasso, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



# fit and evaluate the data on the whole training set

model_lasso.fit(X_train, y_train)

print("Training Score:", model_lasso.score(X_train, y_train))



evaluation = evaluation.append({'Model' : 'Lasso' ,

                                'R-squared (training)' : model_lasso.score(X_train, y_train), 

                                '5-Fold Cross Validation': scores.mean()} ,ignore_index=True)
# collect the model coefficients in a dataframe

df_coef = pd.DataFrame(model_lasso.coef_, index=X_train.columns,

                       columns=['coefficients'])



# calculate the absolute values of the coefficients

df_coef['coef_abs'] = df_coef.coefficients.abs()

df_coef.head()
predictions=model_lasso.predict(X_test)

create_csv_predictions(predictions, 'predictions_lasso80.csv')
# create a RidgeCV model instance

model_RidgeCV = RidgeCV(alphas=np.logspace(-1, 2, 10), cv=5)



# fit the model

model_RidgeCV.fit(X_train, y_train)



# get the best alpha

print('Best alpha:', model_RidgeCV.alpha_)



# evaluate on the training set

print('Training score:', model_RidgeCV.score(X_train, y_train))



evaluation = evaluation.append({'Model' : 'RidgeCV' ,

                                'R-squared (training)' : model_RidgeCV.score(X_train, y_train), 

                                '5-Fold Cross Validation':''} ,ignore_index=True)
# create a Ridge model instance

model_Ridge = Ridge(alpha=1000)



# get cross validated scores

scores = cross_val_score(model_Ridge, X_train, y_train, cv=5)

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



# fit and evaluate the data on the whole training set

model_Ridge.fit(X_train, y_train)

print("Training Score:", model_Ridge.score(X_train, y_train))



evaluation = evaluation.append({'Model' : 'Ridge' ,

                                'R-squared (training)' : model_Ridge.score(X_train, y_train), 

                                '5-Fold Cross Validation': scores.mean()} ,ignore_index=True)
# collect the model coefficients in a dataframe

df_coef = pd.DataFrame(model_Ridge.coef_, index=X_train.columns,

                       columns=['coefficients'])



# calculate the absolute values of the coefficients

df_coef['coef_abs'] = df_coef.coefficients.abs()

df_coef.head()
predictions=model_Ridge.predict(X_test)

create_csv_predictions(predictions, 'predictions_model_Ridge_1000.csv')
predictions=pd.DataFrame(predictions)

predictions['Id']= [i for i in range(1461, 2920)]

predictions.set_index('Id')

predictions.rename(columns={'0':'SalePrice'},inplace = True)
predictions.rename(columns={0:'SalePrice'},inplace = True)
predictions.set_index('Id',inplace=True)

predictions.head()
evaluation
corrmat = new_df_train.corr()

k = 11

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(new_df_train[cols].values.T)



l = list( cm[0])

l

col = cols[1:11]

col

l2 =l[1:11]

l2



f, ax = plt.subplots(figsize=(8,5))

plt.xticks(rotation='90')

sns.barplot(x= col, y=l2)

plt.xlabel('Features')

plt.title('Strongest Features')

f, ax = plt.subplots(figsize=(8,5))

plt.xticks(rotation='90')

sns.barplot(x=evaluation['Model'], y=evaluation['R-squared (training)'])

plt.xlabel('Models')

plt.ylabel('R-squared')

plt.title('R-squared for each model')
