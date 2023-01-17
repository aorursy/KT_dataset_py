# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

import plotly.express as px

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor 

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')



backup_data=train_data
test_data.shape
train_data.shape
target_column=train_data[['SalePrice']]
train_data.shape
# Checking train data

train_data.head()
# Checking test data

test_data.head()
# Checking Sample submission data

sample_submission.head()
target_column.describe()
plt.figure(figsize=(10,10))

sns.distplot(target_column)
# As suggested by many participants, we remove several outliers

train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index, inplace=True)

train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index, inplace=True)

train_data.reset_index(drop=True, inplace=True)
def boxplot(column_name,data):

    title='Box plot for column- {0}'.format(column_name)

    #print(title)

    plt.title(title)

    sns.boxplot(x=column_name,data=data)

    details_na='The total number of NA Values for column - {1} is {0}' .format(data[column_name].isna().sum(),column_name)

    print(details_na)

    

def drop_column(data,column_name):

    train_data=data

    if column_name in data.columns:

        train_data=data.drop(columns=[column_name],axis=1)

        return train_data

    else:

        print('Column already removed from dataset Or column not present \n')

        print(' ,'.join (data.columns))

        return train_data

    

def remove_outliers(column_name,data):

    quantile_1=data[column_name].quantile(0.25)

    quantile_3=data[column_name].quantile(0.75)

    iqr=quantile_3-quantile_1

    lower=quantile_1-1.5*iqr

    upper=quantile_1+1.5*iqr

    data[column_name]=data[(data[column_name]>lower)& (data[column_name]<upper)][column_name]

    return data[column_name]
## Convert to category variable

## Dividing the list between object type variables and int variables and then subsetting it



obj_type_variables = [column for column in train_data.columns if train_data[column].dtype in ['object']]

object_data=train_data[obj_type_variables]

object_data_test=test_data[obj_type_variables]



obj_new_list=['MSSubClass','OverallQual','OverallCond','BsmtFullBath','GarageCars','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd'

              ,'Fireplaces','MoSold']

train_data[obj_new_list]=train_data[obj_new_list].apply(lambda column:column.astype('object'))

test_data[obj_new_list]=test_data[obj_new_list].apply(lambda column:column.astype('object'))



category_data=train_data[obj_new_list]

category_data_test=test_data[obj_new_list]



int_type_variables = [column for column in train_data.columns if train_data[column].dtype in ['int64','float64']]

int_data=train_data[int_type_variables]

int_data_test=test_data[int_type_variables[:-1]]
print(int_type_variables)
test_data[obj_new_list].isna().sum()
object_data_test.isna().sum()
object_data['BsmtExposure'].value_counts()

x=['BsmtHalfBath', 'BsmtFullBath', 'GarageCars']

for i in x:

    print(train_data[i].value_counts())



object_data['Alley']=object_data['Alley'].fillna("Grvl")

object_data['BsmtQual']=object_data['BsmtQual'].fillna("TA")

object_data['BsmtCond']=object_data['BsmtCond'].fillna("TA")

object_data['BsmtExposure']=object_data['BsmtExposure'].fillna("No")

object_data['BsmtFinType1']=object_data['BsmtFinType1'].fillna("Unf")

object_data['BsmtFinType2']=object_data['BsmtFinType2'].fillna("Unf")

object_data['Electrical']=object_data['Electrical'].fillna("SBrkr")

object_data['FireplaceQu']=object_data['FireplaceQu'].fillna("Gd")

object_data['GarageType']=object_data['GarageType'].fillna("Attchd")

object_data['GarageFinish']=object_data['GarageFinish'].fillna("Unf")

object_data['GarageQual']=object_data['GarageQual'].fillna("TA")

object_data['GarageCond']=object_data['GarageCond'].fillna("TA")

object_data['PoolQC']=object_data['GarageCond'].fillna("TA")

object_data['Fence']=object_data['Fence'].fillna("MnPrv")

object_data['MiscFeature']=object_data['MiscFeature'].fillna("Shed")



train_data['BsmtHalfBath']=train_data['BsmtHalfBath'].fillna("0")

train_data['BsmtFullBath']=train_data['BsmtFullBath'].fillna("0")

train_data['GarageCars']=train_data['GarageCars'].fillna("2")







object_data_test['Alley']=object_data_test['Alley'].fillna("Grvl")

object_data_test['BsmtQual']=object_data_test['BsmtQual'].fillna("TA")

object_data_test['BsmtCond']=object_data_test['BsmtCond'].fillna("TA")

object_data_test['BsmtExposure']=object_data_test['BsmtExposure'].fillna("No")

object_data_test['BsmtFinType1']=object_data_test['BsmtFinType1'].fillna("Unf")

object_data_test['BsmtFinType2']=object_data_test['BsmtFinType2'].fillna("Unf")

object_data_test['Electrical']=object_data_test['Electrical'].fillna("SBrkr")

object_data_test['FireplaceQu']=object_data_test['FireplaceQu'].fillna("Gd")

object_data_test['GarageType']=object_data_test['GarageType'].fillna("Attchd")

object_data_test['GarageFinish']=object_data_test['GarageFinish'].fillna("Unf")

object_data_test['GarageQual']=object_data_test['GarageQual'].fillna("TA")

object_data_test['GarageCond']=object_data_test['GarageCond'].fillna("TA")

object_data_test['PoolQC']=object_data_test['GarageCond'].fillna("TA")

object_data_test['Fence']=object_data_test['Fence'].fillna("MnPrv")

object_data_test['MiscFeature']=object_data_test['MiscFeature'].fillna("Shed")



test_data['BsmtHalfBath']=test_data['BsmtHalfBath'].fillna("0")

test_data['BsmtFullBath']=test_data['BsmtFullBath'].fillna("0")

test_data['GarageCars']=test_data['GarageCars'].fillna("2")





#object_data['MiscFeature'].value_counts()

#train_data_X_int=train_data_X_int.fillna(train_data_X_int.median())

test_data['GarageCars'].isna().sum()
object_data.isna().sum()
object_data.shape
boxplot('SalePrice',target_column)
corr_data=int_data.corr()
plt.figure(figsize=(30,30))

sns.heatmap(corr_data,annot=True)
pair_plot_data_inbetween_variables=train_data[['MSSubClass',

          'LotFrontage',

         'TotalBsmtSF',

         'OverallQual',

         'GrLivArea',

         'OverallCond',

         'YearBuilt',

         'EnclosedPorch',

         'GarageYrBlt',

         'YearRemodAdd',

         'BsmtFinSF1',

         'BsmtFullBath',

         'BsmtUnfSF',

         '1stFlrSF',

         '2ndFlrSF',

         'BedroomAbvGr',

         'Fireplaces',

         'GarageCars',

         'GarageArea'

         ]]



pair_plot_data_inbetween_variables.head()
#plt.figure(figsize=(30,30))

#sns.pairplot(pair_plot_data_inbetween_variables)
pair_plot_data_target_variable=train_data[[

    'SalePrice',

    'OverallQual',

    'TotalBsmtSF',

    '1stFlrSF',

    'GrLivArea',

    'GarageCars',

    'GarageArea'

]]
sns.pairplot(pair_plot_data_target_variable)
## Creating graphs between SalePrice and ['GarageArea','GrLivArea','1stFlrSF','TotalBsmtSF']



plt.figure(figsize=(9,9))

sns.relplot(x="SalePrice",

                y="GrLivArea",

                col="BldgType",

                hue="OverallQual",

                kind="scatter",

                height=10,

                aspect=0.3,

                data=train_data)
plt.figure(figsize=(9,9))

sns.relplot(x="SalePrice",

                y="GarageArea",

                col="GarageCars",

                hue="GarageQual",

                kind="scatter",

                height=10,

                aspect=0.3,

                data=train_data)
fig = px.scatter(train_data, x="SalePrice", y="1stFlrSF", color="OverallQual")

fig.show()
fig = px.scatter(train_data, x="SalePrice", y="TotalBsmtSF", color="OverallQual")

fig.show()
#print(obj_type_variables)

#print(int_type_variables)

int_selected=int_type_variables[1:-1]
train_data[obj_type_variables].shape
test_data[obj_type_variables].shape
print(train_data[obj_type_variables].shape[0])

print(test_data[obj_type_variables].shape)
obj_type_variables=obj_type_variables+obj_new_list

obj_type_variables=list(set(obj_type_variables))



train_data_X_obj = train_data[obj_type_variables].apply(lambda column: column.astype('category').cat.codes)

test_data_X_obj=test_data[obj_type_variables].apply(lambda column: column.astype('category').cat.codes) 





#train_data_X_obj = train_data[obj_type_variables].apply(lambda column: column.astype('category'))

#test_data_X_obj=test_data[obj_type_variables].apply(lambda column: column.astype('category'))



#train_data_X_obj = pd.get_dummies(train_data_X_obj)

#test_data_X_obj = pd.get_dummies(test_data_X_obj)
#combined_data = pd.concat((train_data[obj_type_variables], test_data[obj_type_variables]), sort=False).reset_index(drop=True)

#combined_data_dummies=pd.get_dummies(combined_data)
#train_data_X_obj=combined_data_dummies.iloc[:train_data[obj_type_variables].shape[0]].reset_index(drop=True)

#test_data_X_obj = combined_data_dummies.iloc[train_data[obj_type_variables].shape[0]:].reset_index(drop=True)
print(train_data_X_obj.shape)

print(test_data_X_obj.shape)
train_data_X_int=train_data[int_selected]

test_data_X_int=test_data[int_selected]
## Checking for missing values and imputing them

print('Checking for categorical variables for missing values')

print(train_data_X_obj.isna().sum())

print('Checking for integer variables for missing values')



print(train_data_X_int.isna().sum())

train_data_X_int=train_data_X_int.fillna(train_data_X_int.median())

test_data_X_int=test_data_X_int.fillna(train_data_X_int.median())
# scaler=StandardScaler()

#scaler=RobustScaler()

#scaler=MinMaxScaler()

#train_data_X_int=pd.DataFrame(scaler.fit_transform(train_data_X_int),columns=train_data_X_int.columns)

#test_data_X_int=pd.DataFrame(scaler.fit_transform(test_data_X_int),columns=test_data_X_int.columns)
train_Y=train_data['SalePrice']
train_X=train_data_X_int.merge(train_data_X_obj,left_index=True,right_index=True).reset_index(drop=True)

test_X=test_data_X_int.merge(test_data_X_obj,left_index=True,right_index=True).reset_index(drop=True)
print(train_X.shape)

print(train_Y.shape)

print(test_X.shape)
test_X.columns[test_X.isna().any()].tolist()
print(train_X.shape)

print(train_Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y,test_size = .3, random_state=0)


rf = RandomForestRegressor()

params = {"max_depth":[15,20,25], "n_estimators":[24,30,36]}

rf_reg = GridSearchCV(rf, params, cv = 10, n_jobs =10)

rf_reg.fit(X_train, Y_train)

print(rf_reg.best_estimator_)

best_estimator=rf_reg.best_estimator_

y_pred_test = best_estimator.predict(X_test)



print('Mean Square Error test = ' + str(mean_squared_error(Y_test, y_pred_test,squared=False)))

print('Root Mean Square Error test = ' + str((mean_squared_error(np.log(Y_test), np.log(y_pred_test),squared=False))))



print ('R2 square '+str(r2_score(Y_test, y_pred_test))) 
print(len(X_train.columns))

print(len(set(X_train.columns)))



#from xgboost import XGBRegressor

#xgb_regressor = XGBRegressor(learning_rate=0.02,n_estimators = 2400,max_depth=3, min_child_weight=0.01,gamma=0, subsample=0.7,

#                                     colsample_bytree=0.7,objective='reg:linear', nthread=-1, scale_pos_weight=1, seed=27,reg_alpha=0.00006)

#xgb_regressor.fit(X_train,Y_train)

#y_pred_test=xgb_regressor.predict(X_test)

#

#mean_squared_error(Y_test, y_pred_test)

#
from sklearn.ensemble import GradientBoostingRegressor

gb_regressor = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03, max_depth=3, max_features='sqrt', min_samples_leaf=15, min_samples_split=8, loss='huber', random_state =42)

gb_regressor.fit(X_train,Y_train)

y_pred_test=gb_regressor.predict(X_test)



print('Mean Square Error test = ' + str(mean_squared_error(Y_test, y_pred_test,squared=False)))

print('Root Mean Square Error test = ' + str((mean_squared_error(np.log(Y_test), np.log(y_pred_test),squared=False))))



print ('R2 square '+str(r2_score(Y_test, y_pred_test)))
#pred_test = best_estimator.predict(test_X)

pred_test = gb_regressor.predict(test_X)
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission=pd.DataFrame({"Id":sample_submission['Id'],"SalePrice":pred_test})

submission.to_csv('submission.csv',index=False)
sample_submission.head()