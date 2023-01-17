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
X = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

X.head()
X.BsmtQual.value_counts()
X.drop(['Id'], axis = 1, inplace = True)
categorical_cols = []
lel = X.isnull().sum().sort_values(ascending = False)[15:20]

lel

for i in lel.keys():

    print (i + ' number of null is ' + str(lel[i]))

    print (X.groupby(i)['SalePrice'].agg(['mean','count']).sort_values('mean').reset_index())

    
X.GarageFinish.isnull().sum()
col_name = 'GarageFinish'

lolol = X.groupby('GarageFinish')['SalePrice'].mean().sort_values()

new_GarageFinish = []

result = X[col_name].mode()[0]

for i in range(X.shape[0]):

    new_val = X.GarageFinish[i]

    if type(X.GarageFinish[i]) == str:

        new_GarageFinish.append(X.GarageFinish[i])

    else:

        abs_diff = 10000000000

        for a in lolol.keys():

            abs_value = abs(lolol[a] - X.SalePrice[i])

            if  abs_value < abs_diff:

                abs_diff = abs_value

                result = a

        new_GarageFinish.append(result)



X['GarageFinish_new'] = new_GarageFinish

categorical_cols += ['GarageFinish_new']
#BsmtQual

col_name = 'BsmtQual'

lolol = X.groupby(col_name)['SalePrice'].mean().sort_values()

new_vals = []

result = X[col_name].mode()[0]

for i in range(X.shape[0]):

    new_val = X[col_name][i]

    if type(X[col_name][i]) == str:

        new_vals.append(X[col_name][i])

    else:

        abs_diff = 10000000000

        for a in lolol.keys():

            abs_value = abs(lolol[a] - X[col_name][i])

            if  abs_value < abs_diff:

                abs_diff = abs_value

                result = a

        new_vals.append(result)



X['BsmtQual_new'] = new_vals

categorical_cols.append("BsmtQual_new")
#BsmtExposure

col_name = 'BsmtExposure'

lolol = X.groupby(col_name)['SalePrice'].mean().sort_values()

new_vals = []

result = X[col_name].mode()[0]

for i in range(X.shape[0]):

    new_val = X[col_name][i]

    if type(X[col_name][i]) == str:

        new_vals.append(X[col_name][i])

    else:

        abs_diff = 10000000000

        for a in lolol.keys():

            abs_value = abs(lolol[a] - X[col_name][i])

            if  abs_value < abs_diff:

                abs_diff = abs_value

                result = a

        new_vals.append(result)



new_col_name = col_name + '_new'

X[new_col_name] = new_vals

categorical_cols.append(new_col_name)
X.BsmtExposure_new.value_counts()
numerical_cols = []
#GarageYearBuilt

col_name = 'GarageYrBlt'

new_col_name = col_name + '_new'

new_values = []

#new values list

for i in range(X.shape[0]):

    if np.isnan(X[col_name][i]):

        new_values.append(X.YearBuilt[i])

    else:

        new_values.append(X.GarageYrBlt[i])

X[new_col_name] = new_values

numerical_cols.append(new_col_name)

#checking for new values of grgyearbult

lol_hihi = 0

for i in range(1460):

    if X[col_name][i] != new_values[i]:

        lol_hihi += 1

lol_hihi
#MasVnrArea fillna with 0 cus 861 values of 0 (8 nan values only)

col_name = 'MasVnrArea'

new_col_name = col_name + '_new'

X[new_col_name] = X[col_name].fillna(0)

numerical_cols.append(new_col_name)
#Electrical



col_name = 'Electrical'

lolol = X.groupby(col_name)['LotArea'].mean().sort_values()

new_vals = []

result = X[col_name].mode()[0]

for i in range(X.shape[0]):

    new_val = X[col_name][i]

    if type(X[col_name][i]) == str:

        new_vals.append(X[col_name][i])

    else:

        abs_diff = 10000000000

        for a in lolol.keys():

            abs_value = abs(lolol[a] - X[col_name][i])

            if  abs_value < abs_diff:

                abs_diff = abs_value

                result = a

        new_vals.append(result)



new_col_name = col_name + '_new'

X[new_col_name] = new_vals

categorical_cols = categorical_cols + [new_col_name]
categorical_cols


lol = X.corr().SalePrice.reset_index()

lol.columns = ['col_name','corr_Price']

lol['abs_corr'] = lol.corr_Price.apply(lambda x: abs(x))

lol.sort_values('abs_corr', ascending = False)

#using features with corr > 0.2

used_features = lol[lol.abs_corr > 0.2].col_name

used_features



y = X.SalePrice

used_features.pop(36)
#add categorical and numercial cols to used features

#after processing manually, numerical cols automatically added to used_features

used_cols = used_features



used_features = [i for i in used_features]



used_features = used_features + categorical_cols



used_features.remove('MasVnrArea')

used_features.remove('GarageYrBlt')
#start to create and clean XX, prepare to machine learn
XX = X.loc[:, used_features]

#drop na from XX, cus duplicate cols due to corr list



#handing year built values

XX['YearBuilt'] = XX.YearBuilt - 1675

XX['YearRemodAdd'] = XX['YearRemodAdd'] - 1675

XX['GarageYrBlt_new'] = XX['GarageYrBlt_new'] - 1675
#No Categorical Data, ready for machine learning

XXX = pd.get_dummies(XX, columns = categorical_cols, drop_first = True)
XXX.drop(['Electrical_new_Mix'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(XXX, y, train_size = 0.8, test_size = 0.2)



from xgboost import XGBRegressor



model1 = XGBRegressor(n_estimator = 500, learning_rate = 0.5)

model1.fit(X_train, y_train, early_stopping_rounds = 40,

          eval_set = [(X_valid, y_valid)],

          verbose = False)
preds = model1.predict(X_valid)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(preds, y_valid)
X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



for i in ['GarageFinish','BsmtQual','BsmtExposure','Electrical',"GarageYrBlt",'MasVnrArea']:



    X_test.rename(columns = {i:i+'_new'}, inplace=True)

    

XX = X_test.loc[:, used_features]





XX['YearBuilt'] = XX.YearBuilt - 1675

XX['YearRemodAdd'] = XX['YearRemodAdd'] - 1675

XX['GarageYrBlt_new'] = XX['GarageYrBlt_new'] - 1675    

#filling in na

lol = XX.isnull().sum().sort_values(ascending = False)

lol_cols = [i for i in lol.keys() if lol[i]>0]

for i in lol_cols:

    if XX[i].dtype == float:

        XX[i].fillna(XX[i].mean(), inplace = True)

    else:

        XX[i].fillna(XX[i].mode()[0], inplace = True)
categorical_cols
XXX = pd.get_dummies(XX, columns = categorical_cols, drop_first = True)
XXX.shape
test_pred = model1.predict(XXX)
submission = pd.DataFrame({'Id':X_test.Id, 'SalePrice':test_pred})

submission.to_csv('submission.csv', index=False)
def clean_categorical_columns(col_name, data_frame):

    col_name = col_name

    Xx = data_frame

    lolol = Xx.groupby(col_name)['LotArea'].mean().sort_values()

    new_vals = []

    result = 'llol'

    for i in range(Xx.shape[0]):

        new_val = Xx[col_name][i]

        if type(Xx[col_name][i]) == str:

            new_vals.append(Xx[col_name][i])

        else:

            abs_diff = 10000000000

            for a in lolol.keys():

                abs_value = abs(lolol[a] - Xx[col_name][i])

                if  abs_value < abs_diff:

                    abs_diff = abs_value

                    result = a

            new_vals.append(result)



    new_col_name = col_name + '_new'

    Xx[new_col_name] = new_vals
