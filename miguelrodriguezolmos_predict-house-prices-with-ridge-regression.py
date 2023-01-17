#basic modules

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import os

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import Ridge
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')



X = df_train.drop(['SalePrice', 'Id'], axis = 1)

train_labels = df_train['SalePrice'].values

X_test = df_test.drop(['Id'], axis = 1)
#df_data_types = X.dtypes



for colname in X.columns:

    print(colname + ': ' + str(X[colname].dtype))
def prepro_change_column_type(x):

    x['MSSubClass'] = x['MSSubClass'].astype('str')



    

prepro_change_column_type(X)



prepro_test_list =[]

prepro_test_list.append(prepro_change_column_type)

    

# df_train['MSSubClass'] = df_train['MSSubClass'].astype('str')

# df_test['MSSubClass']  = df_test['MSSubClass'].astype('str')
#proportion of NaN in each column

count_var_nulls = X.isnull().sum(axis = 0)/X.shape[0]

variable_nulls = count_var_nulls[count_var_nulls >0]



print('variables with NaN:')

print(variable_nulls)

print('-----')

#remove columns with more than 50% of NaN

remove_variables_index = list(variable_nulls[variable_nulls > 0.5].index)

variable_nulls.drop(remove_variables_index, inplace = True)  #prepro remove_variables_index



def prepro_nan_columns(x):

    x.drop(remove_variables_index, axis =1,  inplace = True)



print('remaining variables with NaN after dropping those with more than 50% missing:')

print(variable_nulls)  



prepro_nan_columns(X)

prepro_test_list.append(prepro_nan_columns)
num_columns = X.select_dtypes(include=np.number).columns.tolist() 

cat_columns = X.select_dtypes(include=['object', 'category']).columns.tolist() 
count_obs_nulls = X.isnull().sum(axis = 1)/X.shape[1]

obs_nulls = count_obs_nulls[count_obs_nulls >0]

remove_obs_index = list(obs_nulls[obs_nulls > 0.5].index)

X.drop(remove_obs_index, axis = 1, inplace = True)

print(len(remove_obs_index),' observations removed because of having more than 50% of null values' )
#prepro

def prepro_nan_objective_imputing(x):

    #categorical

    x['MasVnrType'].fillna('None', inplace = True)

   

    aux_list = ['BsmtQual', 

                  'BsmtCond', 

                  'BsmtExposure',

                  'BsmtFinType1',

                  'BsmtFinType2',

                  'GarageType', 

                  'GarageFinish',

                  'GarageQual', 

                  'GarageCond',

                  'FireplaceQu']

    

    for i in aux_list:

        x[i].fillna('NA', inplace = True)  

        

    #numerical

    x['MasVnrArea'].fillna(0, inplace = True)    

    

  

       

    x.loc[x['BsmtCond'] == 'NA', ['BsmtUnfSF', 

                                   'TotalBsmtSF', 

                                   'BsmtFullBath', 

                                   'BsmtHalfBath', 

                                   'BsmtFinSF1', 

                                   'BsmtFinSF2' ]] = 0

    

prepro_nan_objective_imputing(X)

prepro_test_list.append(prepro_nan_objective_imputing)
X = X.loc[

    (  X['LotArea']<100000) 

    | (X['LotFrontage']<250)

    | (X['1stFlrSF']<4000)

    | (X['BsmtFinSF1']<5000) 

    | (X['BsmtFinSF2']<1400) 

    | (X['EnclosedPorch']<500)

    | (X['GrLivArea']<5000)

    | (X['TotalBsmtSF']<6000), ]
numeric_transformer = Pipeline([

                                ("median", SimpleImputer(strategy='median')),

                                ("standard", StandardScaler())

                               ]

                              )

categorical_transformer = Pipeline([

                                ("mostfreq", SimpleImputer(strategy = 'most_frequent')),

                                ("onehot", OneHotEncoder(handle_unknown='ignore'))

                                   ]

                                  )
prepro_transformer = ColumnTransformer([('num', numeric_transformer, num_columns),

                                       ('cat', categorical_transformer, cat_columns)])
y = np.log10(train_labels)
pipeline = Pipeline([('prepro', prepro_transformer), ('estimator', Ridge())])
param_grid = dict(estimator__alpha =  np.linspace(10,100, 100))

grid_search = GridSearchCV(pipeline, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 0)
grid_search.fit(X, y)
best_model = grid_search.best_estimator_
for func in prepro_test_list:

    func(X_test)
y_pred   = 10**best_model.predict(X_test)

df_submit = pd.DataFrame({'Id':df_test['Id'], 'SalePrice': y_pred})

df_submit.to_csv('df_submit.csv', index = False)
grid_search.best_params_