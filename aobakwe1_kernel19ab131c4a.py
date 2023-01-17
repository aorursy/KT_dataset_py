import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats, special
pd.set_option('display.max_columns', 300)



#import cleaned data into df

train_df = pd.read_csv('../input/train.csv')        #train df



test_df = pd.read_csv('../input/test.csv')     #test df



test_id = test_df.Id     #id's of the test data
#Global variables



dependent_var = 'SalePrice'     #Dependent variable we are trying to predict

correlation_threshold = 0.95    #Variable pairs that are correlated with each other for over 

                                #threshold one will be dropped

corr_with_salePrice = 0.05      #Variable that have correllation with dependent_var of range(-1, 1) 
df = pd.concat([train_df, test_df], sort = True).sort_values(by = 'Id')



df.set_index('Id', inplace = True, verify_integrity = True)
df.head()
for col in ['MSSubClass', 'OverallQual', 'OverallCond']:

    df[col] = df[col].astype('O')
#get a list of columns with missing values



def get_nan_columns(dataframe):

    '''

    Get two lists of:

        categorical column

        numerical column names of the given df

        params:

            dataframe: Pandas.DataFrame

        returns:

            a tuple of lists

            (categorical, Numerical)

'''

    cat_c = []

    num_c = []

    for col in dataframe.columns.values.tolist():

        if dataframe[col].isnull().any():

            if dataframe[col].dtype == 'O':

                cat_c.append(col)

            else:

                if col != dependent_var:

                    num_c.append(col)

                

    return cat_c, num_c
get_nan_columns(df)
#Only one kitchenQual of null. Impute with mode



df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])                                               
#Garage NaN values when GarageArea = 0.0. This means the house does not have a garage.

#categorical type, NG = No Garage

#numerical types, 0.0



for col in ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']:

    df[col] = np.where(df['GarageArea'] == 0.0, 'NG', df[col])



for col in ['GarageCars', 'GarageYrBlt']: 

    df[col] = np.where(df['GarageArea'] == 0.0, 0.0, df[col])
#GarageArea isnan and some value. 2 rows



for col in ['GarageCond', 'GarageFinish', 'GarageQual']:

    df[col] = df[col].fillna(df[col].mode()[0])



for col in ['GarageCars', 'GarageArea']: 

    df[col] = df[col].fillna(df[col].mean()) 

    

#for garageArea take the year of the house

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
#Basement values when TotalBsmtSF = 0.0

#NB = No Basement

for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:

    df[col] = np.where(df['TotalBsmtSF'] == 0.0, 'NB', df[col])



for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']:

    df[col] = np.where(df['TotalBsmtSF'] == 0.0, 0.0, df[col])
#Only one value of TotalBasementArea is zero and all other categories are zero

for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:

    df[col] = np.where(df['TotalBsmtSF'].isna(), 'NB', df[col])



for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']:

    df[col] = np.where(df['TotalBsmtSF'].isna(), 0.0, df[col])
#fill the rest with mode or mean depending on the category



for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:

    df[col] = df[col].fillna(df[col].mode()[0])



for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'TotalBsmtSF']:

    df[col] = df[col].fillna(df[col].mean())
df['PoolQC'] = np.where(df['PoolArea'] == 0.0, 'None', df['PoolQC'])

df['FireplaceQu'] = np.where(df['Fireplaces'] == 0, 'None', df['FireplaceQu'])
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['Alley'] = df['Alley'].fillna('None')

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

df['Functional'] = df['Functional'].fillna('Typ')

df['MiscFeature'] = df['MiscFeature'].fillna('None')
for col in ['Exterior1st', 'Exterior2nd', 'SaleType']:

    df[col] = df[col].fillna('Other')
for col in ['MSZoning', 'Utilities', 'Fence']:

    mode =  df[col].mode()[0]

    df[col] = df.groupby('Neighborhood')[col].apply(lambda x: x.fillna(x.value_counts().idxmax() if x.value_counts().max() >=1 else mode , inplace = False))

    df[col]= df[col].fillna(df[col].value_counts().idxmax())
# 23 out rows have both MasVnrType and MasVnrArea as None.

# Conclusion: MasVnrType = None, MasVnrArea = 0

# IF MasVnrArea = 0.0, MasVnrType = 'None'



df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)

df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['PoolQC'] = df['PoolQC'].fillna(df['PoolArea'].mean())
#df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
#1st get the log1p of all numerical columns



#df[dependent_var] = df[dependent_var].apply(np.log1p)

numerical_columns = df.select_dtypes(exclude=['O']).columns.values.tolist()

df[numerical_columns] = df[numerical_columns].apply(np.log1p)
def get_skewed_cols(dataframe):

    sk_cols = []

    for col in df.select_dtypes(exclude=['O']).columns.values.tolist():

        if col != dependent_var:

            if dataframe[col].dtype != 'O':

                if abs(dataframe[col].skew()) > 0.1:

                    sk_cols.append(col)

    return sk_cols
#transform skewed columns



df[get_skewed_cols(df)] = special.boxcox1p(df[get_skewed_cols(df)], 1e-4)
df.head()
def get_high_kurtosis_cols(dataframe):

    kurt_cols = []

    for col in df.select_dtypes(exclude=['O']).columns.values.tolist():

        if col != dependent_var:

            if dataframe[col].dtype != 'O':

                if abs(dataframe[col].kurt()) > 3:

                    kurt_cols.append(col)

    return kurt_cols
get_high_kurtosis_cols(train_df)
df.shape
#get categorical columns and encode them

categorical_columns = df.select_dtypes(include=['O']).columns.values.tolist()



df = pd.get_dummies(df, columns = categorical_columns, drop_first = True)
df.head()
train_df = df.iloc[:len(train_df)]

test_df = df.iloc[len(train_df):].drop(dependent_var, axis = 1)
#get pairs of highly correlated variables.

def get_correlated_pairs(dataframe):

    

    correlations = dataframe.corr()

    pairsList = []

    for col in correlations.columns.values.tolist():

        corr_series = correlations.loc[:, col]

    

        for row in corr_series.index.tolist():

            if (abs(corr_series.loc[row]) > correlation_threshold and row != col and abs(corr_series.loc[row]) != 1):

                if sorted([col, row]) not in pairsList: 

                    pairsList.append(sorted([col, row]))

    return pairsList
get_correlated_pairs(train_df)
#Most frequent columns turns out to be GarageYrBlt, GarageArea, TotalBsmtSF



train_df = train_df.drop(['GarageYrBlt', 'GarageArea', 'TotalBsmtSF', 

                          'Exterior1st_CemntBd', 'Exterior1st_MetalSd', 

                          'Exterior1st_VinylSd'], axis = 1)

test_df = test_df.drop(['GarageYrBlt', 'GarageArea', 'TotalBsmtSF', 

                          'Exterior1st_CemntBd', 'Exterior1st_MetalSd', 

                          'Exterior1st_VinylSd'], axis = 1)

#get correlation for each col with SalePrice in train_df



def get_corr_with_SalePrice(dataframe):

    

    cor = pd.DataFrame(dataframe.corr()[dependent_var])

    

    return cor[abs(cor.SalePrice) > corr_with_salePrice].index.values.tolist()
train_df.shape
test_df.shape
droplist = get_corr_with_SalePrice(train_df)

train_df = train_df[droplist]

droplist.remove(dependent_var)

test_df = test_df[droplist]
test_df.shape
train_df.shape
#get pairs of highly correlated variables.

def get_correlated_pairs(dataframe):

    

    correlations = dataframe.corr()

    pairsList = []

    for col in correlations.columns.values.tolist():

        corr_series = correlations.loc[:, col]

    

        for row in corr_series.index.tolist():

            if (abs(corr_series.loc[row]) > correlation_threshold and row != col):

                if sorted([col, row]) not in pairsList: 

                    pairsList.append(sorted([col, row]))

    return pairsList
#get columns that have less correlation with SalePrice from the pairs above



def get_less_correlated(dataframe):

    

    less_correlated = []



    for pair in get_correlated_pairs(dataframe):

    

        if dataframe[pair[0]].corr(dataframe[dependent_var]) > dataframe[pair[1]].corr(dataframe[dependent_var]):

            if pair[1] not in less_correlated:

                less_correlated.append(pair[1])

        else:

            if pair[0] not in less_correlated:

                less_correlated.append(pair[0])

    return less_correlated
#Outliers need to figure out how they were identified first, else we do not include them

#train_df = train_df.drop(train_df.index[[30, 88, 462, 631, 1322]])
train_df.shape
test_df.shape
y = train_df[dependent_var]

X = train_df.drop([dependent_var], axis = 1)
#Scale X and test_df



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

test_df_scaled = scaler.fit_transform(test_df)

test_df_scaled = pd.DataFrame(test_df_scaled, columns = test_df.columns)
#split train and test samples from X, y



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)
#train with LinearRegression

from sklearn.linear_model import LinearRegression



lr_model = LinearRegression(normalize=True, n_jobs=-1)

lr_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)
#train with Lasso

from sklearn.linear_model import Lasso



l_model = Lasso(alpha = 0.000395)

l_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)
#train with Ridge

from sklearn.linear_model import Ridge



r_model = Ridge(alpha = 0.0001, max_iter=1000000, normalize=True)

r_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)
#train with RandomForest

from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=100)

rf_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)

#train with GradiientBoosting

from sklearn.ensemble import GradientBoostingRegressor



gb_model = GradientBoostingRegressor()

gb_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)
from xgboost import XGBRegressor



xgb_model = XGBRegressor()

xgb_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)
from lightgbm import LGBMRegressor



lgbm_model = LGBMRegressor()

lgbm_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)
from mlxtend.regressor import StackingCVRegressor



stack_regr = StackingCVRegressor(regressors = (l_model,

                                               r_model,

                                               gb_model,

                                               rf_model

                                               ), meta_regressor=gb_model)



stack_regr.fit(np.array(X_train), np.array(y_train))
from sklearn.metrics import mean_squared_log_error



#print('Linear Model: ', np.sqrt(mean_squared_log_error(y_test, 

 #                              lr_model.predict(X_test))))



print('Lasso: ', np.sqrt(mean_squared_log_error(y_test,

                         l_model.predict(X_test))))



print('Ridge: ', np.sqrt(mean_squared_log_error(y_test,

                         r_model.predict(X_test))))



print('RandomForest: ', np.sqrt(mean_squared_log_error(y_test,

                                                       rf_model.predict(X_test))))



print('GradientBoosting: ', np.sqrt(mean_squared_log_error(y_test,

                                                       gb_model.predict(X_test))))



print('XGBoost: ', np.sqrt(mean_squared_log_error(y_test,

                                                       xgb_model.predict(X_test))))



print('LightGBM: ', np.sqrt(mean_squared_log_error(y_test,

                                                       lgbm_model.predict(X_test))))



print('Stack: ', np.sqrt(mean_squared_log_error(y_test,

                                                       stack_regr.predict(np.array(X_test)))))
'''y_pred = np.expm1(0.2*gb_model.predict(test_df_scaled)+

                   0.15*rf_model.predict(test_df_scaled)+

                   0.2*l_model.predict(test_df_scaled)+

                   0.2*r_model.predict(test_df_scaled)+

                   0.25*stack_regr.predict(np.array(test_df_scaled)))'''

y_pred = np.expm1(l_model.predict(test_df_scaled))
solution = pd.DataFrame({'Id': test_id, 'SalePrice':y_pred})
solution.to_csv('finalTest.csv', index = False)