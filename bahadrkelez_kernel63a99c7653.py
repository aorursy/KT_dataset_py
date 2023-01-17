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
import numpy as np
import pandas as pd
import datetime as dt
import math
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Below, display settings are performed:
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 640)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# House data is imported below:
house_data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', sep=',', header=0)
house_data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', sep=',', header=0)
house_data_test['SalePrice'] = 0
house_data = pd.concat([house_data_train, house_data_test], ignore_index=True)


"""
# Checks:
dummy = house_data.loc[(house_data['LotConfig'] == 'FR2') & (house_data['LotFrontage'].isnull() == True), ['MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt']].sort_values(by='Neighborhood')
dummy['LotDepth'] = dummy['LotArea'] / dummy['LotFrontage']

for ii in house_data.index.to_list():
    if pd.isnull(house_data.loc[ii, 'LotFrontage']) == False:
        house_data.loc[ii, 'LotDepth'] = house_data.loc[ii, 'LotArea'] / house_data.loc[ii, 'LotFrontage']
"""
# Below, Condition1 and Condition2 features are combined into a single feature.
# Condition1 and Condition2 features will probably be dropped while fitting the model into data:
for ii in house_data.index.to_list():
    if (house_data.loc[ii, 'Condition1'] == 'Norm') & (house_data.loc[ii, 'Condition2'] == 'Norm'):
        house_data.loc[ii, 'Combined_condition'] = 'Norm'
    elif (house_data.loc[ii, 'Condition1'] != 'Norm') & (house_data.loc[ii, 'Condition2'] == 'Norm'):
        house_data.loc[ii, 'Combined_condition'] = house_data.loc[ii, 'Condition1']
    elif house_data.loc[ii, 'Condition1'] < house_data.loc[ii, 'Condition2']:
        house_data.loc[ii, 'Combined_condition'] = house_data.loc[ii, 'Condition1'] + '_' + house_data.loc[ii, 'Condition2']
    else:
        house_data.loc[ii, 'Combined_condition'] = house_data.loc[ii, 'Condition2'] + '_' + house_data.loc[ii, 'Condition1']



# Below, ages of the houses based on costruction and remodelling dates are calcualted by using 2010 as current date.
# So, 2 new features are created below:
house_data['YearRemodAdd'].max() # 2010
house_data['YearBuilt'].max() # 2010
# Use 2010 as the base date while calculating ages of the houses
for ii in house_data.index.to_list():
    house_data.loc[ii, 'Original_Age'] = 2010 - house_data.loc[ii, 'YearBuilt']
    house_data.loc[ii, 'Remodelling_Age'] = 2010 - house_data.loc[ii, 'YearRemodAdd']
    if house_data.loc[ii, 'YearBuilt'] == house_data.loc[ii, 'YearRemodAdd']:
        house_data.loc[ii, 'Remodelling_State'] = 0
    else:
        house_data.loc[ii, 'Remodelling_State'] = 1
house_data['Original_Age'] = house_data['Original_Age'].astype(int)
house_data['Remodelling_Age'] = house_data['Remodelling_Age'].astype(int)
house_data['Remodelling_State'] = house_data['Remodelling_State'].astype(int)


# If there exists only one type of exterior covering on the house, 'Exterior1st' and 'Exterior2nd' are the same.
# If there exists more than one types of exterior covering on the house, 'Exterior1st' and 'Exterior2nd' are different from each other.
# Below, a new feature is created in order to combine these two features into a single feature:
house_data['Exterior1st'] = house_data['Exterior1st'].apply(lambda x: 'VinylSd' if pd.isnull(x) else x)
house_data['Exterior2nd'] = house_data['Exterior2nd'].apply(lambda x: 'VinylSd' if pd.isnull(x) else x)

for ii in house_data.index.to_list():
    if house_data.loc[ii, 'Exterior1st'] == house_data.loc[ii, 'Exterior2nd']:
        house_data.loc[ii, 'Combined_exterior_covering'] = house_data.loc[ii, 'Exterior2nd']
    elif house_data.loc[ii, 'Exterior1st'] < house_data.loc[ii, 'Exterior2nd']:
        house_data.loc[ii, 'Combined_exterior_covering'] = house_data.loc[ii, 'Exterior1st'] + '_' + house_data.loc[ii, 'Exterior2nd']
    else:
        house_data.loc[ii, 'Combined_exterior_covering'] = house_data.loc[ii, 'Exterior2nd'] + '_' + house_data.loc[ii, 'Exterior1st']



# Below, 3 new features called Basement_Exists, Bsmt_finishing_status and Bsmt_finishing_percentage are created:
for ii in house_data.index.to_list():
    if house_data.loc[ii, 'TotalBsmtSF'] == 0:
        house_data.loc[ii, 'Basement_Exists'] = 0
    else:
        house_data.loc[ii, 'Basement_Exists'] = 1

    if pd.isnull(house_data.loc[ii, 'BsmtQual']) == True:
        house_data.loc[ii, 'BsmtQual'] = 'None'
    if  pd.isnull(house_data.loc[ii, 'BsmtCond']) == True:
        house_data.loc[ii, 'BsmtCond'] = 'None'
    if  pd.isnull(house_data.loc[ii, 'BsmtExposure']) == True:
        house_data.loc[ii, 'BsmtExposure'] = 'None'
    if  pd.isnull(house_data.loc[ii, 'BsmtFinType1']) == True:
        house_data.loc[ii, 'BsmtFinType1'] = 'None'
    if  pd.isnull(house_data.loc[ii, 'BsmtFinType2']) == True:
        house_data.loc[ii, 'BsmtFinType2'] = 'None'

    if house_data.loc[ii, 'TotalBsmtSF'] == 0:
        house_data.loc[ii, 'Bsmt_finishing_status'] = 0
        house_data.loc[ii, 'Bsmt_finishing_percentage'] = 0
    else:
        if house_data.loc[ii, 'BsmtUnfSF'] == 0:
            house_data.loc[ii, 'Bsmt_finishing_status'] = 1
        else:
            house_data.loc[ii, 'Bsmt_finishing_status'] = 0
        house_data.loc[ii, 'Bsmt_finishing_percentage'] = ((house_data.loc[ii, 'TotalBsmtSF'] - house_data.loc[ii, 'BsmtUnfSF']) / house_data.loc[ii, 'TotalBsmtSF']) * 100

house_data['Basement_Exists'] = house_data['Basement_Exists'].astype(int)
house_data['Bsmt_finishing_status'] = house_data['Bsmt_finishing_status'].astype(int)


# Below, whether the house has a central air conditioning system is specified numerically:
house_data['CentralAir'] = house_data['CentralAir'].apply(lambda x: 1 if x == 'Y' else 0)


# Below, a new feature representing existence of the second floor in a house is created:
house_data['2ndFloorExistence'] = house_data['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)


# Below, a new feature representing existence of a fireplace in a house is created:
# Also, null values within FireplaceQu column is replaced by "None".
house_data['FireplaceExistence'] = house_data['FireplaceQu'].apply(lambda x: 0 if pd.isnull(x) == True else 1)
house_data['FireplaceQu'] = house_data['FireplaceQu'].apply(lambda x: 'None' if pd.isnull(x) == True else x)


# Below, a new feature representing existence of a garage in a house is created:
house_data['GarageExistence'] = house_data['GarageType'].apply(lambda x: 0 if pd.isnull(x) == True else 1)
house_data['GarageType'] = house_data['GarageType'].apply(lambda x: 'None' if pd.isnull(x) == True else x)
house_data['GarageFinish'] = house_data['GarageFinish'].apply(lambda x: 'None' if pd.isnull(x) == True else x)
house_data['GarageQual'] = house_data['GarageQual'].apply(lambda x: 'None' if pd.isnull(x) == True else x)
house_data['GarageCond'] = house_data['GarageCond'].apply(lambda x: 'None' if pd.isnull(x) == True else x)


# There are 81 records with no Garage. For those records, GarageYrBlt is assigned a value of 0.
# The logic is that: Garage existence adds value to the price of the house, however the records with no garage should not lose price because of the year in which the garage was built.
# Therefore, 0 is assigned to GarageYrBlt column's value:
house_data['GarageYrBlt'] = house_data['GarageYrBlt'].apply(lambda x: 0 if pd.isnull(x) else x)
house_data['GarageYrBlt'] = house_data['GarageYrBlt'].astype(int)


# Below, a new feature representing existence of a pool in a house is created:
house_data['PoolExistence'] = house_data['PoolQC'].apply(lambda x: 1 if pd.isnull(x) == False else 0)
house_data['PoolQC'] = house_data['PoolQC'].apply(lambda x: 'None' if pd.isnull(x) else x)


# Below, a new feature representing existence of a fence in a house is created:
house_data['FenceExistence'] = house_data['Fence'].apply(lambda x: 0 if pd.isnull(x) else 1)
house_data['Fence'] = house_data['Fence'].apply(lambda x: 'None' if pd.isnull(x) else x)



# Below, "None" is assigned as a value to the "Miscellaneous" values of the records wit null "Miscellaneous" values:
house_data['MiscFeature'] = house_data['MiscFeature'].apply(lambda x: 'None' if pd.isnull(x) else x)


# LotFrontage column has 199 - many null valeus. Dropping that much of rows decreases the size of training data remarkably.
# Instead, LotFrontage column should be dropped.
house_data.drop(columns='LotFrontage',inplace=True)

# Since 1369-many null values exist within Alley column, Alley should be dropped:
house_data.drop(columns='Alley', inplace=True)


# Condition1 and Condition2 can be dropped since that two features are combined by creating a new feature called Combined_condition:
house_data.drop(columns=['Condition1', 'Condition2'], inplace=True)


# Since age related new features are created using YearBuilt and YearRemodAdd features, YearBuilt and YearRemodAdd can be dropped.
# It is not logical to determine a coefficient for those features in a regression model and multiply the year value with that coefficient:
house_data.drop(columns=['YearBuilt', 'YearRemodAdd'], inplace=True)


# Since Exterior1st and Exterior2nd features are combined by creating a new feature called Combined_exterior using these two,
# these features can be dropped:
house_data.drop(columns=['Exterior1st', 'Exterior2nd'], inplace=True)


# 8-many records do not have MasVnrType and therefore MasVnrArea. Assigning them a MasVnrType is nonsense since it is not possible to make a reliable prediction for that feature's value.
# Therefore, these 8 rows should be dropped:
# Since there is data which has null MasVnrType and which must be assigned a SalePrice, we cannot drop rows with null MasVnrType.
# Instead, most frequent value which is "None" will be replaced with null "MasVnrType" values:
house_data['MasVnrType'] = house_data['MasVnrType'].apply(lambda x: 'None' if pd.isnull(x) else x)


# Only 1 record do not have a value for Electrical feature. This records can be dropped:
house_data.drop(columns='Electrical', inplace=True)


# Below, MSSubClass feature is transformed into a categorical feature:
MSSubClass_dict = {20: 'Type1', 30: 'Type2', 40: 'Type3', 45: 'Type4', 50: 'Type5', 60: 'Type6', 70: 'Type7', 75: 'Type8', 80: 'Type9', 85: 'Type10', 90: 'Type11', 120: 'Type12', 150: 'Type13', 160: 'Type14', 180: 'Type15', 190: 'Type16'}
house_data['MSSubClass'] = house_data['MSSubClass'].apply(lambda x: MSSubClass_dict[x])


# Below, MoSold feature representing the month in which the house was sold is transformed into a categorical feature.
# It is not logical to multply the month no. with a coefficient, instead there should be 12 many features for a total 12 months  of a year.
# The month in which the house was sold should take the value 1, and the others should take the value 0:
MoSold_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
house_data['MoSold'] = house_data['MoSold'].apply(lambda x: MoSold_dict[x])


# Below, YrSold feature representing the year in which the house was sold is transformed into a categorical feature:
house_data['YrSold'] = house_data['YrSold'].apply(lambda x: str(x))


# Garage_Age featıre is created using GarageYrBlt feature and taking 2010 as a base year while calcualting the age of a garage below:
house_data['Garage_Age'] = house_data['GarageYrBlt'].apply(lambda x: 0 if x == 0 else (2010 - x))


# There are 2 records with null "Utilities" value. Most frequent one (2916 many out of 2917) "AllPub" is replaced with null values:
house_data['Utilities'] = house_data['Utilities'].apply(lambda x: 'AllPub' if pd.isnull(x) else x)


# 0 is assigned to "MasVnrArea" of the records whose "MasVnrType" is "None":
house_data.loc[house_data['MasVnrType'] == 'None', 'MasVnrArea'] = 0


# Records with no basement have null values within "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath" and "BsmtHalfBath" columns:
# Those null values are replaced with 0:
house_data.loc[house_data['BsmtQual'] == 'None', 'BsmtFinSF1'] = 0
house_data.loc[house_data['BsmtQual'] == 'None', 'BsmtFinSF2'] = 0
house_data.loc[house_data['BsmtQual'] == 'None', 'BsmtUnfSF'] = 0
house_data.loc[house_data['BsmtQual'] == 'None', 'TotalBsmtSF'] = 0
house_data.loc[house_data['BsmtQual'] == 'None', 'BsmtFullBath'] = 0
house_data.loc[house_data['BsmtQual'] == 'None', 'BsmtHalfBath'] = 0
house_data.loc[house_data['BsmtQual'] == 'None', 'Bsmt_finishing_percentage'] = 0



# House with Id = 2576 has a garage("Detchd") but does not have "GarageArea" and "GarageCars" values.
house_data.loc[house_data['GarageCars'].isnull(), 'GarageCars'] = 2 # Most frequent GarageCars value for "Detchd" type garages.
house_data.loc[house_data['GarageCars'].isnull(), 'GarageArea'] = 419 # Mean GarageArea value for "Detchd" type garages.


# House with Id = 2490 does not have a SaleType value. "None" will be assigned as a "SaleType" value for that house:
house_data.loc[house_data['SaleType'].isnull(), 'SaleType'] = 'None'


# House with Id = 1556 does not have a "KitchenQual" value althought it has a kitchen.
# Its "KitchenAbvGr" value is 1.
# Below, most frequent "KitchenQual" value "TA" for the group which have a value of 1 for "KitchenAbvGr" is replaced with this null value:
house_data.loc[house_data['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'



# For the houses with null "Functional" value, most frequent "Functional" value "Typ" is replaced with those null values:
house_data.loc[house_data['Functional'].isnull(), 'Functional'] = 'Typ'


# There are 4 records with null 'MSZoning' value. I assume that houses within the same 'Neighborhood' have similar 'MSZoning' types.
# Houses with Id in [1916, 2217, 2251] are located in 'IDOTRR' neighborhood. Most often 'MSZoning' type within 'IDOTRR' is 'RM'.
# So, 'RM' will be assigned as 'MSZoning' value for those records.
# Also, house with Id=2905 is located within 'Mitchel' neighborhood. Most frequent 'MSZoning' type within that neighborhood is 'RL'.
# Therefore, 'RL' will be assigned as 'MSZoning' value for that record:
house_data.loc[house_data['Id'].isin([1916, 2217, 2251]), 'MSZoning'] = 'RM'
house_data.loc[house_data['Id'] == 2905, 'MSZoning'] = 'RL'
"""
house_data.loc[house_data['Neighborhood'] == 'IDOTRR', 'MSZoning'].value_counts()
RM         68
C (all)    22

house_data.loc[house_data['Neighborhood'] == 'Mitchel', 'MSZoning'].value_counts()
RL    104
RM      9
"""

# House with Id=2577 has no value for 'GarageArea'. It is known that the 'GarageType' = 'Detchd' and 'GarageCars' = 2 for that house.
# So, mean 'GarageArea' value for the houses with 'GarageType' = 'Detchd' and 'GarageCars' = 2 is assigned as a value to the 'GarageArea'
# of this house:
house_data.loc[(house_data['GarageType'] == 'Detchd') & (house_data['GarageCars'] == 2), 'GarageArea'].mean() # 530
house_data.loc[house_data['Id'] == 2577, 'GarageArea'] = 530



# There is no null value within house_data dataframe as seen below:
np.any(pd.isnull(house_data)) # False


house_data_1 = house_data.copy(deep=True)
house_data_1 = house_data_1.loc[house_data_1['Id'] <= 1460]
house_data_1.drop(columns='Id', inplace=True)
house_data_1.drop(columns='GarageYrBlt', inplace=True)
house_data_1 = pd.get_dummies(data=house_data_1)

# Since LotDepth feature is created to predict the LotFrontage value and LotFrontage feature is dropped, LotDepth feature is not useful anymore.
# LotDepth can be dropped:
# house_data.drop(columns='LotDepth', inplace=True)

# house_data_1 is divided in to train and test sets below:
training_columns = house_data_1.columns.values.tolist()
del training_columns[30]
# training_columns.pop(30) --> removes the element with index=30 from that list inplace
x_train, x_test, y_train, y_test = train_test_split(house_data_1[training_columns], house_data_1['SalePrice'], test_size=0.2, random_state=42,)



# Below, LinearRegression is used:
linreg_1 = linear_model.LinearRegression(n_jobs=-1)
scores = cross_validate(linreg_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [34966.73022486446, 7498.100313127563]
# [Mean R^2, Std dev. of R^2] --> [0.8014416326854731, 0.07170108119327159]


# Below, Ridge regression is used:
ridge =  Ridge(random_state=42)
parameter_grid = {'max_iter': [1000, 2000, 3000, 4000, 5000], 'tol': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]}
grsearch = GridSearchCV(estimator=ridge, param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'max_iter': 1000, 'tol': 0.001}


ridge_1 = Ridge(random_state=42, max_iter=1000, tol=0.001)
scores = cross_validate(ridge_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [31614.867448381767, 7508.960407305801]
# [Mean R^2, Std dev. of R^2] --> [0.8354079570451608, 0.07086869589166794]
# Ridge regression performance is better than LinearRegression's performance in terms of root_mean_squared_error and r2 scores as seen above.



# Below, Lasso regression is used:
lasso =  Lasso(random_state=42)
parameter_grid = {'max_iter': [1000, 2000, 3000, 4000, 5000], 'tol': [0.0001, 0.00002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]}
# When I tried to use tolerance options of Ridge regression in Lasso, the performance was worse than that of the model in which tolerance options above were used:
# Therefore, I continued using the better tolerance options which are different than those used in ridge resgression model.
grsearch = GridSearchCV(estimator=lasso, param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'max_iter': 4000, 'tol': 2e-05}


lasso_1 = Lasso(tol=2e-05, random_state=42, max_iter=4000)
scores = cross_validate(lasso_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [34109.337133481924, 7886.301977162523]
# [Mean R^2, Std dev. of R^2] --> [0.8105955903122632, 0.07427986411167581]
# Ridge is better than Lasso based on r2 and RMSE metrics calcualted above.


# Below, DecisionTreeRegressor is used:
tree = DecisionTreeRegressor(random_state=42)
parameter_grid = {'max_depth': np.arange(11)[1:],
                  'min_samples_leaf': [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4],
                  'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.004, 0.005]}
grsearch = GridSearchCV(estimator=tree,param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'max_depth': 9, 'min_impurity_decrease': 0, 'min_samples_leaf': 0.02}



tree_1 = DecisionTreeRegressor(random_state=42, max_depth=9, min_samples_leaf=0.02)
scores = cross_validate(tree_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [37337.29305735366, 4083.057832689985]
# [Mean R^2, Std dev. of R^2] --> [0.777185059699525, 0.026265942777495534]
# Ridge is better than DecisionTreeRegressor based on r2 and RMSE metrics calculated above.



# Below, VotingRegressor is used:
linreg = linear_model.LinearRegression(n_jobs=-1)
ridge = Ridge(random_state=42, max_iter=1000, tol=0.001)
lasso = Lasso(tol=2e-05, random_state=42, max_iter=4000)
tree = DecisionTreeRegressor(random_state=42, max_depth=9, min_samples_leaf=0.02)
vote = VotingRegressor(estimators=[('lr', linreg), ('ridge', ridge), ('lasso', lasso), ('dt', tree)], n_jobs=-1)


scores = cross_validate(vote, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [30427.20618655525, 6822.082269573937]
# [Mean R^2, Std dev. of R^2] --> [0.8489665847775353, 0.05957521819346141]
# Ridge is better than DecisionTreeRegressor based on r2 and RMSE metrics calculated above.
# VotingRegressor is better than Ridge r2 and RMSE metrics calculated above.



# Below, RandomForestRegressor is used:
parameter_grid = {'n_estimators': [50, 100, 150], 'max_depth': np.arange(7)[1:], 'max_features': np.linspace(0, 1, 11)[1:]}
rforest = RandomForestRegressor(random_state=42, bootstrap=False)
grsearch = GridSearchCV(estimator=rforest, param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'max_depth': 6, 'max_features': 0.30000000000000004, 'n_estimators': 150}


rforest_1 = RandomForestRegressor(max_depth=6, max_features=0.30000000000000004, n_estimators=150, random_state=42, bootstrap=False, n_jobs=-1)
scores = cross_validate(rforest_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [30611.42802866335, 3848.4340760220175]
# [Mean R^2, Std dev. of R^2] --> [0.8503039424933926, 0.022227887603178842]
# RandomForestRegressor is better than VotingRegressor based on r2 and RMSE metrics calculated above.




# Up to now, Ridge is better than all the other single estimators based on r2 and RMSE metrics.
# I will use RidgeRegressor as the base estimator within BaggingRegressor below:
# Below, BaggingRegressor is used:
ridge = Ridge(random_state=42, max_iter=1000, tol=0.001)
parameter_grid = {'n_estimators': [50, 100, 150], 'max_features': np.linspace(0, 1, 11)[1:]}
bag = BaggingRegressor(base_estimator=ridge, random_state=42, bootstrap=True, bootstrap_features=False, n_jobs=-1)
grsearch = GridSearchCV(estimator=bag, param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'max_features': 0.5, 'n_estimators': 50}

bag_1 = BaggingRegressor(base_estimator=ridge, n_estimators=50, random_state=42, max_features=0.5, bootstrap=True, bootstrap_features=False, n_jobs=-1)
scores = cross_validate(bag_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [30210.64037792497, 7573.846054826548]
# [Mean R^2, Std dev. of R^2] --> [0.8496047591515061, 0.06733898273009267]
# RandomForestRegressor is better than BaggingRegressor based on r2 and RMSE metrics.
# In terms of mean RMSE score, BaggingRegressor is slightly better than RandomForestRegressor, however its RMSE std dev. is much more than
# that of RandomForestregressor. Since difference in mean RMSE score is negligible, it can be said that RandomForestRegressor is better.



# Below, AdaBoostRegressor is used:
ridge = Ridge(random_state=42, max_iter=1000, tol=0.001)
parameter_grid = {'n_estimators': [50, 100, 150, 200, 250], 'learning_rate': np.linspace(0, 0.01, 11)[1:]}
adaboostreg = AdaBoostRegressor(base_estimator=ridge, random_state=42)
grsearch = GridSearchCV(estimator=adaboostreg, param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'learning_rate': 0.001, 'n_estimators': 50}


adaboostreg_1 = AdaBoostRegressor(base_estimator=ridge, n_estimators=50, learning_rate=0.001, random_state=42)
scores = cross_validate(adaboostreg_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [30920.03560531548, 7874.597238860882]
# [Mean R^2, Std dev. of R^2] --> [0.8416716070951793, 0.07471977762211392]
# RandomForestRegressor is better than AdaBoostRegressor based on r2 and RMSE metrics.





# Below, GradientBoostingRegressor is used:
parameter_grid = {'learning_rate': np.linspace(0.004, 0.01, 7)[1:], 'n_estimators': [50, 100], 'subsample': np.linspace(0, 1, 5)[1:], 'max_depth': np.arange(9)[2:], 'max_features': np.linspace(0, 1, 6)[1:]}
gradientboostingreg = GradientBoostingRegressor(random_state=42)
grsearch = GridSearchCV(estimator=gradientboostingreg, param_grid=parameter_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grsearch.fit(house_data_1[training_columns], house_data_1['SalePrice'])
grsearch.best_params_ # {'learning_rate': 0.01, 'max_depth': 8, 'max_features': 0.6000000000000001, 'n_estimators': 100, 'subsample': 1.0}


gradientboostingreg_1 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100, max_depth=8, max_features=0.6000000000000001, subsample=1.0, random_state=42)
scores = cross_validate(gradientboostingreg_1, house_data_1[training_columns], house_data_1['SalePrice'], cv=5, scoring=['neg_mean_squared_error', 'r2'], return_train_score=False)
[np.mean(np.sqrt(abs(scores['test_neg_mean_squared_error']))), np.std(np.sqrt(abs(scores['test_neg_mean_squared_error'])))]
[np.mean(scores['test_r2']),np.std(scores['test_r2'])]
# [Mean RMSE, Std dev. of RMSE] --> [40798.36737611999, 3783.869218260222]
# [Mean R^2, Std dev. of R^2] --> [0.7338714394310952, 0.014370593737952776]
# RandomForestRegressor is better than GradientBoostingRegressor based on r2 and RMSE metrics.


# RandomForestRegressor is the best among all the estimators based on both r2 and RMSE scores.
# So, it will be used in fitting and predicting in the final model.






# House_data için aşağıdaki işlemleri yaparak house_data'yı final model2de kullan.
house_data.drop(columns='GarageYrBlt', inplace=True)
house_data = pd.get_dummies(data=house_data)

train_data = house_data.loc[house_data['Id'] <= 1460]
prediction_data = house_data.loc[house_data['Id'] > 1460]

train_data.drop(columns='Id', inplace=True)
prediction_data.drop(columns='Id', inplace=True)

training_columns = train_data.columns.values.tolist()
# training_columns.index('SalePrice') --> 30
del training_columns[30]

prediction_columns = prediction_data.columns.values.tolist()
# prediction_columns.index('SalePrice') --> 30
del prediction_columns[30]


randomforestregressor = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, max_features=0.30000000000000004,  bootstrap=False)
randomforestregressor.fit(train_data[training_columns], train_data['SalePrice'])
predictions_y = randomforestregressor.predict(prediction_data[prediction_columns])

# Predictions are formtatted based on the specified upload format of the file:
output = pd.DataFrame({'Id': house_data_test['Id'].values.tolist(), 'SalePrice': predictions_y.tolist()})
output.to_csv('/kaggle/working/House_Price_Predictions.csv', sep=',', index=False, header=True)

