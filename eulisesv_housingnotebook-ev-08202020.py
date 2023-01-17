import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

import numpy as np



# set the option for all columns to show when printing df



pd.set_option('display.max_columns', 200)



#cancel warnings



from warnings import filterwarnings

filterwarnings("ignore")
# import the data to be used as training



data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data.head(10)
data.describe()
# plot the average price of sale based on year built



plt.figure(figsize=(24,12))

avg_sale_price = sns.barplot(data[['YearBuilt','SalePrice']].groupby('YearBuilt').mean()['SalePrice'].index, 

                             data[['YearBuilt','SalePrice']].groupby('YearBuilt').mean()['SalePrice'],

                             color='blue')

avg_sale_price.set(ylabel='Avg Sale Price')

avg_sale_price.set_xticklabels(avg_sale_price.get_xticklabels(),rotation=60);
# plot the average price of sale based on neighborhood



plt.figure(figsize=(24,12))

avg_sale_price = sns.barplot(data[['Neighborhood','SalePrice']].groupby('Neighborhood').mean()['SalePrice'].index, 

                             data[['Neighborhood','SalePrice']].groupby('Neighborhood').mean()['SalePrice'],

                             color='blue')

avg_sale_price.set(ylabel='Avg Sale Price')

avg_sale_price.set_xticklabels(avg_sale_price.get_xticklabels(),rotation=60);
# plot the count of values in each column



plt.figure(figsize=(24,12))

missing_plot = sns.barplot(data.columns, 

                             data.count(),

                             color='blue')

missing_plot.set_xticklabels(missing_plot.get_xticklabels(),rotation=90);
# fill in some of the missing values either with mean for numerical columns or some other categorical value



# numerical columns fill in with mean

data.fillna(value = data.mean(), inplace=True)



# columns that are categorical we will fill in with None

none_fillin = [i for i,j in zip(data.count().index, data.count()) if j != 1460]



data.fillna(value='None', inplace = True)

final_data = data.copy()
# # drop the records with missing values except for 'LotFrontage'



# # dropped_df = df_filled_in.dropna(subset=['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'Alley',

# #        'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

# #        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

# #        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

# #        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

# #        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

# #        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

# #        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',

# #        'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

# #        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

# #        'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

# #        'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',

# #        'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',

# #        'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

# #        'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',

# #        'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'])



# # for LotFrontage we are going to fill in missing values with the average for each neighborhood



# LotFrontage_Avg = pd.DataFrame(data.groupby('Neighborhood')['LotFrontage'].mean().round())

# temp_LotFrontage = dropped_df.merge(LotFrontage_Avg, how='outer', on='Neighborhood')



# new_LotFrontage = []

# for i,j in zip(temp_LotFrontage['LotFrontage_x'], temp_LotFrontage['LotFrontage_y']):

#     if math.isnan(i):

#         new_LotFrontage.append(j)    

#     else:

#         new_LotFrontage.append(i)

        

# # assign the new list to the LotFrontage variable        

# temp_LotFrontage['LotFrontage'] = new_LotFrontage



# #drop the LotFrontage_x, LotFrontage_y columns

# final_data = temp_LotFrontage.drop(['LotFrontage_x', 'LotFrontage_y'], axis=1).set_index('Id')
# plot the count of values in each column



plt.figure(figsize=(24,12))

missing_plot = sns.barplot(final_data.columns, 

                             final_data.count(),

                             color='blue')

missing_plot.set_xticklabels(missing_plot.get_xticklabels(),rotation=90);
# transform all the categorical columns into binary columns



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)



cat_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC',

       'CentralAir', 'Electrical',

       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond',

       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']



encoded = pd.DataFrame(encoder.fit_transform(final_data[cat_columns])).set_index(final_data.index)

encoded.columns = encoder.get_feature_names(final_data[cat_columns].columns)
# create a dataframe of the numerical columns only

num_columns = final_data.drop(['MSSubClass', 'MSZoning', 'Street', 'Alley',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC',

       'CentralAir', 'Electrical',

       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond',

       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition','SalePrice'], axis=1).columns



numerical_df = final_data[num_columns]



# join the dataframes of the enconded categorical columns 

# and the numerical columns on index



joined_df = numerical_df.join(encoded)
full_test_csv = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

full_test_csv.head()
# plot the count of values in each column



plt.figure(figsize=(24,12))

missing_plot = sns.barplot(full_test_csv.columns, 

                             full_test_csv.count(),

                             color='blue')

missing_plot.set_xticklabels(missing_plot.get_xticklabels(),rotation=90);
# fill in some of the missing values either with mean for numerical columns or some other categorical value



# numerical columns fill in with mean

full_test_csv.fillna(value = full_test_csv.mean(), inplace=True)



# columns that are categorical we will fill in with None

none_fillin = [i for i,j in zip(data.count().index, data.count()) if j != 1460]



full_test_csv.fillna(value='None', inplace = True)

final_test_csv_data = full_test_csv.copy()
# # fill in some of the missing values either with 0 or some other categorical value



# none_fillin = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']



# filling_in_df = full_test_csv[none_fillin].fillna(value='None')

# df_temp = full_test_csv.drop(none_fillin, axis=1)

# df_filled_in = df_temp.merge(filling_in_df, how='outer',left_index=True, right_index=True)
# # drop the records with missing values except for 'LotFrontage'



# dropped_df = df_filled_in.dropna(subset=['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'Alley',

#        'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

#        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

#        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

#        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

#        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

#        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',

#        'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

#        'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

#        'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',

#        'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',

#        'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

#        'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',

#        'MoSold', 'YrSold', 'SaleType', 'SaleCondition'])



# # for LotFrontage we are going to fill in missing values with the average for each neighborhood



# LotFrontage_Avg = pd.DataFrame(full_test_csv.groupby('Neighborhood')['LotFrontage'].mean().round())

# temp_LotFrontage = dropped_df.merge(LotFrontage_Avg, how='outer', on='Neighborhood')



# new_LotFrontage = []

# for i,j in zip(temp_LotFrontage['LotFrontage_x'], temp_LotFrontage['LotFrontage_y']):

#     if math.isnan(i):

#         new_LotFrontage.append(j)    

#     else:

#         new_LotFrontage.append(i)

        

# # assign the new list to the LotFrontage variable        

# temp_LotFrontage['LotFrontage'] = new_LotFrontage



# #drop the LotFrontage_x, LotFrontage_y columns

# final_test_csv_data = temp_LotFrontage.drop(['LotFrontage_x', 'LotFrontage_y'], axis=1).set_index('Id')
# plot the count of values in each column



plt.figure(figsize=(24,12))

missing_plot = sns.barplot(final_test_csv_data.columns, 

                             final_test_csv_data.count(),

                             color='blue')

missing_plot.set_xticklabels(missing_plot.get_xticklabels(),rotation=90);
# transform all the categorical columns into binary columns



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)



cat_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC',

       'CentralAir', 'Electrical',

       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond',

       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']



#convert columns to integers so we dont have floats in encoded names

final_test_csv_data['BsmtFullBath'] = final_test_csv_data['BsmtFullBath'].astype(int)

final_test_csv_data['GarageCars'] = final_test_csv_data['GarageCars'].astype(int)

final_test_csv_data['BsmtHalfBath'] = final_test_csv_data['BsmtHalfBath'].astype(int)



encoded_test_csv = pd.DataFrame(encoder.fit_transform(final_test_csv_data[cat_columns])).set_index(final_test_csv_data.index)

encoded_test_csv.columns = encoder.get_feature_names(final_test_csv_data[cat_columns].columns)
# create a dataframe of the numerical columns only

test_csv_num_columns = final_test_csv_data.drop(cat_columns, axis=1).columns



numerical_test_csv_df = final_test_csv_data[test_csv_num_columns]



# join the dataframes of the enconded categorical columns 

# and the numerical columns on index



joined_test_csv_df = numerical_test_csv_df.join(encoded_test_csv)
# scale the entire test.csv dataset

from sklearn.preprocessing import StandardScaler



scaler_test_csv = StandardScaler()



scaler_test_csv.fit(joined_test_csv_df[test_csv_num_columns])

scaled_test_csv = pd.DataFrame(scaler_test_csv.transform(joined_test_csv_df[test_csv_num_columns]), 

                                columns=joined_test_csv_df[test_csv_num_columns].columns, 

                                index=joined_test_csv_df.index)

test_csv_scaled_encoded = joined_test_csv_df[encoded_test_csv.columns].join(scaled_test_csv)
# create a df for columns missing from test_csv_scales_encoded



test_csv_missing_cols = joined_df.columns.difference(test_csv_scaled_encoded.columns) #columns in joined df, but missing in test_csv

test_csv_zeroes = np.zeros(shape= (len(test_csv_scaled_encoded.index), len(test_csv_missing_cols)))



df_missing_cols_test_csv = pd.DataFrame(test_csv_zeroes, columns= test_csv_missing_cols, index= test_csv_scaled_encoded.index)



test_csv_scaled_encoded = test_csv_scaled_encoded.join(df_missing_cols_test_csv)
# create a df for columns missing from joined_df

# which is the training data



train_csv_missing_cols = test_csv_scaled_encoded.columns.difference(joined_df.columns) #columns in joined df, but missing in test_csv

train_csv_zeroes = np.zeros(shape= (len(joined_df.index), len(train_csv_missing_cols)))



df_missing_cols_train_csv = pd.DataFrame(train_csv_zeroes, columns= train_csv_missing_cols, index= joined_df.index)



joined_df = joined_df.join(df_missing_cols_train_csv)
# split data into training and testing



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(joined_df, final_data['SalePrice'], test_size=0.20, random_state=42)
# scale the numerical columns



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



scaler.fit(X_train[num_columns])

scaled_X_train = pd.DataFrame(scaler.transform(X_train[num_columns]), columns=X_train[num_columns].columns, index=X_train.index)

scaled_X_test = pd.DataFrame(scaler.transform(X_test[num_columns]), columns=X_train[num_columns].columns, index=X_test.index)
X_train_scaled = X_train[encoded.columns].join(scaled_X_train)

X_test_scaled = X_test[encoded.columns].join(scaled_X_test)
# create a function to calc accuracy by doing 

# |actual - prediction| / actual * 100%



def GetAccuracy(actuals, predictions):

    return 100 - np.mean((abs(actuals - predictions) / actuals) * 100)    
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [100000, 150000, 250000],  

              'epsilon': [1000, 100], 

              'degree': [2, 3, 4]}



svm_grid = GridSearchCV(SVR(kernel='poly'), param_grid= param_grid, refit=True)

svm_grid.fit(X_train_scaled, y_train);
print(svm_grid.best_score_)
svm_grid.best_params_
svm_preds = svm_grid.predict(X_test_scaled)

GetAccuracy(y_test, svm_preds)
from sklearn.linear_model import ElasticNet, ElasticNetCV
eln_param_grid = {'alpha': [50, 100, 500, 1000],  

              'l1_ratio': [.5, 1]}



eln_grid = GridSearchCV(ElasticNet(), eln_param_grid, refit=True)

eln_grid.fit(X_train_scaled, y_train);
print(eln_grid.best_score_)
eln_grid.best_params_
eln_preds = eln_grid.predict(X_test_scaled)

GetAccuracy(y_test, eln_preds)
from sklearn.ensemble import GradientBoostingRegressor
gbr_param_grid = {'n_estimators': [50, 100, 150], 

                 'max_depth':[2, 4, 6, 8],

                 'learning_rate': [.1, .25, .5, 1]}



gbr_grid = GridSearchCV(GradientBoostingRegressor(), gbr_param_grid,  refit=True)

gbr_grid.fit(X_train_scaled, y_train);
print(gbr_grid.best_score_)
gbr_grid.best_params_
gbr_preds = gbr_grid.predict(X_test_scaled)

GetAccuracy(y_test, gbr_preds)
ensemble_preds = (svm_preds + eln_preds + gbr_preds)/3

GetAccuracy(y_test, ensemble_preds)
scaler_train_csv = StandardScaler()



scaler_train_csv.fit(joined_df[num_columns]) #joined df is the encoded train.csv data without scaling for num columns

scaled_train_csv = pd.DataFrame(scaler_train_csv.transform(joined_df[num_columns]), 

                                columns=joined_df[num_columns].columns, 

                                index=joined_df.index)

train_csv_scaled_encoded = joined_df[encoded.columns].join(scaled_train_csv)
# fit the 3 algorithms to the entire train.csv dataset



svm_grid.fit(train_csv_scaled_encoded, final_data['SalePrice'])

eln_grid.fit(train_csv_scaled_encoded, final_data['SalePrice'])

gbr_grid.fit(train_csv_scaled_encoded, final_data['SalePrice']);
svm_preds_test_csv = svm_grid.predict(test_csv_scaled_encoded[X_train_scaled.columns])

eln_preds_test_csv = eln_grid.predict(test_csv_scaled_encoded[X_train_scaled.columns])

gbr_preds_test_csv = gbr_grid.predict(test_csv_scaled_encoded[X_train_scaled.columns])
ensemble_preds_test_csv = (svm_preds_test_csv + eln_preds_test_csv + gbr_preds_test_csv)/3
test_preds = pd.DataFrame(ensemble_preds_test_csv, columns= ['SalePrice'], index= full_test_csv['Id']).reset_index()

test_preds.columns = ['Id', 'SalePrice']
test_preds
test_preds.to_csv("Housing_TestPredictions_EulisesValdovinos.csv", index=False)