# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt #plottting

import seaborn as sns # more plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load in both the training and test data 

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



# Add the missing column to the test data but just fill it with NaNs so we can separate it later

test_data['SalePrice'] = np.nan



# Combine the two dataframes

all_data = pd.concat([train_data, test_data])



# Check the lengths are as expected

print("Rows in train data = {}".format(len(train_data)))

print("Rows in test data = {}".format(len(test_data)))
# Check for NaNs in the data

print("NaNs in each training Feature")

dfNull = all_data.isnull().sum().to_frame('nulls')

print(dfNull.loc[dfNull['nulls'] > 0]) # Print only features that have Null values
# Drop the features with lots of NaNs

all_data.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)
# Create mapping which works for Kitchen, Garage and Basement

mapping = {'Ex': 5.0, 'Gd': 4.0, 'TA': 3.0, 'Fa': 2.0, 'Po': 1.0}



# Remap values

all_data.replace({'KitchenQual': mapping}, inplace=True)

all_data.replace({'BsmtQual': mapping}, inplace=True)

all_data.replace({'BsmtCond': mapping}, inplace=True)

all_data.replace({'GarageQual': mapping}, inplace=True)

all_data.replace({'GarageCond': mapping}, inplace=True)
# Plot the distributions

keys = ['MSZoning', 'Utilities', 'Electrical', 'Functional']

fig, axs = plt.subplots(4,1, figsize=(8,16))

axs = axs.flatten()

for i in range(len(keys)):

    all_data[keys[i]].value_counts().plot(kind='bar', rot=20.0, fontsize=16, color='darkblue', 

                                          title=keys[i], ax=axs[i])

fig.subplots_adjust(hspace=0.5)

plt.show()
# MSZoning: Fill the NaNs with RL

all_data['MSZoning'].fillna('RL', inplace=True)



# Utilities: Drop the feature

all_data.drop(columns=['Utilities'], inplace=True)



# Electrical: Fill the NaNs with SBrkr

all_data['Electrical'].fillna('SBrkr', inplace=True)



# Functional: Fill the NaNs with Typ

all_data['Functional'].fillna('Typ', inplace=True)
# Plot the distributions

all_data['Exterior1st'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='Exterior1st')

plt.show()



all_data['Exterior2nd'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='Exterior2nd')

plt.show()
# Check if both exteriors are missing for the same property

print(all_data['Exterior2nd'].values[all_data['Exterior1st'].isnull() == True])
# Fill the NaNs with VinylSd

all_data['Exterior1st'].fillna('VinylSd', inplace=True)

all_data['Exterior2nd'].fillna('VinylSd', inplace=True)
# Plot the distribution

all_data['MasVnrType'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='MasVnrType')

plt.show()
# Find the vales of MasVnrArea when MasVnrType is Nan

print(all_data['MasVnrArea'].values[all_data['MasVnrType'].isnull() == True])
# Find the indices that correspond to the NaNs in MasVnrType but values in MasVnrArea (i.e. the MasVnrArea=198 index)

indices = (np.argwhere((all_data['MasVnrType'].isnull().values) & (all_data['MasVnrArea'].isnull().values == False))).flatten()

print(indices)



# Now fill the MasVnrType corresponding to MasVnrArea != NaN with BrkFace

all_data.iloc[indices[0], all_data.columns.get_loc('MasVnrType')] = 'BrkFace'



# And fill the remaining MasVnrType and MasVnrArea

all_data['MasVnrType'].fillna('None', inplace=True)

all_data['MasVnrArea'].fillna(0.0, inplace=True)
# Plot the distributions

keys = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','BsmtFullBath', 'BsmtHalfBath']

fig, axs = plt.subplots(4,2, figsize=(12,16))

axs = axs.flatten()

for i in range(len(keys)):

    all_data[keys[i]].value_counts().plot(kind='bar', rot=20.0, fontsize=16, color='darkblue', 

                                          title=keys[i], ax=axs[i])

fig.subplots_adjust(hspace=0.5)

fig.delaxes(axs[-1]) # remove the extra subplot (only need 7 but created 8)

plt.show()
# Fill the BsmtCond values with 3.0

all_data['BsmtCond'].fillna(3.0, inplace=True)



# Fill the rest

all_data['BsmtQual'].fillna(3.0, inplace=True)

all_data['BsmtExposure'].fillna('No', inplace=True)

all_data['BsmtFinType1'].fillna('Unf', inplace=True)

all_data['BsmtFinType2'].fillna('Unf', inplace=True)

all_data['BsmtFullBath'].fillna(0.0, inplace=True)

all_data['BsmtHalfBath'].fillna(0.0, inplace=True)
# Plot the distribution

all_data['KitchenQual'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='KitchenQual')

plt.show()
# Find the 5 largest correlations to KitchenQual

bigCorr = all_data.corr().nlargest(5, 'KitchenQual')['KitchenQual']

print(bigCorr)



# Find the 5 largest anti-correlations to KitchenQual

bigAnti = all_data.corr().nsmallest(5, 'KitchenQual')['KitchenQual']

print(bigAnti)
# Calculate the mean of the KitchenQual feature and round to the nearest integer (i.e. nearest quality category)

meanKitchenQual = np.rint(all_data.groupby(['OverallQual'])['KitchenQual'].mean())

print(meanKitchenQual)



# Check what OverallQual is for the missing KitchenQual value

print(all_data['OverallQual'].values[np.isnan(all_data['KitchenQual'].values)])



# Fill the missing KitchenQual value

all_data['KitchenQual'].fillna(3.0, inplace=True)
# Plot the categorical features

fig, axs = plt.subplots(3,2, figsize=(13,15))

axs = axs.flatten()

keys = ['GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond']

for i in range(len(keys)):

    all_data[keys[i]].value_counts().plot(kind='bar', rot=40.0, fontsize=16, color='darkblue', 

                                                   title=keys[i], ax=axs[i])

fig.subplots_adjust(hspace=0.5)

fig.delaxes(axs[-1]) # remove the extra subplot (only need 5 but created 6)

plt.show()
# Fill with most common values 

all_data['GarageType'].fillna('Attchd', inplace=True)

all_data['GarageFinish'].fillna('Unf', inplace=True)

all_data['GarageCars'].fillna(2.0, inplace=True)

all_data['GarageQual'].fillna(3.0, inplace=True)

all_data['GarageCond'].fillna(3.0, inplace=True)
# Sort this plotting out so it looks good

keys = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

# Plot the continuous distributions

fig, axs = plt.subplots(2,2, figsize=(12,12))

axs = axs.flatten()

for i in range(len(keys)):

    all_data[keys[i]].hist(color='darkblue', ax=axs[i], bins=40)

    axs[i].set_title(keys[i])

fig.subplots_adjust(hspace=0.5)

plt.show()
# Check which indices are NaN for each features

for i in keys:

    print(np.argwhere(np.isnan(all_data[i].values)))
# Calculate median

median_total = np.median(all_data['TotalBsmtSF'].values[~np.isnan(all_data['TotalBsmtSF'].values)])

median_SF1 = np.median(all_data['BsmtFinSF1'].values[~np.isnan(all_data['BsmtFinSF1'].values)])

median_Unf = np.median(all_data['BsmtUnfSF'].values[~np.isnan(all_data['BsmtUnfSF'].values)])

print(median_total, median_SF1, median_Unf)



# Fill the missing values

all_data['BsmtFinSF1'].fillna(median_SF1, inplace=True)

all_data['BsmtFinSF2'].fillna(0.0, inplace=True)

all_data['BsmtUnfSF'].fillna(median_Unf, inplace=True)

all_data['TotalBsmtSF'].fillna(median_total, inplace=True)
keys = [ 'GarageYrBlt', 'GarageArea']

# Plot the distribution

all_data['GarageArea'].hist(color='darkblue', bins=20)

plt.show()
# Find the NaNs

garage_yr_built = all_data['GarageYrBlt'].values

indices = np.argwhere(np.isnan(garage_yr_built))



# Update the data

yearbuilt = all_data['YearBuilt'].values[indices]

garage_yr_built[indices] = yearbuilt



# Save to the dataframe

all_data['GarageYrBlt'] = garage_yr_built





# Fill area with median

median = np.median(all_data['GarageArea'].values[~np.isnan(all_data['GarageArea'].values)])

print(median)

all_data['GarageArea'].fillna(median, inplace=True)
# Plot the distribution

all_data['SaleType'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='SaleType')

plt.show()



# Fill with the most common value

all_data['SaleType'].fillna('WD', inplace=True)
# Fill Frontage with median

median = np.median(all_data['LotFrontage'].values[~np.isnan(all_data['LotFrontage'].values)])

print(median)

all_data['LotFrontage'].fillna(median, inplace=True)
# Check everything has been filled as expected

print(all_data.isnull().sum().values)
# ==== Get features ready for one-hot encoding

# MoSold

mapping = {1: 'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 

           8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

all_data.replace({'MoSold': mapping}, inplace=True)



# MSSubClass

mapping = {20:'A', 30:'B', 40:'C', 45:'D', 50:'E', 60:'F', 70:'G', 75:'H', 80:'I', 85:'J', 

           90:'K', 120:'L', 150:'M', 160:'N', 180:'O', 190:'P'}

all_data.replace({'MSSubClass': mapping}, inplace=True)





# ==== Create some new features

all_data['sale_age'] = 2020 - all_data['YrSold'] 

all_data['house_age'] = 2020 - all_data['YearBuilt'] 

all_data['remodel_age'] = 2020 - all_data['YearRemodAdd']

all_data['garage_age'] = 2020 - all_data['GarageYrBlt']

all_data.drop(columns=['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=True)





all_data['TotalArea'] = (all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

                         + all_data['GrLivArea'] + all_data['GarageArea'])

all_data['TotalBathrooms'] = all_data['FullBath'] + all_data['HalfBath']*0.5 
# Specify which columns to one hot encode

columns = ['MSZoning', 'MSSubClass', 'PavedDrive', 'GarageFinish', 'Foundation', 'Functional', 'LandContour', 'Condition1', 

           'Condition2', 'Street', 'LotShape', 'ExterQual', 'ExterCond', 'LotConfig', 'LandSlope', 'Neighborhood','BldgType',

           'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType', 'BsmtExposure', 'BsmtFinType1',

           'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition', 

           'MoSold']



# Do the encoding

one_hot = pd.get_dummies(all_data.loc[:, columns], drop_first=True)

all_data.drop(columns=columns, inplace=True)

for i in range(len(one_hot.columns.values)):

    all_data[one_hot.columns.values[i]] = one_hot[one_hot.columns.values[i]].values
# Separate the data into prediction and model data

predict_data = all_data.loc[np.isnan(all_data['SalePrice'].values)]

model_data = all_data.loc[~np.isnan(all_data['SalePrice'].values)]

predict_ids = predict_data['Id'].values



# Remove ID so that the model is not trained on this. We add this back into the data before submission

predict_data.drop(columns=['Id'], inplace=True)

model_data.drop(columns=['Id'], inplace=True)
# Remove outliers from the training data 

from collections import Counter



def detect_outliers(df,n,features):

    outlier_indices = []

    for col in features:

        Q1 = np.percentile(df[col], 25) # 1st quartile (25%)

        Q3 = np.percentile(df[col],75) # 3rd quartile (75%)

        IQR = Q3 - Q1 # Interquartile range (IQR)

        outlier_step = 1.5 * IQR # outlier step

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

    # select observations containing more than n outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers 



# Detect outliers from SalePrice, LotArea, GarageArea

Outliers_to_drop = detect_outliers(model_data,1,["SalePrice", 'LotArea', 'GarageArea'])

model_data = model_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
# Scale the data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(model_data.drop(columns=['SalePrice']))

X_train = scaler.transform(model_data.drop(columns=['SalePrice']))

y_train = model_data['SalePrice'] 



X_predict_data_scaled = scaler.transform(predict_data.drop(columns=['SalePrice']))
# Import some required modules for the fitting

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
# Search for the optimal hyperparameters for RF

param_grid = { 

    'criterion' : ['mse'],

    'n_estimators': [90, 100, 110],

    'max_features': ['auto', 'log2'],

    'max_depth' : [7, 9, 11, 13]    

                }



randomForest_CV = GridSearchCV(estimator = RandomForestRegressor(), param_grid = param_grid, cv = 3)

randomForest_CV.fit(X_train, y_train)

print(randomForest_CV.best_params_)



# Print the best score

print(randomForest_CV.best_score_)



# Get the parameters from the best fit

criterion = randomForest_CV.best_params_.get('criterion')

nRF = randomForest_CV.best_params_.get('n_estimators')

max_features = randomForest_CV.best_params_.get('max_features')

max_depth = randomForest_CV.best_params_.get('max_depth')



# Train the classifier on all the data for predicting the test data (use best hyperparams)

bestRF = RandomForestRegressor(n_estimators=nRF, max_depth=max_depth, max_features=max_features, criterion=criterion)

bestRF.fit(X_train, y_train)
import xgboost as xgb

xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                    max_depth=3)

xgboost.fit(X_train,y_train)
# Use the models to predict the output of the test data

testPredictionsRF = bestRF.predict(X_predict_data_scaled)

testPredictionsXGB = xgboost.predict(X_predict_data_scaled)



# Save the outputs to csv files

outputRF = pd.DataFrame({'Id': predict_ids, 'SalePrice': testPredictionsRF})

outputRF.to_csv('my_submissionRF.csv', index=False)



outputXGB = pd.DataFrame({'Id': predict_ids, 'SalePrice': testPredictionsXGB})

outputXGB.to_csv('my_submissionXGB.csv', index=False)