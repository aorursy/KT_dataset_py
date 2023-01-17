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
#Open Dataframe
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
#Check columns dtypes and NaNs
print('Dataframe shape: ', df.shape, '\n')
print('Dataframe columns: ', df.columns.tolist(), '\n')
print('Dataframe column dtypes: ', df.dtypes, '\n')
#percentage of NaN values in df

pd.set_option('display.max_rows',1000)
print('Total rows- Before Cleaning: ', df.shape[0])
print('Total Columns- Before Cleaning: ', df.shape[1], '\n')
nan_percent = (df.isnull().sum()/len(df))*100
print('Percentage of NaN values: \n', nan_percent)
#Drop columns with < 70% NaN values using thresh argument in df.dropna
df_no_nan = df.dropna(thresh=800, axis=1)

#Impute the missing values in the columns with <70% NaN values
df_no_nan = df_no_nan.fillna(method='ffill')
print('Total rows- After Cleaning: ', df_no_nan.shape[0])
print('Total columns- After Cleaning: ', df_no_nan.shape[1], '\n')
print('Number of NaN values in the columns:\n',df_no_nan.isnull().sum())
#Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
#Identifying Groups of Features
location = ['MSZoning' , 'Neighborhood', 'Street', 'LandSlope']
shape = ['LotShape', 'LandContour', 'LandSlope', 'BldgType', 'HouseStyle']
room_quality = ['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']
amenities = ['Utilities', 'Heating', 'HeatingQC', 'Electrical', 'MasVnrType', 'CentralAir']
built_value = ['YearBuilt', 'YearRemodAdd', 'YrSold']
add_value = ['MiscVal', 'GarageType', 'Functional']
quality = ['BsmtCond', 'KitchenQual', 'OverallQual', 'OverallCond', 'GarageQual', 'GarageCond']
sale_value = ['SaleType', 'SaleCondition']

found_cat_list = [location, shape, room_quality, amenities, built_value, add_value, sale_value]
#Subsetting df_no_nan for Visualizing influence of Grouped Features
loc_df = df_no_nan.loc[:, ['MSZoning' , 'Neighborhood', 'Street', 'LandSlope','SalePrice']]
shape_df = df_no_nan.loc[:, ['LotShape', 'LandContour', 'LandSlope', 'BldgType', 'HouseStyle', 'SalePrice']]
room_qual_df = df_no_nan.loc[:,['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'SalePrice']]
amenities_df = df_no_nan.loc[:, ['Utilities', 'Heating', 'HeatingQC', 'Electrical', 'MasVnrType', 'CentralAir', 'SalePrice']]
built_val_df = df_no_nan.loc[:, ['YearBuilt', 'YearRemodAdd', 'YrSold', 'OverallQual', 'OverallCond', 'SalePrice']]
add_val_df = df_no_nan.loc[:, ['MiscVal', 'GarageType', 'Functional', 'SalePrice']]
quality_df = df_no_nan.loc[:, ['BsmtCond', 'KitchenQual', 'OverallQual', 'OverallCond', 'GarageQual', 'GarageCond', 'SalePrice']]
sale_val_df = df_no_nan.loc[:, ['SaleType', 'SaleCondition', 'SalePrice']]

found_cat_df = [loc_df, shape_df, room_qual_df, amenities_df, built_val_df, add_val_df, quality_df, sale_val_df]
#Melting Location features
melt_loc = pd.melt(loc_df, id_vars='SalePrice', value_vars=location)
melt_loc.head()
#Plot stripplot with melt_loc
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_loc)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Location VS Price')
plt.show()
#Melting Shape features
melt_shape = pd.melt(shape_df, id_vars='SalePrice', value_vars=shape)
melt_shape.head()
#Plot stripplot with melt_shape dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_shape)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Shape of House VS Price')
plt.show()
#Melting Room features
melt_rooms = pd.melt(room_qual_df, id_vars='SalePrice', value_vars=room_quality)
melt_rooms.head()
#Plot strippplot for melt_rooms dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_rooms)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Rooms Quality VS Price')
plt.show()
#Melting Amenities features
melt_amenities = pd.melt(amenities_df, id_vars='SalePrice', value_vars=amenities)
melt_amenities.head()
#Plot strippplot for melt_amenities dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_amenities)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Amenities VS Price')
plt.show()
#Melting Building value features
melt_built_val = pd.melt(built_val_df, id_vars='SalePrice', value_vars=built_value)
melt_built_val.head()
#Plot strippplot for melt_built_val dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_built_val)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Building value VS Price')
plt.show()
#Melting Additional value features
melt_add_val = pd.melt(add_val_df, id_vars='SalePrice', value_vars=add_value)
melt_add_val.head()
#Plot strippplot for melt_add_val dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_add_val)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Additional value VS Price')
plt.show()
#Melting Quality features
melt_quality = pd.melt(quality_df, id_vars='SalePrice', value_vars=quality)
melt_quality.head()
#Plot strippplot for melt_quality dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_quality)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Quality VS Price')
plt.show()
#Melting Quality features
melt_sale_val = pd.melt(sale_val_df, id_vars='SalePrice', value_vars=sale_value)
melt_sale_val.head()
#Plot strippplot for melt_quality dataframe
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.stripplot(ax=ax, x='SalePrice', y='variable', hue='value', data=melt_sale_val)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Sale value VS Price')
plt.show()
loc_encoded = pd.get_dummies(loc_df, drop_first=True)
corr_data = loc_encoded.corr(method='spearman')


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
cmap_sns = sns.diverging_palette(5, 200, as_cmap=True)
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, cmap=cmap_sns)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
plt.title('Correlation Map')
plt.show()


#Import neccesary libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error as MSE

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#Splitting dataset into Target and Features
y = df_no_nan.SalePrice
X = df_no_nan.drop(['SalePrice'], axis=1)

#one-hot encoding categorical Features and reducing multicollinearity by dropping first category
X_encoded = pd.get_dummies(X, drop_first=True)

#Setting seed for reproducibility
SEED = 42
#Splitting dataset to 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=SEED)
#Instantiating Model
rf_reg = RandomForestRegressor(n_estimators=500, n_jobs=-1,
                               warm_start=True,
                              random_state=SEED)
et_reg = ExtraTreesRegressor(n_estimators=500, n_jobs=-1,
                               warm_start=True,
                              random_state=SEED)
vote_reg = VotingRegressor([('rf',rf_reg),('et', et_reg)], n_jobs=-1)

#Fitting the training data
vote_reg.fit(X_train, y_train)

#Predicting the test data
y_pred = vote_reg.predict(X_test)
#Evaluating Model accuracy
score = vote_reg.score(X_test, y_test)*100
rmse_test = MSE(y_test, y_pred)**(0.5)
print(f'Model Performance:\n Scoring:{score}\n rmse_score:{rmse_test}')
#print(cross_val_score(rf_reg, X_train, y_train, cv=10, n_jobs=-1))
#Import test data
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_nan_cols = test_df.columns[test_df.isnull().sum()>800]
test_nan_cols
df_nan_cols = df.columns[df.isnull().sum()>800]
df_nan_cols
#Default filling of NaN values
test_no_nan = test_df.dropna(thresh=700, axis=1)
test_no_nan = test_no_nan.fillna(method='ffill')
test_no_nan.isnull().sum()
#Encoding categorical data
test_encoded = pd.get_dummies(test_no_nan)
shapes = [df, df_no_nan, X_encoded, test_df, test_no_nan, test_encoded]
for i in shapes:
    print('Shape of df:{}'.format(i.shape))
#Predicting Test data
test_pred = vote_reg.predict(test_encoded)
pred_df = pd.DataFrame(test_pred, columns=['SalePrice'])
pred_df.head()
