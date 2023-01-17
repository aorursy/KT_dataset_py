import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
train = "../input/train.csv"
test = "../input/test.csv"

train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

train_df_id = train_df['Id']
train_df = train_df.drop(columns = 'Id')

test_df_id = test_df['Id']
test_df = test_df.drop(columns = 'Id')

print(test_df.head())
#Checking GrLivArea wrt to SalePrice
fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.grid(True)
plt.show()
correlated = train_df.corr()
high_corelated_variable = correlated[correlated['SalePrice'] > 0.5].index
print(high_corelated_variable)
plt.figure(figsize = (10,10))
g = sns.heatmap(train_df[high_corelated_variable].corr(), annot=True)
#Deleting the 2 Outlier vlues which might affect the traiing of our model
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
Y_col = train_df['SalePrice']

train_df = train_df.drop(columns = 'SalePrice')

all_data = pd.concat((train_df, test_df), sort=False)

print(test_df.shape)
print(train_df.shape)
# Check if any column contains a NaN value
#print(df.isnull().any())

# Provides a NaN count for each column.
#print(df.isnull().sum())

# Checking % data is null set in the data columns
null_col = (all_data.isnull().sum()/len(all_data)) * 100
null_col = null_col.sort_values(ascending=False)
#print(null_col[:40])
#print(all_data.isnull().sum().sort_values(ascending=False)[:20])

# Remove column if all value in it are NaN.
#df = df.dropna(axis=1, how='all')

# Remove row if all value in it are NaN.
#df = df.dropna(axis=0, how='all')

#df = df.isnull().sum()
#print (df)
# Dropping columns which has mostly all data cell as Null
all_data = all_data.drop(columns = {'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'})
print(all_data.columns)
#Filling LotForntage cell with median value
LotFrontAge_Col_Grouped = all_data.groupby("Neighborhood")["LotFrontage"]

#Check each tuple values
#for item in LotFrontAge_Col_Grouped:
    #print(item)
    
all_data['LotFrontage'] = LotFrontAge_Col_Grouped.transform(lambda x : x.fillna(x.median()))

#Check NaN values
print(all_data['LotFrontage'].isnull().sum())
#Filling Garage related columns missing value with 0
garage_col = ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond')

print(type(garage_col))
for col in garage_col:
    all_data[col] = all_data[col].fillna('None')
    
for col in garage_col:
    null_value_in_col = all_data[col].isnull().sum()
    print("Null Values in {} is {}".format(col, null_value_in_col))
#Filling Basement related columns missing value with 0
bsmt_cols = ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2')
for col in bsmt_cols:
    all_data[col]=all_data.fillna('None')
    
for col in bsmt_cols:
    null_value_in_col = all_data[col].isnull().sum()
    print("Null Values in {} is {}".format(col, null_value_in_col))
#Filling Masonry veneer type and area with None and 0 repectively
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
#Filling columns with mode values where very few values are missing.
col_for_mode_fill = ('MSZoning', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Utilities', 'SaleType', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'Electrical')

for col in col_for_mode_fill:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])


for col in ('TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageCars', 'GarageArea'):
    all_data[col] = all_data[col].fillna(0)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(int)
#Performing label encoading to categorical columns
from sklearn.preprocessing import LabelEncoder
categorical_columns = all_data.select_dtypes(exclude=["int64","float64"]).columns
print(categorical_columns[:])

all_data = pd.get_dummies(all_data)
print(all_data.shape)
print(all_data.columns)
ntrain = train_df.shape[0]
train_data = all_data[:ntrain]
test_data = all_data[ntrain:]
print(train_data.shape)
print(Y_col.shape)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

X_train, X_test, Y_train, Y_test = train_test_split(train_data, Y_col)

def train_fit_model(model, train_values_X, train_values_Y, test_values_X):
    model.fit(train_values_X, train_values_Y)
    return model.predict(test_values_X)

def printMetricsData(predicted_data, test_values_Y):
    print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(test_values_Y, predicted_data)))
    print('R2 score: %.2f' % r2_score(test_values_Y, predicted_data))
    print('Median absolute error: %.2f' % median_absolute_error(test_values_Y, predicted_data))
regr = linear_model.LinearRegression()
printMetricsData(train_fit_model(regr, X_train, Y_train, X_test),Y_test) 
regr_ridge = linear_model.Ridge(alpha = .23, fit_intercept=False, random_state = 5, solver= 'auto', normalize=False)
printMetricsData(train_fit_model(regr_ridge, X_train, Y_train, X_test),Y_test) 
regr_ridge = linear_model.Lasso(fit_intercept=False, normalize=False, max_iter = 2000, random_state= 10, tol = .01)
printMetricsData(train_fit_model(regr_ridge, X_train, Y_train, X_test),Y_test) 
regr = linear_model.Ridge(alpha = .23, fit_intercept=False, random_state = 5, solver= 'auto', normalize=False)
regr.fit(train_data, Y_col)
kaggle_predict = regr.predict(test_data)

kaggle_df = pd.DataFrame()
kaggle_df['Id'] = test_df_id
kaggle_df['SalePrice'] = kaggle_predict
kaggle_file_csv = kaggle_df.to_csv("kaggle_df.csv", index=None)
