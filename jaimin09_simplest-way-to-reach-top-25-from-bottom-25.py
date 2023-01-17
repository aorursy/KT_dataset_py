import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
pd.set_option('display.max_columns', None)               # To get all the columns
train.head()
def describe_features_with_null_values(df):
    null_count = df.isnull().sum()
    null_values = null_count[null_count > 0]

    null_values = null_values.sort_values(ascending = False)
    
    null_perc = null_values*100/len(df)

    null = pd.DataFrame(null_values, columns = ['Null Count'])
    null['Percentage'] = round(null_perc, 2)

    return null
df_null = describe_features_with_null_values(train)
df_null
higher_null_values_list = list(df_null[df_null['Percentage'] > 15].index)
print("Features having more than 15% Null values are :", higher_null_values_list)
train = train.drop(higher_null_values_list, axis = 1)
corrmat = train.corr()        # Finds correlation between all the columns
f, ax = plt.subplots(figsize=(12, 9))             # Increases the figure size to (12, 9)
sns.heatmap(corrmat, vmax = 0.8, square=True);
train = train.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt'], 1)
df_null = describe_features_with_null_values(train)
df_null
def fill_null_values(df):
    
    null_df = describe_features_with_null_values(df)
    
    # Below 2 lines will give us features with object, float/int datatype respectively.
    obj_features = df[null_df.index].dtypes[df[null_df.index].dtypes == object].index
    float_features = df[null_df.index].dtypes[df[null_df.index].dtypes == float].index
        
    for feature in obj_features:
        df[feature] = df[feature].fillna(df[feature].mode().values[0])
    
    for feature in float_features:
        df[feature] = df[feature].fillna(df[feature].mean())
        
    return df
train = fill_null_values(train)
print("Features which most affects to our SalePrice : \n")
related_cols = corrmat.nlargest(10, 'SalePrice')
print(related_cols['SalePrice'])
fig, axes = plt.subplots(2,2, figsize = (8,7))

axes[0][0].scatter(train['OverallQual'], train['SalePrice'])

axes[0][1].scatter(train['GrLivArea'], train['SalePrice'])

axes[1][0].scatter(train['GarageCars'], train['SalePrice'])

axes[1][1].scatter(train['TotalBsmtSF'], train['SalePrice'])

fig.tight_layout()
ind1 = train['TotalBsmtSF'][train['TotalBsmtSF'] > 5000].index.values
ind2 = train['GrLivArea'][train['GrLivArea'] > 4500].index.values

print('Index of datapoint in TotalBsmtSF different from croud :', ind1)
print('Index of datapoint in GrLivArea different from croud :', ind2)
train = train.drop(ind2)
sns.distplot(train['SalePrice']);
# Applying Normality
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice']);
obj_mask = train.dtypes == object
obj_features = list(obj_mask[obj_mask].index)

le = LabelEncoder()
train[obj_features] = train[obj_features].apply(le.fit_transform)
train.head()
Y_train = train['SalePrice'].values
X_train = train.drop(['SalePrice'], 1).values
print("Shape of X_train :", X_train.shape)
print("Shape of Y_train :", Y_train.shape)
lr = LinearRegression()
lr.fit(X_train, Y_train)

ypred = np.exp(lr.predict(X_train))

# Note : we are applying np.exp(). Thats because our Y_train is Normalized by applying Logarithm

err = round(mean_absolute_error(np.exp(Y_train), ypred)/100000, 5)

print("Error Score for Training Set :", err)
test = test.drop(higher_null_values_list, 1)
test = test.drop(['1stFlrSF', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'], 1)
test = fill_null_values(test)
obj_mask_test = test.dtypes == object
obj_features_test = list(obj_mask_test[obj_mask_test].index)

le = LabelEncoder()
test[obj_features_test] = test[obj_features_test].apply(le.fit_transform)
X_test = test.values
test_pred = np.exp(lr.predict(X_test))
test_ind = np.arange(1461, 1461 + len(test))
test_series = pd.Series(test_pred, index = test_ind)
#test_series.to_csv('predictions.csv')
