import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from scipy import stats

from scipy.special import boxcox1p
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
print(test.shape)
test.head()
print(train.shape)
train.head()

y_train = train.SalePrice
trainID = train.Id
testID = test.Id

# Drop Id column since it is not useful for the model
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
# Evaluate Missing Values
count_nan = train.isna().sum()
missing_nan = count_nan[count_nan > 0].sort_values(ascending=False)

plt.figure()
plt.xticks(rotation='90')
plt.bar(missing_nan.index,missing_nan)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Number of missing values', fontsize=15)
plt.show()
# Evaluate Features with more than 15% Missing Values
count_nan = train.isna().sum()
missing_nan = count_nan[count_nan > train.shape[0] * 0.15].sort_values(
    ascending=False)

plt.figure()
plt.xticks(rotation='90')
plt.bar(missing_nan.index,missing_nan)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Number of missing values', fontsize=15)
plt.show()
# Remove columns with greater 15% missing values
drop_list = ["LotFrontage", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

train.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)
# Examine the shape of SalePrice Distribution
plt.figure()
sns.distplot(train.SalePrice.values)

plt.figure()
stats.probplot(train.SalePrice, plot=plt)
plt.show()
# The shape of the probability graph is not linear. 
# The sale price should be transform into log graph for better
# predictions.

mu, sigma = stats.norm.fit(train.SalePrice)
print('Previous mu = {:.2f} and sigma = {:.2f}'.format(mu, 
       sigma))
train.SalePrice = np.log1p(train.SalePrice)
# Examine the current shape of SalePrice Distribution
plt.figure()
sns.distplot(train.SalePrice.values)

plt.figure()
stats.probplot(train.SalePrice, plot=plt)
plt.show()

mu, sigma = stats.norm.fit(train.SalePrice)
print('Now mu = {:.2f} and sigma = {:.2f}'.format(mu, 
       sigma))
numerical = all_data.dtypes[all_data.dtypes != "object"].index

# apply Box Cox Transformation on skewed features
skewed = all_data[numerical].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewed_features = skewed[abs(skewed) > 0.75].index

lam = 0.15
for feature in skewed_features:
    all_data[feature] = boxcox1p(all_data[feature], lam)
# Now the probablity fit into a linear graph. We can move on
# to deal with missing values

# Let's combine the train and test data
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)
y_train = train.SalePrice
all_data.drop(['SalePrice'], axis=1, inplace=True)
missing_data = all_data.isnull().sum()

# Data frame with missing Data
missing_df = missing_data[missing_data > 0]
print("Missing Indices are: {}".format(missing_df.index.tolist()))
all_data[missing_df.index].dtypes
# See how many categorical and numerical features are in the missing indices
categorical = all_data[missing_df.index].dtypes[all_data[missing_df.index].dtypes == 'object'].index
numerical = all_data[missing_df.index].dtypes[all_data[missing_df.index].dtypes != 'object'].index

print("There are {} cat features and {} num features".format(
    len(categorical), len(numerical)))
# Replace categorical features' missing values to None
all_data[categorical] = all_data[categorical].fillna('None')

# Replace numerical features' missing values to its average
for feature in numerical:
    all_data[feature].fillna(all_data[feature].mean(), inplace=True)
# Check Missing Values again
all_data.isnull().sum().sum()
# See how many categorical and numerical features
categorical = all_data.dtypes[all_data.dtypes == 'object'].index
numerical = all_data.dtypes[all_data.dtypes != 'object'].index

print("There are {} cat features and {} num features".format(
    len(categorical), len(numerical)))
# Transform categorical features to one-hot-encoding
all_data[categorical].head()
from sklearn.preprocessing import LabelEncoder

def label_encode_categorical(df, categorical):
    df = df.copy()
    for feature in categorical:
        label = LabelEncoder()
        label.fit(df[feature].values)
        df[feature] = label.transform(df[feature].values)
    return df
all_data = label_encode_categorical(all_data, categorical)
test.head()
train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]
test.shape[0]
# Split train and validation set
x_train, x_valid, y_train, y_valid = train_test_split(train, y_train)
# Apply Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
lr = LinearRegression()
lr.fit(x_train, y_train)

# RMSE
y_train_pred = lr.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print(train_mse)

# plot
plt.figure()
plt.scatter(y_train, y_train_pred - y_train)

y_valid_pred = lr.predict(x_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
print(valid_mse)
lr_ridge = Ridge(alpha=1)
lr_ridge.fit(x_train, y_train)

# RMSE
y_train_pred = lr_ridge.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print(train_mse)

# plot
plt.figure()
plt.scatter(y_train, y_train_pred - y_train)

y_valid_pred = lr_ridge.predict(x_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
print(valid_mse)
lr_lasso = Lasso(alpha=0.002)
lr_lasso.fit(x_train, y_train)

# RMSE
y_train_pred = lr_lasso.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print(train_mse)

# plot
plt.figure()
plt.scatter(y_train, y_train_pred - y_train)

y_valid_pred = lr_lasso.predict(x_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
print(valid_mse)
test_scaled = scaler.transform(test)
test_pred = np.expm1(lr_ridge.predict(test_scaled))

ridge_submission = pd.DataFrame()
ridge_submission['ID'] = testID
ridge_submission['SalePrice'] = test_pred
ridge_submission.to_csv('ridge_submission.csv', index=False)
test_scaled = scaler.transform(test)
test_pred = np.expm1(lr_lasso.predict(test_scaled))

lasso_submission = pd.DataFrame()
lasso_submission['ID'] = testID
lasso_submission['SalePrice'] = test_pred
lasso_submission.to_csv('lasso_submission.csv', index=False)
lasso_submission

