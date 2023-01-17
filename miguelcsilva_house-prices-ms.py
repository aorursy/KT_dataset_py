# Import dependencies

%matplotlib inline



# Data manipulation

import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Statistics

from scipy import stats



# Models

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Warnings

import warnings

warnings.filterwarnings("ignore")


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sale_price_id = df_test['Id']

df_all = df_train.append(df_test)

df_all.drop('SalePrice', axis = 1, inplace = True)

df_all.reset_index(inplace = True)

list_dfs = [df_train, df_test, df_all]

print(f"Length of df_train dataset: {len(df_train)}")

print(f"Length of df_test dataset: {len(df_test)}")

print(f"Length of df_train + df_test dataset: {len(df_train) + len(df_test)}")

print(f"Length of df_all dataset: {len(df_all)}")

print(f"Number of datasets in list_dfs: {len(list_dfs)}")
print(f"The training set has {len(df_train.columns)} features")

print(f"The testing set has {len(df_test.columns)} features")
df_train['SalePrice'].describe()
fig, ax = plt.subplots(1, 1)

sns.distplot(df_train['SalePrice'], label = "Sale Price", fit = stats.norm)

ax.legend()
print(f"Skewness: {df_train['SalePrice'].skew()}")

print(f"Kurtosis: {df_train['SalePrice'].kurt()}")
correlation_matrix = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(correlation_matrix, vmax = 0.8, square = True);
most_correlated_columns = correlation_matrix.nlargest(11, 'SalePrice').index

most_correlated_matrix = df_train[most_correlated_columns].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(most_correlated_matrix, annot=True, square=True, fmt='.2f')
discarded_features = ['SalePrice', 'GarageArea', '1stFlrSF']

selected_features = list(feature for feature in most_correlated_columns if feature not in discarded_features)

print(f"Number of continuous features: {len(selected_features)}")

print(f"The list of features is as follows:  {selected_features}")
missingno.matrix(df_train, figsize= (30, 5))
def missing_data(df):

    missing_data_total = df.isnull().sum().sort_values(ascending = False)

    missing_data_pct = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending = False)

    missing_data = pd.concat([missing_data_total, round(missing_data_pct, 2)], axis = 1, keys = ['Total', 'Percentage'])

    return missing_data
# Function by ntg on stackoverflow post: Jupyter notebook display two pandas tables side by side. Thanks for this.

from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
display_side_by_side(missing_data(df_train).head(10), missing_data(df_test).head(10))
display_side_by_side(missing_data(df_train)[missing_data(df_train).index.isin(selected_features)].head(10),

                     missing_data(df_test)[missing_data(df_test).index.isin(selected_features)].head(10))
for df in list_dfs:

    df['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean(), inplace = True)

    df['GarageCars'].fillna(df_test['GarageCars'].mode()[0], inplace = True)



display_side_by_side(missing_data(df_train)[missing_data(df_train).index.isin(selected_features)].head(10),

                     missing_data(df_test)[missing_data(df_test).index.isin(selected_features)].head(10))
columns_to_delete = list(missing_data(df_all)[missing_data(df_all)['Total'] >= 1].index)



for df in list_dfs:

    df = df.drop(columns_to_delete, axis = 1,inplace = True)



print(f"df_train shape: {df_train.shape}")

print(f"df_test shape: {df_test.shape}")

print(f"df_all shape: {df_all.shape}")
print(f"All continuous features are in df_train? {all(features in list(df_train.columns) for features in selected_features)}")

print(f"All continuous features are in df_test? {all(features in list(df_test.columns) for features in selected_features)}")
fig, ax = plt.subplots(len(selected_features), 1, figsize = (15, 60))

for i, feature in enumerate(selected_features):

    sns.regplot(df_train[feature], df_train['SalePrice'], ax = ax[i])
# Let's do some cleaning to the data

print(df_train.shape)

df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index, axis = 0, inplace = True)

df_train.drop(df_train[(df_train['TotalBsmtSF'] > 5000) & (df_train['SalePrice'] < 300000)].index, axis = 0, inplace = True)

print(df_train.shape)
#histogram and normal probability plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df_train['SalePrice'], fit = stats.norm, ax = ax[0])

res = stats.probplot(df_train['SalePrice'], plot = plt)
# Logarithmic transformation

for df in list_dfs:

    try:

        df['SalePrice'] = np.log(df['SalePrice'])

    except:

        pass



# Histogram and normal probability plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df_train['SalePrice'], fit = stats.norm, ax = ax[0])

res = stats.probplot(df_train['SalePrice'], plot = plt)
#histogram and normal probability plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df_train['GrLivArea'], fit = stats.norm, ax = ax[0])

res = stats.probplot(df_train['GrLivArea'], plot = plt)
# Logarithmic transformation

for df in list_dfs:

    df['GrLivArea'] = np.log(df['GrLivArea'])



# Histogram and normal probability plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df_train['GrLivArea'], fit = stats.norm, ax = ax[0])

res = stats.probplot(df_train['GrLivArea'], plot = plt)
# Histogram and normal probability plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df_train['TotalBsmtSF'], fit = stats.norm, ax = ax[0])

res = stats.probplot(df_train['TotalBsmtSF'], plot = plt)
# Log transform for non zeros

for df in list_dfs:

    df['TotalBsmtSF'].loc[df[df['TotalBsmtSF'] != 0].index] = np.log(df['TotalBsmtSF'][df['TotalBsmtSF'] != 0])



# Histogram and normal probability plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit = stats.norm, ax = ax[0])

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot = plt)
#scatter plot

fig, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.regplot(df_train['GrLivArea'], df_train['SalePrice'], ax = ax[0])

sns.regplot(df_train['TotalBsmtSF'][df_train['TotalBsmtSF'] > 0], df_train['SalePrice'][df_train['TotalBsmtSF']>0], ax = ax[1])
selected_features_train = selected_features + ['SalePrice']

df_train = df_train[selected_features_train]

df_test = df_test[selected_features]

df_test['GarageCars'] = df_test['GarageCars'].astype(int)



display_side_by_side(df_train.head(), df_test.head())
M = df_train.drop(["SalePrice"], axis=1).copy()

n = df_train["SalePrice"].copy()

M_train, M_test, n_train, n_test = train_test_split(M, n, test_size = 0.2, random_state = 0)
lin_regr = LinearRegression()

lin_regr.fit(M_train, n_train)

n_pred = lin_regr.predict(M_test)
coeff_df = pd.DataFrame(lin_regr.coef_, M.columns, columns=['Coefficient'])  

coeff_df
difference_df = pd.DataFrame({'Actual': np.rint(np.exp(n_test)),

                              'Predicted': np.rint(np.exp(n_pred)),

                              'Difference': np.round((np.exp(n_test) - np.exp(n_pred)) / np.exp(n_test) * 100, 2)})

difference_df.head(25)
print(f"Accuracy of Linear Regression Model on testing: {round(lin_regr.score(M_train, n_train) * 100, 2)}")

print('Mean Absolute Error:', metrics.mean_absolute_error(n_test, n_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(n_test, n_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(n_test, n_pred)))
X_train = df_train.drop(["SalePrice"], axis=1).copy()

y_train = df_train["SalePrice"].copy()

X_test = df_test.copy()
lin_regr = LinearRegression()

lin_regr.fit(X_train, y_train)

y_pred_log = lin_regr.predict(X_test)



print(f"Accuracy of Linear Regression Model for submission: {round(lin_regr.score(X_train, y_train) * 100, 2)}")
y_pred = np.exp(y_pred_log)

submission = pd.DataFrame()

submission["Id"] = sale_price_id

submission['SalePrice'] = y_pred
#Let's submit the predictions

submission.to_csv("submission.csv",index = False)



#Check the submission to make sure all is good

submission_check = pd.read_csv("submission.csv")

submission_check.head()