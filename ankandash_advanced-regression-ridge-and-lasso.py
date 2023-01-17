import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/housetrain.csv')
data.head()
df = data.drop('Id',axis=1)
df.shape
df.info()
df.columns
# Categotical columns

df.select_dtypes(include='object').columns
# Numeric columns

df.select_dtypes(exclude='object').columns
df.describe()
# Sales column

df['SalePrice'].describe()
plt.figure()

sns.distplot(df['SalePrice'],color='r')



plt.tight_layout()
# Since there a lot of numerical columns it is not easy to understand which variables are important so

# we will use the correlation matrix to solve this issue

plt.figure(figsize=(12,10),dpi=200)

sns.heatmap(df.corr())



plt.tight_layout()
# sales price correlation matrix



n = 10 # number of variables which have the highest correlation with 'Sales price'



corrmat = df.corr()



cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index

plt.figure(dpi=100)

sns.heatmap(df[cols].corr(),annot=True)
# Plotting the scatter plots for the above variables to observe the kind of relationship with the target variable

columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF','FullBath', 'YearBuilt']

sns.pairplot(df[cols])
sns.boxplot(x="OverallQual", y='SalePrice', data = df)
null = pd.DataFrame(round(df.isnull().sum()/len(df.index)*100,2).sort_values(ascending=False),columns=["Null %"])

null.index.name = 'Features'

null.head()
# dataframe with features having null values

null_df = null[null["Null %"] > 0]

null_df
# lets observe the columsn with highest percentage of missing values

print('The unique values in columsn with highest number if nan or missing values')

print('\n')

print('PoolQC: ',df['PoolQC'].unique())

print('\n')

print('MiscFeature: ',df['MiscFeature'].unique())

print('\n')

print('Alley: ',df['Alley'].unique())

print('\n')

print('Fence: ',df['Fence'].unique())

print('\n')

print('FireplaceQu: ',df['FireplaceQu'].unique())

print('\n')

print('LotFrontage: ',df['LotFrontage'].unique())

print('\n')

print('GarageCond: ',df['GarageCond'].unique())

print('\n')

print('GarageType: ',df['GarageType'].unique())

print('\n')

print('GarageYrBlt: ',df['GarageYrBlt'].unique())

print('\n')

print('GarageFinish: ',df['GarageFinish'].unique())

print('\n')

print('GarageQual: ',df['GarageQual'].unique())

print('\n')

print('BsmtExposure: ',df['BsmtExposure'].unique())

print('\n')

print('BsmtFinType2: ',df['BsmtFinType2'].unique())

print('\n')

print('BsmtFinType1: ',df['BsmtFinType1'].unique())

print('\n')

print('BsmtCond: ',df['BsmtCond'].unique())

print('\n')

print('BsmtQual: ',df['BsmtQual'].unique())

print('\n')

print('MasVnrArea: ',df['MasVnrArea'].unique())

print('\n')

print('MasVnrType: ',df['MasVnrType'].unique())

print('\n')

print('Electrical: ',df['Electrical'].unique())
# we can impute the nan or missing values for each of the columns with missing values by analyzing their description



# for the columns below we will impute missing values with 'none' as these are categorical in nature

df["PoolQC"] = df["PoolQC"].fillna("None")

df["MiscFeature"] = df["MiscFeature"].fillna("None")

df["Alley"] = df["Alley"].fillna("None")

df["Fence"] = df["Fence"].fillna("None")

df["FireplaceQu"] = df["FireplaceQu"].fillna("None")



# for the LotFrontage columns we will impute the missing values with the median since the feature contains outliers

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())



# for the "garage" columns we will impute the null values with 'none'

for col in ('GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish','GarageQual'):

    df[col] = df[col].fillna('None')

    

# for the "Bsmt" columns we will impute the null values with 'none'

for col in ('BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond','BsmtQual'):

    df[col] = df[col].fillna('None')

    

# MasVnrArea impute with 0

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)



# MasVnrType impute with 'None'

df['MasVnrType'] = df['MasVnrType'].fillna('None')



# Electrical column inpute the missing value with Mode

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
# dropping any rows with null or missing values

df = df.dropna(axis=0)
# checking for the presence of any more null values

df.isnull().values.any()
print("Shape of dataframe is: ", df.shape)
# some of the numerical column are categorical in nature so will transform them



df['MSSubClass'] = df['MSSubClass'].apply(str)

df['OverallCond'] = df['OverallCond'].astype(str)

df['OverallQual'] = df['OverallQual'].astype(str)

df['YearBuilt'] = df['YearBuilt'].astype(str)

df['YearRemodAdd'] = df['YearRemodAdd'].astype(str)

df['BsmtFullBath'] = df['BsmtFullBath'].astype(str)

df['BsmtHalfBath'] = df['BsmtHalfBath'].astype(str)

df['FullBath'] = df['FullBath'].astype(str)

df['HalfBath'] = df['HalfBath'].astype(str)

df['KitchenAbvGr'] = df['KitchenAbvGr'].astype(str)

df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].astype(str)

df['Fireplaces'] = df['Fireplaces'].astype(str)

df['GarageCars'] = df['GarageCars'].astype(str)

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)
# other columns to be removed
df['PoolArea'].value_counts() 
df['MiscVal'].value_counts()
# we will drop the 'Pool Area' and 'MiscVal' column as it dominated by one value and it won't add any extra information to our model

df = df.drop(['PoolArea','MiscVal'],axis=1)
df.describe()
df['GrLivArea'].describe() # outliers to be removed
sns.scatterplot(x = df['GrLivArea'], y = df['SalePrice'])
# removing outliers which have a value greater than 4000

df = df[df['GrLivArea']<4000]
df.shape
df.head()
# Categotical columns

df.select_dtypes(include='object').columns
# numeric columns

df.select_dtypes(exclude='object').columns
df = pd.get_dummies(df,drop_first=True)

df.head()
df.shape
y = df.pop('SalePrice')

y.head()
X = df

X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.head()
print('X_train shape',X_train.shape)

print('X_test shape',X_test.shape)

print('y_train shape',y_train.shape)

print('y_test shape',y_test.shape)
X_train.head()
y_train.head()
X_test.head()
y_test.head()
# columns to be scaled

X_train.select_dtypes(include=['int64','int32','float64','float32']).columns
num_vars = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BedroomAbvGr', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch']



X_train[num_vars].head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_train.head()
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}





ridge = Ridge()



# cross validation

folds = 5

ridge_model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

ridge_model_cv.fit(X_train, y_train) 
print(ridge_model_cv.best_params_)

print(ridge_model_cv.best_score_)
alpha = 10

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
lasso = Lasso()



# cross validation

lasso_model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



lasso_model_cv.fit(X_train, y_train)
print(lasso_model_cv.best_params_)

print(lasso_model_cv.best_score_)
alpha =100



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
lasso.coef_
# ridge train pred

ridge_train_pred = ridge.predict(X_train)
# lasso train pred

lasso_train_pred = lasso.predict(X_train)
X_test[num_vars] = scaler.transform(X_test[num_vars])
# ridge predictions

ridge_pred = ridge.predict(X_test)
# lasso predictions

lasso_pred = lasso.predict(X_test)
from sklearn.metrics import r2_score 
# model evaluation on the training set 

r2_score(y_train,ridge_train_pred)
# model evaluation on the training set 

r2_score(y_train,lasso_train_pred)
# ridge model evaluation for the test set

r2_score(y_test, ridge_pred)
# lasso model evaluation for the test set

r2_score(y_test, lasso_pred)
# Plotting y_test and y_pred to understand the spread for ridge regression.

fig = plt.figure(dpi=200)

plt.scatter(y_test,ridge_pred)

fig.suptitle('y_test vs ridge_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('ridge_pred', fontsize=16)  
# Plotting y_test and y_pred to understand the spread for lasso regression.

fig = plt.figure(dpi=200)

plt.scatter(y_test,lasso_pred)

fig.suptitle('y_test vs lasso_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('lasso_pred', fontsize=16)  
lasso_non_zero_coef = lasso.coef_[lasso.coef_ != 0]
selected_features = X_train.columns[lasso.coef_ != 0]
selected_features
lasso_feat_df = pd.DataFrame(lasso_non_zero_coef, index=selected_features, columns=['Value'])

lasso_feat_df = lasso_feat_df.sort_values(['Value'],ascending = False)
lasso_feat_df.head(20)