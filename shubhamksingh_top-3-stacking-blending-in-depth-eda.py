import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

from matplotlib import rcParams

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



#ignoring all the warnings because we don't need them

import warnings

warnings.filterwarnings('ignore')



sb.set()

sb.set_style("white")

%matplotlib inline

rcParams['figure.figsize'] = [9,6]



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# let us bring our guns

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



# perform our final prediction on

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# Save the Id

test_id = test['Id']
train.head(6)
# Lets check that test dataset has all the columns in train dataset except SalePrice

diff_train_test = set(train.columns) - set(test.columns)

diff_train_test
# looking at the decoration of our data

train.columns
print((train.shape), (test.shape), '\n')

print("Shape of train.csv: ", train.shape)

print("Shape of test.csv: ", test.shape)
train.info()
# missing_values_bool = train.isnull().any()

# display(missing_values_bool[missing_values_bool == True])



missing_values = train.isnull().sum()

display(missing_values[missing_values>0].sort_values(ascending=False).to_frame(name='Missing Values'))
percent = ((train.isnull().sum()/train.isnull().count()) * 100).sort_values(ascending=True)

percent = percent[percent>0]



plt.xticks(rotation=90); plt.title('Percent Missing Values')

sb.barplot(x=percent.index, y=percent, palette="viridis")
train['SalePrice'].describe()
plt.subplots(figsize=(10, 5))

plt.figure(1)

ax = sb.distplot(train['SalePrice'], bins=30, fit=norm, color="mediumslateblue")

ax.set(xlabel='SalePrice', ylabel='Frequency')

plt.xticks(rotation=-45)



plt.subplots(figsize=(10,5))

plt.figure(2)

stats.probplot(train['SalePrice'], plot=plt)
sell = train['SalePrice']



print(f"Skewness: {sell.skew()}")

print(f"Kurtosis: {sell.kurt()}")
plt.figure(1); plt.title('Boxen plot of the Sales Prices')

sb.boxenplot(train['SalePrice'],color="mediumslateblue")



plt.figure(2); plt.title('Violin plot of the Sales Prices')

sb.violinplot(train['SalePrice'], color="mediumslateblue")



plt.figure(3); plt.title('Strip plot of the Sales Prices')

sb.stripplot(train['SalePrice'], alpha=0.6, color="mediumslateblue")
#swarmplot of overall quality

plt.subplots(figsize=(10, 6))

plt.title('Overall Quality of homes')

sb.swarmplot(train['OverallQual'], train['Id'], palette="viridis")
train['OverallQual'].value_counts()
# sb.pairplot(train[features_ver1])



data_seg1 = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)



plt.subplots(figsize=(9,5))

plt.figure(1); plt.title("SalePrice vs Overall Quality")

sb.boxplot(x='OverallQual', y='SalePrice', data=data_seg1, color="mediumslateblue")



plt.subplots(figsize=(9,5))

plt.figure(2); plt.title("SalePrice vs Overall Quality")

sb.lineplot(x='OverallQual', y='SalePrice', data=data_seg1, color="mediumslateblue")
data_seg2 = pd.concat([train['SalePrice'], train['OverallCond']], axis=1)



plt.subplots(figsize=(9,5))

plt.figure(1); plt.title("SalePrice vs Overall Condition")

sb.boxplot(x='OverallCond', y='SalePrice', data=data_seg2, color="mediumslateblue")



plt.subplots(figsize=(9,5))

plt.figure(2); plt.title("SalePrice vs Overall Condition")

sb.lineplot(x='OverallCond', y='SalePrice', data=data_seg2, color="mediumslateblue")
data_seg3 = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)



plt.subplots(figsize=(10, 6))

plt.figure(1); plt.title('SalePrice vs Above Ground Living Area')

sb.scatterplot(x='GrLivArea', y='SalePrice', data=data_seg3, alpha=0.8, color="mediumslateblue")
# drop the outliers

# we drop the rows containing value of GrLivArea greater than 4000 and 

# SalePrice less than 200000

# these are huge outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)



plt.subplots(figsize=(10,6))

sb.scatterplot(train['GrLivArea'], train['SalePrice'], alpha=0.8, color="mediumslateblue")
plt.subplots(figsize=(10,6))

sb.scatterplot(train['TotalBsmtSF'], train['SalePrice'], alpha=0.8, color="mediumslateblue")
# we change the graph size for this particular graph using subplots unlike rcParams which changes for the entire notebook

plt.subplots(figsize=(20,8))

plt.xticks(rotation=90)

plt.figure(1); plt.title("SalePrice vs YearBuilt")

sb.stripplot(train['YearBuilt'], train['SalePrice'])



plt.subplots(figsize=(20,8))

plt.xticks(rotation=90)

plt.figure(2); plt.title("SalePrice vs YearBuilt")

sb.boxenplot(train['YearBuilt'], train['SalePrice'])
# this is a really important code section

# look at how we compute the correlation with df.corr() function and then feed it to sb.heatmap()

# play around by tweaking argument values



corrmat = train.corr()

plt.subplots(figsize=(17,17))

plt.title("Correlation Matrix")

# sb.heatmap(corrmat, vmax=0.9, vmin=0.5, square=True, cmap='YlGnBu')

sb.heatmap(corrmat, vmax=0.9, square=True, cmap="Oranges", annot=True, fmt='.1f', linewidth='.1')
# we can convert a series object to a dataframe using to_frame() method on the series

imp_ftr = corrmat['SalePrice'].sort_values(ascending=False).head(11).to_frame()



imp_ftr
# first graph

plt.subplots(figsize=(5,8))

plt.title('SalePrice Correlation Matrix')

sb.heatmap(imp_ftr, vmax=0.9, annot=True, fmt='.2f', cmap="Oranges", linewidth='.1')
degree_correlation = corrmat['SalePrice'].sort_values(ascending=False)

degree_correlation.to_frame()
plt.subplots(figsize=(10,12))

plt.title('Correlation with SalePrice')

degree_correlation.plot(kind='barh', color="mediumslateblue")
plt.subplots(figsize=(15, 15))

sb.heatmap(corrmat>0.8, annot=True, square=True, cmap="Oranges", linewidth='.1')
# Dropping unwanted features

train.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True) 

test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
degree_correlation = corrmat['SalePrice'].sort_values(ascending=False)

degree_correlation.to_frame()
numerical_data = train.select_dtypes(exclude=['object']).drop(['SalePrice', 'Id'],axis=1).copy()

categorical_data = train.select_dtypes(include=['object']).columns
fig = plt.figure(figsize=(17,22))

for i in range(len(numerical_data.columns)):

    fig.add_subplot(9,4,i+1)

    sb.distplot(numerical_data.iloc[:,i].dropna(), hist=False, kde_kws={'bw':0.1}, color='mediumslateblue')

    plt.xlabel(numerical_data.columns[i])

plt.tight_layout()

plt.show()
cmap = sb.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

fig1 = plt.figure(figsize=(17,22))

for i in range(len(numerical_data.columns)):

    fig1.add_subplot(9, 4, i+1)

    sb.scatterplot(numerical_data.iloc[:, i],train['SalePrice'], palette='spring', marker='+', hue=train['SalePrice'], legend=False)

plt.tight_layout()

plt.show()
columns = (len(categorical_data)/5)+1



fg, ax = plt.subplots(figsize=(18, 30))



for i, col in enumerate(categorical_data):

    fg.add_subplot(columns, 5, i+1)

    sb.countplot(train[col], palette='spring')

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])



plt.subplots(figsize=(11, 5))

plt.figure(1)

ax = sb.distplot(train['SalePrice'], bins=100, fit=norm, color="mediumslateblue")

ax.set(xlabel='SalePrice', ylabel='Frequency')

plt.xticks(rotation=-45)



plt.subplots(figsize=(10,5))

plt.figure(2)

stats.probplot(train['SalePrice'], plot=plt)
#MSSubClass=The building class

train['MSSubClass'] = train['MSSubClass'].apply(str)

test['MSSubClass'] = test['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

train['OverallCond'] = train['OverallCond'].astype(str)

test['OverallCond'] = test['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

train['YrSold'] = train['YrSold'].astype(str)

test['YrSold'] = test['YrSold'].astype(str)



train['MoSold'] = train['MoSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)
# Find columns with missing values

cols_with_missing = [col for col in train.columns if train[col].isnull().any()]

cols_with_missing
# Finding total number of missing values in each column

total_missing = train.isnull().sum().sort_values(ascending=False)

total_missing = total_missing[total_missing>0]



percent = ((train.isnull().sum()/train.isnull().count()) * 100).sort_values(ascending=False)

percent = percent[percent>0]



data_missing = pd.concat([total_missing, percent], axis=1, keys=['Total', 'Percent'])

data_missing
object_cols = [col for col in train.columns if train[col].dtype == "object"]

object_cols
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
for df in [train, test]:

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence', 'PoolQC', 'MiscFeature', 'Alley'):

        df[col] = df[col].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)

test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
# Check for missing values



missing = train.isnull().sum().sort_values(ascending=False)

missing = missing[missing>0]

missing
train['Electrical'].describe()
train['Electrical'].mode()[0]
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# Check for missing values



missing = train.isnull().sum().sort_values(ascending=False)

missing = missing[missing>0]

missing
# Categorical missing values in test data

missing = test.select_dtypes(include='object').isnull().sum().sort_values(ascending=False)

missing = missing[missing>0]

missing
# Numerical missing values in test data

missing = test.select_dtypes(exclude='object').isnull().sum().sort_values(ascending=False)

missing[missing>0]
# Handling MSZoning

train['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

test['MSZoning'] = test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
display(test['Utilities'].describe())

display(train['Utilities'].describe())
test = test.drop(['Utilities'], axis=1)

train = train.drop(['Utilities'], axis=1)
for df in [train, test]:

    for col in ('Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'KitchenQual'):

        df[col] = df[col].fillna(df[col].mode()[0])
# Categorical missing values in test data

missing = test.select_dtypes(include='object').isnull().sum().sort_values(ascending=False)

missing = missing[missing>0]

missing
for df in [train, test]:

    for col in ('MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'TotalBsmtSF',

               'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1'):

        df[col] = df[col].fillna(0)
# Numerical missing values in test data

missing = test.select_dtypes(exclude='object').isnull().sum().sort_values(ascending=False)

missing[missing>0]
train.shape, test.shape
train.head(10)
sf = [col for col in train.columns if 'SF' in col]

sf
train['TotalSF'] = train['TotalBsmtSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['2ndFlrSF']
train[['TotalBsmtSF', '2ndFlrSF', 'TotalSF']].head()
plt.subplots(figsize=(10, 7))

sb.scatterplot(x=train['TotalSF'], y=train['SalePrice'], color="mediumslateblue")
bath = [col for col in train.columns if 'Bath' in col]

bath
train['TotalBath'] = train['BsmtFullBath'] + train['BsmtHalfBath'] + train['FullBath'] + train['HalfBath']

test['TotalBath'] = test['BsmtFullBath'] + test['BsmtHalfBath'] + test['FullBath'] + test['HalfBath']
train[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotalBath']].head()
plt.subplots(figsize=(10, 7))

sb.boxenplot(x=train['TotalBath'], y=train['SalePrice'], color="mediumslateblue")
porch = [col for col in train.columns if 'Porch' in col]

porch
train['PorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch'] + train['WoodDeckSF']

test['PorchSF'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch'] + test['WoodDeckSF']
train[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF', 'PorchSF']].head()
plt.subplots(figsize=(10, 7))

sb.scatterplot(x=train['PorchSF'], y=train['SalePrice'], color="mediumslateblue")
pool = [col for col in train.columns if 'Pool' in col]

pool
# train data

train['HasBsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x>0 else 0)

train['Has2ndFlr'] = train['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)

train['HasFirePlace'] = train['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

train['HasGarage'] = train['GarageCars'].apply(lambda x: 1 if x>0 else 0)



# test data

test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

test['HasPool'] = test['PoolArea'].apply(lambda x: 1 if x>0 else 0)

test['Has2ndFlr'] = test['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)

test['HasFirePlace'] = test['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

test['HasGarage'] = test['GarageCars'].apply(lambda x: 1 if x>0 else 0)
train[['HasBsmt', 'HasPool', 'Has2ndFlr', 'HasFirePlace', 'HasGarage']].head()
plt.figure(1)

sb.boxenplot(train['HasGarage'], train['SalePrice'], color="mediumslateblue")



plt.figure(2)

sb.boxenplot(train['HasBsmt'], train['SalePrice'], color="mediumslateblue")



plt.figure(3)

sb.boxenplot(train['HasPool'], train['SalePrice'], color="mediumslateblue")



plt.figure(4)

sb.boxenplot(train['Has2ndFlr'], train['SalePrice'], color="mediumslateblue")



plt.figure(5)

sb.boxenplot(train['HasFirePlace'], train['SalePrice'], color="mediumslateblue")
print(train.shape, test.shape)
train['MasVnrArea'].head()
train['MasVnrArea'] = train['MasVnrArea'].astype(int)

test['MasVnrArea'] = test['MasVnrArea'].astype(int)
train['MasVnrArea'].head()
plt.subplots(figsize=(10, 7))

sb.scatterplot(train['MasVnrArea'], train['SalePrice'], color="mediumslateblue")
# Import library

from sklearn.preprocessing import LabelEncoder



label_enc_variables = ['FireplaceQu', 'LotShape', 'OverallCond', 'ExterQual',

                       'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                       'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'GarageFinish', 'GarageQual',

                       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']





# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in label_enc_variables:

    train[col] = label_encoder.fit_transform(train[col])

    test[col] = label_encoder.transform(test[col])
print(train.shape, test.shape)
# Perform train test split on this data



X = train.drop(['SalePrice'], axis=1)

y = train['SalePrice']
# Filling numerical columns

num_cols = [col for col in X.columns if X[col].dtype!='object']

X.update(X[num_cols].fillna(0))

test.update(test[num_cols].fillna(0))



# Filling categorical columns

cat_cols = [col for col in X.columns if X[col].dtype=='object']

X.update(X[cat_cols].fillna('None'))

test.update(test[cat_cols].fillna('None'))
print(X.shape, test.shape)
# Importing our libraries



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



# Added in version 7 of this notebook

from sklearn.model_selection import GridSearchCV

from mlxtend.regressor import StackingCVRegressor

from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVR
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=2)
numerical_cols = [col for col in X_train.columns

                  if X_train[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]



categorical_cols= [col for col in X_train.columns

                   if X_train[col].nunique()<=30 and X_train[col].dtype=='object']



final_cols = numerical_cols + categorical_cols

X_train = X_train[final_cols].copy()

X_valid = X_valid[final_cols].copy()



test = test[final_cols].copy()
X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

test = pd.get_dummies(test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, test = X_train.align(test, join='left', axis=1)
print("X_train Shape", X_train.shape)

print("X_valid Shape", X_valid.shape)

print("test shape", test.shape)
# Reversing log transform on y ('SalePrice')

def rev_y(trans_y):

    return np.expm1(trans_y)
estimators = [2000, 2500, 3000, 3500]

for n in estimators:

    my_dict = dict()

    model = XGBRegressor(n_estimators=n, learning_rate=0.01, colsample_bytree=0.45, max_depth=3)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    mae = mean_absolute_error(rev_y(preds), rev_y(y_valid))

    my_dict[f"n_est {n}"] = mae

    print(my_dict)
my_dict = dict()

n=3500

model = XGBRegressor(n_estimators=n, learning_rate=0.01, colsample_bytree=0.45, max_depth=3,

                     gamma=0, subsample=0.4, reg_alpha=0, reg_lambda=1, objective='reg:squarederror')

    

model.fit(X_train, y_train)

preds = model.predict(X_valid)

mae = mean_absolute_error(rev_y(preds), rev_y(y_valid))

my_dict[f"n_est {n}"] = mae

print(my_dict)
# Version 7

n = 3500

my_dict = dict()



# XGBoost Regressor

xgb = XGBRegressor(n_estimators=n,

                   learning_rate=0.01,

                   colsample_bytree=0.45,

                   max_depth=3,

                   gamma=0,

                   subsample=0.4,

                   reg_alpha=0,

                   reg_lambda=1,

                   objective='reg:squarederror')



# Light Gradient Boosting Regressor

lgb = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=n,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# Ridge Regressor

ridge_alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas))



# ElasticNet Regressor

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, l1_ratio=e_l1ratio))



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=n,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)



# Stacking models

stack = StackingCVRegressor(regressors=(ridge, gbr, xgb, elasticnet, svr, lgb),

                                meta_regressor=xgb,

                                use_features_in_secondary=True)
Scores = {}
# Function for checking Cross-val scores                              

def rmse(model, X, y):

    scores = np.sqrt(-1 * cross_val_score(model, X, y,

                        cv=10, 

                        scoring='neg_mean_squared_error'))

    return scores



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
# Cross-val scores                              

scores = rmse(xgb, X_train, y_train)

print("XGBoost Regressor\n")

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))

Scores['XGB'] = scores.mean()
scores = rmse(lgb, X_train, y_train)

print("Light Gradient Boosting Regressor\n")

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))

Scores['LGB'] = scores.mean()
scores = rmse(ridge, X_train, y_train)

print("Ridge Regressor\n")

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))

Scores['Ridge'] = scores.mean()
scores = rmse(svr, X_train, y_train)

print("Support Vector Regressor\n")

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))

Scores['SVR'] = scores.mean()
scores = rmse(gbr, X_train, y_train)

print("Gradient Boosting Regressor\n")

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))

Scores['GBR'] = scores.mean()
scores = rmse(elasticnet, X_train, y_train)

print("ElasticNet Regressor\n")

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))

Scores['ElasticNet'] = scores.mean()
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)
ridge.fit(X_train, y_train)
svr.fit(X_train, y_train)
gbr.fit(X_train, y_train)
elasticnet.fit(X_train, y_train)
stack.fit(np.array(X_train), np.array(y_train))
def blended_predictions(X):

    return ((0.2 * ridge.predict(X)) + \

            (0.2 * elasticnet.predict(X)) + \

            (0.05 * svr.predict(X)) + \

            (0.1 * gbr.predict(X)) + \

            (0.1 * xgb.predict(X)) + \

            (0.1 * lgb.predict(X)) + \

           (0.25 * stack.predict(np.array(X))))
X_valid = X_valid.fillna(0)

test = test.fillna(0)
blended_score = rmsle(y_valid, blended_predictions(X_valid))



Scores['Blended'] = blended_score

blended_score
plt.subplots(figsize=(15, 7));plt.title("Scores of different models") 

plt.xticks(rotation=45)

sb.pointplot(x=list(Scores.keys()), y=[score for score in Scores.values()], markers=['o'], linestyles=['-'], color="mediumslateblue")
# Read in sample_submission dataframe

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.shape
# Append predictions from blended models

submission.iloc[:,1] = np.floor(rev_y(blended_predictions(test)))
# Brutal approach to deal with outliers

q1 = submission['SalePrice'].quantile(0.0045)

q2 = submission['SalePrice'].quantile(0.99)



submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission['SalePrice'] *= 1.001619

submission.to_csv("new_submission.csv", index=False)