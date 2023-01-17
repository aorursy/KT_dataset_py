import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



from math import sqrt

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error



%matplotlib inline
# Import dataset

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



train_data.shape, test_data.shape
# Preview train dataset

train_data.head()
# Preview test dataset

test_data.head()
# Remove IDs from train and test set, not useful for model

train_ID = train_data['Id']

test_ID = test_data['Id']

train_data.drop(['Id'], axis=1, inplace=True)

test_data.drop(['Id'], axis=1, inplace=True)

train_data.shape, test_data.shape
# Analyze SalePrice

train_data['SalePrice'].describe()
# Distribution plot

f, ax = plt.subplots(figsize=(10,5))

sns.distplot(train_data['SalePrice'])

ax.set(xlabel="SalePrice")

ax.set(ylabel="Frequency")

ax.set(title="SalePrice Distribution")

plt.show()
# Skewness and Kurtosis

# Skewness - measure of the lack of symmetry in the data

# Kurtosis - shows whether there is many outliers in the data

print("Skewness: %f" % train_data['SalePrice'].skew())

print("Kurtosis: %f" % train_data['SalePrice'].kurt())
# Find the numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_features = []

for i in train_data.columns:

    if train_data[i].dtype in numeric_dtypes:

        numeric_features.append(i)
# Visualizing the outliers in numeric features

plt.subplots(ncols=2, nrows=0, figsize=(12,120))

plt.subplots_adjust(right=2)

plt.subplots_adjust(top=2)

for i, feature in enumerate(list(train_data[numeric_features]), 1):

    plt.subplot(len(list(numeric_features)), 3, i)

    sns.scatterplot(x=feature, y='SalePrice', data=train_data)

    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

    plt.ylabel('SalePrice', size=15, labelpad=12.5)

    

plt.show()
# Look at data correlation using heatmap

corr = train_data.corr()

plt.subplots(figsize=(15,15))

sns.heatmap(corr, fmt='.1f', cmap="Blues", square=True)

plt.show()
# Box plot SalePrice/OverallQual

feature = 'OverallQual'

plt.figure(figsize=(10,5))

sns.boxplot(train_data[feature], train_data['SalePrice'])

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000)

plt.show()
# Box plot SalePrice/YearBuilt

feature = 'YearBuilt'

plt.figure(figsize=(20,10))

sns.boxplot(train_data[feature], train_data['SalePrice'])

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);

plt.show()
# Scatter plot SalePrice/GrLivArea

feature = 'GrLivArea'

plt.figure(figsize=(10,5))

plt.scatter(train_data[feature], train_data['SalePrice'], c='b', alpha=0.3)

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000)

plt.show()
# Scatter plot SalePrice/GarageArea

feature = 'GarageArea'

plt.figure(figsize=(10,5))

plt.scatter(train_data[feature], train_data['SalePrice'], c='b', alpha=0.3)

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000)

plt.show()
# Scatter plot SalePrice/LotArea

feature = 'LotArea'

plt.figure(figsize=(10,5))

plt.scatter(train_data[feature], train_data['SalePrice'], c='b', alpha=0.3)

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000)

plt.show()
# Scatter plot SalePrice/TotalBsmtSF

feature = 'TotalBsmtSF'

plt.figure(figsize=(10,5))

plt.scatter(train_data[feature], train_data['SalePrice'], c='b', alpha=0.3)

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000)

plt.show()
# Box plot SalePrice/GarageCars

feature = 'GarageCars'

plt.figure(figsize=(10,5))

sns.boxplot(train_data[feature], train_data['SalePrice'])

plt.xlabel(feature)

plt.ylabel('SalePrice')

plt.axis(ymin=0, ymax=800000);

plt.show()
# Remove outliers

train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index, inplace=True)

train_data.drop(train_data[(train_data['GrLivArea']>4500) & (train_data['SalePrice']<300000)].index, inplace=True)

train_data.reset_index(drop=True, inplace=True)

train_data.shape
# Some non-numeric features stored as numbers, convert to strings

train_data['MSSubClass'] = train_data['MSSubClass'].astype(str)

#train_data['YrSold'] = train_data['YrSold'].astype(str)

train_data['MoSold'] = train_data['MoSold'].astype(str)

#train_data['GarageYrBlt'] = train_data['GarageYrBlt'].astype(str)



# How about OverallQual and OverallCond?

# YrSold and GarageYrBlt should be numeric, not categorial
# Split into train and validation datasets

# Random train and validation dataset

train_data, validation_data = train_test_split(train_data, test_size=0.25, random_state=42, shuffle=True)
train_data.describe()
validation_data.describe()
diffmeanpercent = (train_data.mean() - validation_data.mean())/ train_data.mean()

diffmeanpercent
# Split features

Y_train = train_data['SalePrice'].reset_index(drop=True)

X_train = train_data.drop(['SalePrice'], axis=1)

Y_validation = validation_data['SalePrice'].reset_index(drop=True)

X_validation = validation_data.drop(['SalePrice'], axis=1)

f, ax = plt.subplots(figsize=(10,5))

sns.distplot(Y_train, fit=norm)

ax.set(xlabel="SalePrice")

ax.set(ylabel="Frequency")

ax.set(title="SalePrice Distribution (training set)")

plt.show()
f, ax = plt.subplots(figsize=(10,5))

sns.distplot(Y_validation, fit=norm)

ax.set(xlabel="SalePrice")

ax.set(ylabel="Frequency")

ax.set(title="SalePrice Distribution (validation set)")

plt.show()
Y_train = np.log1p(Y_train)

Y_validation = np.log1p(Y_validation)
f, ax = plt.subplots(figsize=(10,5))

sns.distplot(Y_train, fit=norm)

ax.set(xlabel="SalePrice")

ax.set(ylabel="Frequency")

ax.set(title="SalePrice Distribution (training set)")

plt.show()
f, ax = plt.subplots(figsize=(10,5))

sns.distplot(Y_validation, fit=norm)

ax.set(xlabel="SalePrice")

ax.set(ylabel="Frequency")

ax.set(title="SalePrice Distribution (validation set)")

plt.show()
Y_train.shape, X_train.shape, Y_validation.shape, X_validation.shape
X_test = test_data
# Function to calculate the percentage of missing data of each feature

def calc_percentage_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    output = pd.DataFrame(sorted(dict_x.items(), key=lambda x: x[1], reverse=True), columns=['Feature', 'Percentage missing data'])

    return output
# Training set

X_train_missing = calc_percentage_missing(X_train)

X_train_missing.head(10)
# Validation set

X_validation_missing = calc_percentage_missing(X_validation)

X_validation_missing.head(10)
# Test set

X_test_missing = calc_percentage_missing(X_test)

X_test_missing.head(10)
# Create function to visualize the amount of missing values

def viz_missing_vals(df):

    df = df[df['Percentage missing data']>0]

    plt.subplots(figsize=(10,5))

    plt.bar(df['Feature'], df['Percentage missing data'])

    plt.xticks(rotation=90);

    plt.xlabel('Features')

    plt.ylabel('Percentage of missing values')

    plt.title('Percentage of missing data by feature')

    plt.show()
# Training data

viz_missing_vals(X_train_missing)
# Validation data

viz_missing_vals(X_validation_missing)
# no of features with missing values in training set

len(X_train_missing[X_train_missing['Percentage missing data'] > 0])
# no of features with missing values in validation set

len(X_validation_missing[X_validation_missing['Percentage missing data'] > 0])
# no of features with missing values in test set

len(X_test_missing[X_test_missing['Percentage missing data'] > 0])
# Function to handle missing values in each feature in training set

def handle_missing_features_train(features):

    # categorial features

    features['PoolQC'] = features['PoolQC'].fillna("None")

    features['MiscFeature'] = features['MiscFeature'].fillna("None")

    features['Alley'] = features['Alley'].fillna("None")

    features['Fence'] = features['Fence'].fillna("None")

    features['FireplaceQu'] = features['FireplaceQu'].fillna("None")

    features['MasVnrType'] = features['MasVnrType'].fillna("None")

    

    # numerical features

    features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mode()[0])

    features['MasVnrArea'] = features['MasVnrArea'].fillna(0)

    features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

    features['Utilities'] = features['Utilities'].fillna(features['Utilities'].mode()[0])

    features['Functional'] = features['Functional'].fillna(features['Functional'].mode()[0])

    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

    features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

    features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



    # garage features

    for col in ('GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):

        features[col]  = features[col].fillna("None")

    for col in ('GarageYrBlt', 'GarageCars', 'GarageArea'):

        features[col] = features[col].fillna(0)

    

    # basement features

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

        features[col]  = features[col].fillna("None")

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath'):

        features[col] = features[col].fillna(0)

        

    return features
# Function to handle missing values in each feature in datasets other than training set

# Uses what was filled in the training set to fill into the other datasets

def handle_missing_features_other(dataset, train_set):

    dataset['PoolQC'] = dataset['PoolQC'].fillna("None")

    dataset['MiscFeature'] = dataset['MiscFeature'].fillna("None")

    dataset['Alley'] = dataset['Alley'].fillna("None")

    dataset['Fence'] = dataset['Fence'].fillna("None")

    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna("None")

    dataset['MasVnrType'] = dataset['MasVnrType'].fillna("None")

    

    # numerical features

    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(train_set['LotFrontage'].mode()[0])

    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)

    dataset['MSZoning'] = dataset['MSZoning'].fillna(train_set['MSZoning'].mode()[0])

    dataset['Utilities'] = dataset['Utilities'].fillna(train_set['Utilities'].mode()[0])

    dataset['Functional'] = dataset['Functional'].fillna(train_set['Functional'].mode()[0])

    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(train_set['Exterior1st'].mode()[0])

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(train_set['Exterior2nd'].mode()[0])

    dataset['Electrical'] = dataset['Electrical'].fillna(train_set['Electrical'].mode()[0])

    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(train_set['KitchenQual'].mode()[0])

    dataset['SaleType'] = dataset['SaleType'].fillna(train_set['SaleType'].mode()[0])



    

    # garage features

    for col in ('GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):

        dataset[col]  = dataset[col].fillna("None")

    for col in ('GarageYrBlt', 'GarageCars', 'GarageArea'):

        dataset[col] = dataset[col].fillna(0)

    

    # basement features

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

        dataset[col]  = dataset[col].fillna("None")

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath'):

        dataset[col] = dataset[col].fillna(0)

        

    return dataset
# Training data

handle_missing_features_train(X_train)
# Validaton data

handle_missing_features_other(X_validation, X_train)
# Test data

handle_missing_features_other(X_test, X_train)
# Check that we did not miss any features with missing values

len(X_train_missing[X_train_missing['Percentage missing data'] > 0]), len(X_validation_missing[X_validation_missing['Percentage missing data'] > 0]), len(X_test_missing[X_test_missing['Percentage missing data'] > 0])
# Find the numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_features = []

for i in X_train.columns:

    if X_train[i].dtype in numeric_dtypes:

        numeric_features.append(i)
# Find skewed numeric features

features_skew = X_train[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skewed_features = features_skew[features_skew>0.5]

skew_index = high_skewed_features.index



high_skewed_features
high_skewed_features_names = list(high_skewed_features.index.values)
high_skewed_features.count()
# Train data

f, ax = plt.subplots(figsize=(10,5))

sns.boxplot(data=X_train[high_skewed_features_names], orient='h')

ax.set_xscale("log")

ax.set(ylabel="Name of Features")

ax.set(xlabel="Numeric value")

ax.set(title="Distribution of Numeric Features (train_data)")

plt.show()
# Validation data

f, ax = plt.subplots(figsize=(10,5))

sns.boxplot(data=X_validation[high_skewed_features_names], orient='h')

ax.set_xscale("log")

ax.set(ylabel="Name of Features")

ax.set(xlabel="Numeric value")

ax.set(title="Distribution of Numeric Features (validation_data)")

plt.show()
# Test data

f, ax = plt.subplots(figsize=(10,5))

sns.boxplot(data=X_test[high_skewed_features_names], orient='h')

ax.set_xscale("log")

ax.set(ylabel="Name of Features")

ax.set(xlabel="Numeric value")

ax.set(title="Distribution of Numeric Features (test_data)")

plt.show()
plt.subplots(figsize=(12,120))

for i, feature in enumerate(list(X_train[high_skewed_features_names]), 1):

    plt.subplot(len(list(high_skewed_features_names)), 3, i)

    try:

        sns.distplot(X_train[feature], bins=50, fit=norm, color='r')

    except:

        sns.distplot(X_train[feature], bins=50, fit=norm, color='r', kde_kws={'bw':0.1}) # fix for the problems described above

plt.show() 
# Transform skewed features in training set

for i in skew_index:

    #X_train[i] = boxcox1p(X_train[i], boxcox_normmax(X_train[i]+1))

    X_train[i] = np.log1p(X_train[i])

    #X_train[i] = np.sqrt(X_train[i])
# Calculate skewness after transform

skewed_features_after = X_train[high_skewed_features_names].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_features_after
# Show difference in skewedness after transform

skew_diff = high_skewed_features - skewed_features_after

skew_diff
# Validation set

for i in skew_index:

    #X_validation[i] = boxcox1p(X_validation[i], boxcox_normmax(X_validation[i]+1))

    X_validation[i] = np.log1p(X_validation[i])

    #X_validation[i] = np.sqrt(X_validation[i])
# Test set

for i in skew_index:

    #X_test[i] = boxcox1p(X_test[i], boxcox_normmax(X_test[i]+1))

    X_test[i] = np.log1p(X_test[i])

    #X_test[i] = np.sqrt(X_test[i])
# Combine the datasets

X_all = [X_train, X_validation, X_test]

X_all = pd.concat(X_all)
# Create custom features

#X_all['TotalBath'] = X_all.apply(lambda row: (row.FullBath + (0.5 * row.HalfBath) + row.BsmtFullBath + (0.5 * row.BsmtHalfBath)), axis=1)

#X_all['HouseAge'] = X_all.apply(lambda row: 2010 - row.YearBuilt, axis=1)

#X_all['GarageAreaPerCar'] = X_all.apply(lambda row: row.GarageArea / row.GarageCars if row.GarageCars>=1 else 0, axis=1) # gives inf in predictions

#X_all['HomeOverallQuality'] = X_all.apply(lambda row: row.OverallQual + row.OverallCond, axis=1) # gives inf in predictions



# Include powers of features which appear to have non linear correlation

#X_all['OverallQual_sq'] = X_all.apply(lambda row: row.OverallQual **2, axis=1) # gives inf

#X_all['OverallCond_sq'] = X_all.apply(lambda row: row.OverallCond **2, axis=1) # gives inf
X_all = pd.get_dummies(X_all).reset_index(drop=True)

X_all.shape
# Remove any duplicated column names

X_all = X_all.loc[:,~X_all.columns.duplicated()]

X_all.shape
# Split into respective sets

X_train = X_all.iloc[:len(Y_train), :]

X_validation = X_all.iloc[len(Y_train):(len(Y_train)+len(Y_validation)), :]

X_test = X_all.iloc[(len(Y_train)+len(Y_validation)):, :]

X_train.shape, X_validation.shape, X_test.shape
def use_model(model):

    model.fit(X_train, Y_train)

    

    # Calculate train and validation predictions

    Y_train_prediction = model.predict(X_train)

    Y_train_prediction = np.expm1(Y_train_prediction)

    Y_validation_prediction = model.predict(X_validation)

    Y_validation_prediction = np.expm1(Y_validation_prediction)



    msle_train = mean_squared_log_error(Y_train, Y_train_prediction)

    rmsle_train = sqrt(msle_train)

    print("Training set rmsle:", rmsle_train)

    msle_validation = mean_squared_log_error(Y_validation, Y_validation_prediction)

    rmsle_validation = sqrt(msle_validation)

    print("Validations set rmsle:", rmsle_validation)

    

    # Make predictions on competition test set

    Y_test_prediction = model.predict(X_test)

    Y_test_prediction = np.expm1(Y_test_prediction)

    print("Y test prediction")

    print(Y_test_prediction)

    

    return Y_test_prediction
# Build model

linear_reg = LinearRegression()

Y_test_prediction = use_model(linear_reg)
# Lasso regression

lasso = LassoCV(alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])

Y_test_prediction = use_model(lasso)

alpha = lasso.alpha_

print('Best alpha is:', alpha)
# Use the calculated best alpha and optimize it again

lasso2 = LassoCV(alphas = [alpha*0.1, alpha*0.2, alpha*0.3, alpha*0.4, alpha*0.5, alpha*0.6, alpha*0.7, alpha*0.8, alpha*0.9, alpha*1, alpha*1.2, alpha*1.4, alpha*1.6, alpha*1.8, alpha*2])

Y_test_prediction = use_model(lasso2)

alpha2 = lasso2.alpha_

print('New best alpha is:', alpha2)
coef = pd.Series(lasso.coef_, index = X_train.columns)



print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Ridge regression

ridge = RidgeCV(alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])

Y_test_prediction = use_model(ridge)

alpha = ridge.alpha_

print('Best alpha is:', alpha)
coef = pd.Series(ridge.coef_, index = X_train.columns)



print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Make submission csv file

submission = pd.DataFrame({'Id': test_ID, 'SalePrice': Y_test_prediction})

submission.to_csv('my_submission.csv', index=False)

print("Submission saved")