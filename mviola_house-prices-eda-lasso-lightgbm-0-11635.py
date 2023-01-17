import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')



from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from category_encoders import TargetEncoder

from scipy import stats



from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import Lasso

from lightgbm import LGBMRegressor



import warnings

warnings.simplefilter('ignore')

print('Setup complete')
# Read data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
def plot_numerical(col, discrete=False):

    if discrete:

        fig, ax = plt.subplots(1,2,figsize=(12,6))

        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])

        sns.countplot(train[col], ax=ax[1])

        fig.suptitle(str(col) + ' analysis')

    else:

        fig, ax = plt.subplots(1,2,figsize=(12,6))

        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])

        sns.distplot(train[col], kde=False, ax=ax[1])

        fig.suptitle(str(col) + ' analysis')



def plot_categorical(col):

    fig, ax = plt.subplots(1,2,figsize=(12,6), sharey=True)

    sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])

    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax[1])

    fig.suptitle(str(col) + ' analysis')

    

print('plot_numerical() & plot_categorical() are ready to use')
plt.figure(figsize=(8,5))

a = sns.distplot(train.SalePrice, kde=False)

plt.title('SalePrice distribution')

a = plt.axvline(train.SalePrice.describe()['25%'], color='b')

a = plt.axvline(train.SalePrice.describe()['75%'], color='b')

print('SalePrice description:')

print(train.SalePrice.describe().to_string())
# Select numerical features only

num_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]

# Remove Id & SalePrice 

num_features.remove('Id')

num_features.remove('SalePrice')

# Create a numerical columns only dataframe

num_analysis = train[num_features].copy()

# Impute missing values with the median just for the moment

for col in num_features:

    if num_analysis[col].isnull().sum() > 0:

        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1,1))

# Train a model   

clf = ExtraTreesRegressor(random_state=42)

h = clf.fit(num_analysis, train.SalePrice)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,num_features)), columns=['Value','Feature'])

plt.figure(figsize=(16,10))

sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))

plt.title('Most important numerical features with ExtraTreesRegressor')

del clf, h;
plt.figure(figsize=(8,8))

plt.title('Correlation matrix with SalePrice')

selected_columns = ['OverallQual', 'GarageCars', 'GrLivArea', 'YearBuilt', 'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea']

a = sns.heatmap(train[selected_columns + ['SalePrice']].corr(), annot=True, square=True)
plot_numerical('OverallQual', True)
plot_numerical('GarageCars', True)
plot_numerical('GrLivArea')
plot_numerical('YearBuilt')
plot_numerical('FullBath', True)
plot_numerical('1stFlrSF')
plot_numerical('TotalBsmtSF')
plot_numerical('GarageArea')
# Select categorical features only

cat_features = [col for col in train.columns if train[col].dtype =='object']

# Create a categorical columns only dataframe

cat_analysis = train[cat_features].copy()

# Impute missing values with NA just for the moment

for col in cat_analysis:

    if cat_analysis[col].isnull().sum() > 0:

        cat_analysis[col] = SimpleImputer(strategy='constant').fit_transform(cat_analysis[col].values.reshape(-1,1))

# Target encoding

target_enc = TargetEncoder(cols=cat_features)

cat_analysis = target_enc.fit_transform(cat_analysis, train.SalePrice) 

# Train a model 

clf = ExtraTreesRegressor(random_state=42)

h = clf.fit(cat_analysis, train.SalePrice)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cat_features)), columns=['Value','Feature'])

plt.figure(figsize=(16,10))

sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))

plt.title('Most important categorical features with ExtraTreesRegressor')

del clf, h;
fig, ax = plt.subplots(1,2,figsize=(16,6), sharey=True)

sns.stripplot(x='Neighborhood', y='SalePrice', data=train, ax=ax[0])

sns.boxplot(x='Neighborhood', y='SalePrice', data=train, ax=ax[1])

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

fig.suptitle('Neighborhood analysis')

plt.show()
plot_categorical('ExterQual')
plot_categorical('BsmtQual')
plot_categorical('KitchenQual')
fig, ax = plt.subplots(1,2,figsize=(16,6), sharey=True)

train_missing = round(train.isnull().mean()*100, 2)

train_missing = train_missing[train_missing > 0]

train_missing.sort_values(inplace=True)

sns.barplot(train_missing.index, train_missing, ax=ax[0], color='orange')

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

ax[0].set_ylabel('Percentage of missing values')

ax[0].set_title('Train set')

test_missing = round(test.isnull().mean()*100, 2)

test_missing = test_missing[test_missing > 0]

test_missing.sort_values(inplace=True)

sns.barplot(test_missing.index, test_missing, ax=ax[1], color='orange')

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

ax[1].set_title('Test set')

plt.show()
plot_numerical('LotFrontage')
print('LotFrontage minimum:', train.LotFrontage.min())
plot_categorical('FireplaceQu')
fig, ax = plt.subplots(2,2,figsize=(12,10), sharey=True)

sns.stripplot(x='Fence', y='SalePrice', data=train, ax=ax[0][0])

sns.stripplot(x='Alley', y='SalePrice', data=train, ax=ax[0][1])

sns.stripplot(x='MiscFeature', y='SalePrice', data=train, ax=ax[1][0])

sns.stripplot(x='PoolQC', y='SalePrice', data=train, ax=ax[1][1])

fig.suptitle('Analysis of columns with more than 80% of missing values')

plt.show()
# Fill high % missing columns with NA

for col in ['FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC']:

    train[col].fillna('NotAvailable', inplace=True)

    test[col].fillna('NotAvailable', inplace=True)

    

# Fill Garage variables with NA

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    train[col].fillna('NotAvailable', inplace=True)

    test[col].fillna('NotAvailable', inplace=True)

    

# Fill GarageYrBlt with a new value

# I chose the min over train + test set minus 1

garage_min = min(train['GarageYrBlt'].min(), test['GarageYrBlt'].min()) -1

train['GarageYrBlt'].fillna(garage_min, inplace=True)

test['GarageYrBlt'].fillna(garage_min, inplace=True)



# Fill these Garage columns with 0

for col in ['GarageArea', 'GarageCars']:

    train[col].fillna(0, inplace=True)

    test[col].fillna(0, inplace=True)

    

# Fill numerical Bsmt variables with 0

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:

    train[col].fillna(0, inplace=True)

    test[col].fillna(0, inplace=True)

    

# Fill categorical Bsmt variables with NA  

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:

    train[col].fillna('NotAvailable', inplace=True)

    test[col].fillna('NotAvailable', inplace=True)

    

# Fill low % missing columns with the mode

for col in ['MasVnrType', 'MSZoning', 'Functional', 'Electrical', 'KitchenQual', 'Exterior1st',

           'Exterior2nd', 'SaleType','MSSubClass', 'Utilities', 'MasVnrArea']:

    imputer = SimpleImputer(strategy='most_frequent')

    train[col] = imputer.fit_transform(train[col].values.reshape(-1,1))

    test[col] = imputer.transform(test[col].values.reshape(-1,1))

    

# Fill LotFrontage with KNNImputer on LotArea

imputer = KNNImputer(n_neighbors=5)

train[['LotFrontage', 'LotArea']] = imputer.fit_transform(train[['LotFrontage', 'LotArea']])

test[['LotFrontage', 'LotArea']] = imputer.transform(test[['LotFrontage', 'LotArea']])



print('Imputation finished!')

print('Number of missing values in the train set:', train.isnull().sum().sum())

print('Number of missing values in the test set:', test.isnull().sum().sum())
# Select categorical features

cat_features = [col for col in train.columns if train[col].dtype =='object']

cat_features = cat_features + ['MSSubClass','MoSold','YrSold']

# Select numerical features

num_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64'] and col not in cat_features]

# Remove Id & SalePrice 

num_features.remove('Id')

num_features.remove('SalePrice')

# Encode categorical features

target_enc = TargetEncoder(cols=cat_features)

train_TE = train[num_features].join(target_enc.fit_transform(train[cat_features], train.SalePrice)) 

# Train a model

clf = ExtraTreesRegressor(random_state=0)

h = clf.fit(train_TE, train.SalePrice)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,train_TE)), columns=['Value','Feature'])

plt.figure(figsize=(18,14))

sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))

plt.title('Feature importance with ExtraTreesRegressor')

del clf, h;
cols_to_keep = list(feature_imp.sort_values(by='Value', ascending=False).reset_index(drop=True).loc[:59, 'Feature'])

print('Keeping the first {:d} most informative features'.format(len(cols_to_keep)))
fig, ax = plt.subplots(2,2,figsize=(12,10), sharey=True)

plt.suptitle('Outliers analysis')

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, ax=ax[0][0])

sns.scatterplot(x='OverallQual', y='SalePrice', data=train, ax=ax[0][1])

sns.scatterplot(x='LotFrontage', y='SalePrice', data=train, ax=ax[1][0])

a = sns.scatterplot(x='LotArea', y='SalePrice', data=train, ax=ax[1][1])
outliers_id = set(train.loc[train.GrLivArea > 4500, 'Id']) 

print('Outliers saved in outliers_id')
# Applies log to all elements of the target column

y_train = train.loc[~train.Id.isin(outliers_id), 'SalePrice'].apply(lambda x: np.log(x)).reset_index(drop=True)

# Plot the difference

plt.figure(figsize=(16,6))

plt.subplot(1, 2, 1)

plt.title('Log transformed SalePrice')

sns.distplot(y_train, fit=stats.norm)

#Getting the parameters used by the function

(mu, sigma) = stats.norm.fit(y_train)

plt.legend(['Normal distr. ($\mu=${:.2f}, $\sigma=${:.2f})'.format(mu, sigma)])

plt.ylabel('Density')

#QQ-plot

plt.subplot(1, 2, 2)

a = stats.probplot(y_train, plot=plt)
# Define the train set

X_train = train.loc[~train.Id.isin(outliers_id), cols_to_keep].reset_index(drop=True)

X_test = test[cols_to_keep]

# Intersect cat_features and cols_to_keep

cat_cols = [col for col in cols_to_keep if col in cat_features]

# Intersect num_features and cols_to_keep

num_cols = [col for col in cols_to_keep if col not in cat_features]



# Lasso preprocessing 

lasso_preprocessor = ColumnTransformer(transformers=[

        ('num', PowerTransformer(standardize=True), num_cols),

        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)

])
oof_lasso = np.zeros(len(X_train))

preds_lasso = np.zeros(len(X_test))

out_preds_lasso = np.zeros(2)



kf = KFold(n_splits=5, random_state=42, shuffle=True)

print('Training {:d} Lasso models...\n'.format(kf.n_splits))



for i, (idxT, idxV) in enumerate(kf.split(X_train, y_train)):

    print('Fold', i)

    print(' rows of train =', len(idxT),'rows of holdout =', len(idxV))

  

    train_prep = lasso_preprocessor.fit_transform(X_train.loc[idxT])

    valid_prep = lasso_preprocessor.transform(X_train.loc[idxV])

    out_prep = lasso_preprocessor.transform(train.loc[train.Id.isin(outliers_id), cols_to_keep])

    test_prep = lasso_preprocessor.transform(X_test)

    

    clf = Lasso(alpha=0.0005)

    h = clf.fit(train_prep, y_train.loc[idxT])

    

    oof_lasso[idxV] += clf.predict(valid_prep)

    out_preds_lasso += clf.predict(out_prep)/kf.n_splits

    preds_lasso += clf.predict(test_prep)/kf.n_splits

    

    del h, clf

    

print ('\nOOF CV:', mean_squared_error(y_train, oof_lasso, squared=False))
fig, ax = plt.subplots(1,2,figsize=(16,6))

ax[0].set_title('Lasso OOF predictions vs real SalePrice')

ax[0].set_xlabel('OOF predictions')

sns.scatterplot(np.exp(oof_lasso), np.exp(y_train), ax=ax[0])

x = np.linspace(0,700000, 100)

ax[0].plot(x, x, 'r-', lw=1)

# Residuals plot

ax[1].set_title('Residuals')

sns.scatterplot(x=np.exp(oof_lasso), y=np.exp(y_train)-np.exp(oof_lasso), ax=ax[1])

ax[1].set_ylabel('SalePrice - OOF predictions')

ax[1].set_xlabel('OOF predictions')

plt.show()
plt.figure(figsize=(8,5))

plt.title('Train vs predictions comparision')

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, label='Real train')

sns.scatterplot(x=train.loc[train.Id.isin(outliers_id), 'GrLivArea'], y = np.exp(out_preds_lasso), label='Outliers predictions')

a = sns.scatterplot(test.GrLivArea, np.exp(preds_lasso), label='Test predictions')
lasso_submission = np.exp(preds_lasso)

lasso_submission[1089] = train.loc[train.GrLivArea>4500, 'SalePrice'].mean()
tree_preprocessor = ColumnTransformer(transformers=[

        ('num', 'passthrough', num_cols),

        ('cat', TargetEncoder(cols=cat_cols), cat_cols)

])
oof_LGBM = np.zeros(len(X_train))

preds_LGBM = np.zeros(len(X_test))

out_preds_LGBM = np.zeros(2)



kf = KFold(n_splits=5, random_state=42, shuffle=True)

print('Training {:d} LGBM models...\n'.format(kf.n_splits))



for i, (idxT, idxV) in enumerate(kf.split(X_train, y_train)):

    print('Fold', i)

    print(' rows of train =', len(idxT),'rows of holdout =', len(idxV))

  

    train_prep = tree_preprocessor.fit_transform(X_train.loc[idxT], y_train.loc[idxT])

    valid_prep = tree_preprocessor.transform(X_train.loc[idxV])

    out_prep = tree_preprocessor.transform(train.loc[train.Id.isin(outliers_id), cols_to_keep])

    test_prep = tree_preprocessor.transform(X_test)

    

    clf = LGBMRegressor(n_estimators=5000, random_state=42, num_leaves=8, colsample_bytree=0.8)

    h = clf.fit(train_prep, y_train.loc[idxT], 

                eval_set=[(valid_prep, y_train.loc[idxV])],

                verbose=300, early_stopping_rounds=200, eval_metric='rmse')

    

    oof_LGBM[idxV] += clf.predict(valid_prep)

    out_preds_LGBM += clf.predict(out_prep)/kf.n_splits

    preds_LGBM += clf.predict(test_prep)/kf.n_splits

    

    del h, clf

    

print ('\nOOF CV', mean_squared_error(y_train, oof_LGBM, squared=False))
fig, ax = plt.subplots(1,2,figsize=(16,6))

ax[0].set_title('LGBM OOF predictions vs SalePrice')

ax[0].set_xlabel('OOF predictions')

sns.scatterplot(np.exp(oof_LGBM), np.exp(y_train), ax=ax[0])

x = np.linspace(0,650000, 100)

ax[0].plot(x, x, 'r-', lw=1)

# Residuals plot

ax[1].set_title('Residuals')

sns.scatterplot(x=np.exp(oof_LGBM), y=np.exp(y_train)-np.exp(oof_LGBM), ax=ax[1])

ax[1].set_ylabel('SalePrice - OOF predictions')

ax[1].set_xlabel('OOF predictions')

plt.show()
plt.figure(figsize=(8,5))

plt.title('Train vs predictions comparision')

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, label='Real train')

sns.scatterplot(x=train.loc[train.Id.isin(outliers_id), 'GrLivArea'], y = np.exp(out_preds_LGBM), label='Outliers predictions')

a = sns.scatterplot(test.GrLivArea, np.exp(preds_LGBM), label='Test predictions')
LGBM_submission = np.exp(preds_LGBM)

LGBM_submission[1089] = train.loc[train.GrLivArea>4500, 'SalePrice'].mean()
all = []

for w in np.linspace(0,1,101):

    ensemble_pred = w * oof_lasso + (1-w) * oof_LGBM

    ensemble_rmsle = mean_squared_error(ensemble_pred, y_train, squared=False)

    all.append(ensemble_rmsle)

# Print the best weight

best_weight = np.argmin(all)/100

print('Best weight =', best_weight)

# Plot the ensemble_rmsle against w

plt.figure(figsize=(8,5))

plt.title('Ensemble RMSLE against weight $w$')

plt.xlabel('$w$')

plt.ylabel('Ensemble RMSLE')

a = sns.scatterplot(x=np.linspace(0,1,101), y=all, color='orange')

plt.legend(['$w\cdot$oof_lasso + $(1-w)\cdot$oof_LGBM'])

plt.show()
kaggle_sub = best_weight*lasso_submission + (1-best_weight)*LGBM_submission

# Save the output

output = pd.DataFrame({'Id': test.Id, 'SalePrice': kaggle_sub})

output.to_csv('my_submission.csv', index=False)

print('Your submission was successfully saved!')