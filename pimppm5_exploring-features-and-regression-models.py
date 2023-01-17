%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from matplotlib import cm

from sklearn import preprocessing, manifold, linear_model, metrics, model_selection, ensemble

import seaborn as sns
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.head()
data_train.info()
pd.concat([data_train, data_test])['SalePrice'].plot(kind='hist', bins=20, title='Price frequency')
# set features and labels (removing Id from features)

X, y = data_train.iloc[:,1:-1], data_train['SalePrice']

X_test = data_test.iloc[:,1:]
X_all = pd.concat([X, X_test])



# Convert CentralAir to binary feature

X_all['CentralAir'] = X_all['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)



# Convert MSSubClass to categorial feature

X_all['MSSubClass'] = X_all['MSSubClass'].astype(str)
# types of features

binary_features = ['CentralAir']

categorial_features = X_all.select_dtypes(include=[object]).columns.values 

numeric_features = X_all.select_dtypes(exclude=[object]).columns.values

numeric_features = np.delete(numeric_features, np.argwhere(numeric_features=='CentralAir'))
nans = X_all.isnull().sum()

nans = nans[nans > 0]

print(nans)
# 'MiscFeature' and 'PoolQC' have more than 96% nan values, so we can remove them

to_remove = ['MiscFeature', 'PoolQC']

X_all.drop(to_remove, axis=1, inplace=True)

categorial_features = categorial_features[~np.in1d(categorial_features, to_remove)]
# For following categorial columns change NaN for most frequent values

nan2frequent = ['MasVnrType', 'Electrical', 'MSZoning', 'Utilities']

for column in nan2frequent:

    X_all[column].fillna(X_all[column].value_counts().idxmax(), inplace=True)



# For following categorial columns change NaN for new NA category

nan2new = categorial_features[np.in1d(categorial_features, nan2frequent, invert=True)]

for column in nan2new:

    X_all[column].fillna('NA', inplace=True)
# Numeric features with NaN

nans = X_all[numeric_features].isnull().sum()

nan2numeric = nans[nans > 0].index.values

print(nan2numeric)
# Let's look on the distribution of numerical features with many NaNs

X_all[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].hist(bins=80, figsize=(10,5))
# Replace NaNs with medians (for mean they are too skewed)

for column in nan2numeric:

    X_all[column].fillna(X_all[column].median(), inplace=True)
# Check that we didn't miss anything

nans = X_all.isnull().sum()

print(nans[nans > 0])
fig, axes = plt.subplots(9, 5, figsize=(15, 30))

for i, feature in enumerate(categorial_features):

    sns.countplot(x=feature, data=X_all, ax=axes[i//5][i%5])
print(X_all['Street'].value_counts())

print(X_all['Utilities'].value_counts())
# Remove Street and Utilities features

to_remove = ['Street', 'Utilities']

X_all.drop(to_remove, axis=1, inplace=True)

categorial_features = categorial_features[~np.in1d(categorial_features, to_remove)]
expl_data = X_all[:data_train.shape[0]][numeric_features]

expl_data['SalePrice'] = y

# heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(expl_data.corr())
# Select features poorly correlated with target

bad_features = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', '3SsnPorch', 'ScreenPorch', 

                'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

expl_data[bad_features].hist(bins=20, figsize=(9, 9))

plt.show()
# Remove first 3 of these features because almost all their values are 0

to_remove = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath']

X_all.drop(to_remove, axis=1, inplace=True)

numeric_features = numeric_features[~np.in1d(numeric_features, to_remove)]
# encode with dummy features

X_all = pd.get_dummies(data=X_all, columns=categorial_features)

X_all.info()
scaler = preprocessing.StandardScaler()

X_all[numeric_features] = scaler.fit_transform(X_all[numeric_features])

X_all.info()
# extract train and test parts of the data

X = X_all[:data_train.shape[0]]

X_test = X_all[data_train.shape[0]:]

print(X.shape, X_test.shape)
def price_category(y):

    cl = pd.Series(index=y.index)

    cl[y < 100000] = 0

    cl[(y >= 100000) & (y < 150000)] = 1

    cl[(y >= 150000) & (y < 200000)] = 2

    cl[(y >= 200000) & (y < 250000)] = 3

    cl[(y >= 250000) & (y < 300000)] = 4

    cl[y >= 300000] = 5

    return cl

price_classes = price_category(y)

labels = ['<100K', '100-150K', '150-200K', '200-250K', '250-300K', '>300K']
from sklearn.manifold import MDS

mds = MDS(random_state=123)

MDS_transformed = mds.fit_transform(X)



plt.figure(figsize=(10, 8))

colors = cm.rainbow(np.linspace(0, 1, 6))

for cls, color, label in zip(range(6), colors, labels):

    plt.scatter(MDS_transformed[price_classes.values==cls, 0], 

                MDS_transformed[price_classes.values==cls, 1], c=color, alpha=0.5, label=label)

plt.legend()
# Root mean squared logarithmic error (RMSLE) - underprediction is penalized greater than overprediction

def rmsle_score(y, p):

    return -np.sqrt(np.sum((np.log(1+y) - np.log(1+p))**2)/y.shape[0])

rmsle = metrics.make_scorer(rmsle_score)
# Ridge regression: Count RMSLE on cross-validation

param_grid = {

              'alpha': [0.5, 1, 2, 6, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150],

             }



ridge = linear_model.Ridge()

ridge_gs = model_selection.GridSearchCV(ridge, param_grid, cv=3, scoring=rmsle)

ridge_gs.fit(X, y)

print(ridge_gs.best_score_)

print(ridge_gs.best_params_)
plt.plot([item['alpha'] for item in ridge_gs.cv_results_['params']], 

         [-item for item in ridge_gs.cv_results_['mean_test_score']])

plt.xlabel('alpha')

plt.ylabel('RSMLE')

plt.title('Ridge grid search')
# Lasso regression

param_grid = {

              'alpha': [75, 100, 125, 150, 175],

             }

lasso = linear_model.Lasso()

lasso_gs = model_selection.GridSearchCV(lasso, param_grid, cv=3, scoring=rmsle)

lasso_gs.fit(X, y)

print(lasso_gs.best_score_)

print(lasso_gs.best_params_)
plt.plot([item['alpha'] for item in lasso_gs.cv_results_['params']], 

         [-item for item in lasso_gs.cv_results_['mean_test_score']])

plt.xlabel('alpha')

plt.ylabel('RSMLE')

plt.title('Lasso grid search')
# Check how many coefficients become zero

coef = lasso_gs.best_estimator_.coef_

not_zero_indices = np.where(coef!=0)



# Display most important features

large_indices = np.where(abs(coef) >= 5000)

plt.barh(range(len(large_indices[0])), coef[large_indices[0]])

plt.yticks(range(len(large_indices[0])), X.columns[large_indices[0]])

plt.title('Most imporant features')
# let's throw out unimportant features (that become zero in lasso regression)

X_selected = X.iloc[:,not_zero_indices[0]]
# look at the residuals

predicts = lasso_gs.best_estimator_.predict(X)

plt.scatter(predicts, predicts-y, alpha=0.5)

plt.xlabel('true y values')

plt.ylabel('residuals')

plt.show()

print('R2 score: %s' % metrics.r2_score(predicts, y))
# log of y

y_log = np.log(y)

plt.hist(y_log)

plt.xlabel('log y')

plt.show()
# In case of log y Ridge regression perfoms better. 

param_grid = {

              'alpha': [0.005, 0.01, 0.05, 1],

             }

ridge = linear_model.Ridge()

ridge_gs = model_selection.GridSearchCV(ridge, param_grid, cv=3, scoring=rmsle)

ridge_gs.fit(X_selected, y_log)

print(ridge_gs.best_score_)

print(ridge_gs.best_params_)



# the real score

ridge_regr = ridge_gs.best_estimator_

predicts = ridge_regr.predict(X_selected)

rmsle_score(np.exp(y_log), np.exp(predicts))
plt.scatter(np.exp(predicts), np.exp(predicts) - np.exp(y_log), alpha=0.5)

plt.xlabel('true y values')

plt.ylabel('residuals')

plt.show()

print('R2 score: %s' % metrics.r2_score(np.exp(predicts), np.exp(y_log)))
# Square root of y

y_root = np.sqrt(y)

plt.hist(y_root)

plt.xlabel('sqrt y')

plt.show()
# Lasso regression for square root y

param_grid = {

              'alpha': [0.005, 0.01, 0.05, 1],

             }

lasso = linear_model.Lasso()

lasso_gs = model_selection.GridSearchCV(lasso, param_grid, cv=3, scoring=rmsle)

lasso_gs.fit(X_selected, y_root)

print(lasso_gs.best_score_)

print(lasso_gs.best_params_)
# the real score

lasso_regr = lasso_gs.best_estimator_

predicts = lasso_regr.predict(X_selected)

rmsle_score(y_root**2, predicts**2)
plt.scatter(predicts**2, predicts**2 - y_root**2, alpha=0.5)

plt.xlabel('true y values')

plt.ylabel('residuals')

plt.show()

print('R2 score: %s' % metrics.r2_score(predicts**2, y_root**2))
# Tune hyperparameters with grid search

# I've checked more values, this is just for example

param_grid = {

              'n_estimators': [100, 200, 300],

              'min_samples_leaf': [1, 3],  

             }

forest = ensemble.RandomForestRegressor()

forest_gs = model_selection.GridSearchCV(forest, param_grid, cv=3, scoring=rmsle)

forest_gs.fit(X_selected, y)

print(forest_gs.best_score_)

print(forest_gs.best_params_)
# look at the residuals

predicts = forest_gs.best_estimator_.predict(X_selected)

plt.scatter(predicts, predicts-y, alpha=0.5)

plt.xlabel('true y values')

plt.ylabel('residuals')

plt.show()

print('R2 score: %s' % metrics.r2_score(predicts, y))
import xgboost as xgb

xgb_regressor = xgb.XGBRegressor()
# Tune hyperparameters with grid search

# I've checked more values, this is just for example

param_grid = {

              'n_estimators': [400, 500],

              'learning_rate': [0.05, 0.1],

             }

xgb_gs = model_selection.GridSearchCV(xgb_regressor, param_grid, cv=3, scoring=rmsle)

xgb_gs.fit(X_selected, y)

print(xgb_gs.best_score_)

print(xgb_gs.best_params_)
# look at the residuals

predicts = xgb_gs.best_estimator_.predict(X_selected)

plt.scatter(predicts, predicts-y, alpha=0.5)

plt.xlabel('true y values')

plt.ylabel('residuals')

plt.show()

print('R2 score: %s' % metrics.r2_score(predicts, y))
X_test_selected = X_test.iloc[:,not_zero_indices[0]]
best_regressor = lasso_gs.best_estimator_ # Lasso regression {'alpha': 0.05}

best_regressor.fit(X_selected, y_root)

y_test = best_regressor.predict(X_test_selected)

y_test = y_test**2 # back to the real values



result_df = pd.DataFrame(columns=['Id', 'SalePrice'])

result_df.Id = data_test.Id

result_df.SalePrice = y_test

result_df.head()
result_df.to_csv('output.csv', index=False)