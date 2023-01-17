#STEP 1: IMPORTING LIBRARIES AND DATASET

# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate, GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

# Importing the dataset from Kaggle
traindf = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
testdf = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

outcome = 'SalePrice' # Within quotes

traindf[outcome].describe()
# Plotting the curve to understand data distribution
sns.distplot(traindf[outcome], fit=norm);
fig = plt.figure()
res = stats.probplot(traindf[outcome], plot=plt)
# Applying log transformation to resolve skewness
traindf[outcome] = np.log(traindf[outcome])
sns.distplot(traindf[outcome], fit=norm);
fig = plt.figure()
res = stats.probplot(traindf[outcome], plot=plt)
#correlation matrix for all numerical features
cor = traindf.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cor, vmax=.8, square=True, cmap="YlGnBu");
# top 10 correlated numerical features
k = 10 #number of variables for heatmap
cols = cor.nlargest(k, outcome)[outcome].index
cm = np.corrcoef(traindf[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(cm, vmax=.8, cbar=True, cmap="YlGnBu", annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Identifying outliers through scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(traindf[cols], height = 2.5)
plt.show();
#deleting outliers points by index --> GrLivArea
var = 'GrLivArea'
temp = pd.concat([traindf[var], traindf[outcome]], axis=1)
temp.plot.scatter(x=var, y=outcome)
temp.sort_values(by = var, ascending = True)
traindf = traindf.drop(traindf[traindf[var] == 4676].index, axis=0)
traindf = traindf.drop(traindf[traindf[var] == 5642].index, axis=0)


# Identifying outliers through scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(traindf[cols], height = 2.5)
plt.show();
#STEP 3: DATA PRE-PROCESSING AND FEATURE ENGINEERING ON COMBINED DATASET
#finding number of missing data
df = pd.concat([traindf, testdf], axis=0, sort=False).reset_index(drop=True) #combining the datasets
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(50)
#Systematic approach to missing data in each feature

#Columns to fill with 'None'
cols_to_fill = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for col in cols_to_fill:
    df[col] = df[col].fillna('None')

#Columns to fill with 0
cols_to_fill = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in cols_to_fill:
    df[col] = df[col].fillna(0)

#Columns to fill with mean
cols_to_fill = []
for col in cols_to_fill:
    df[col] = df[col].fillna(df[col].mean()[0])
    
#Columns to fill with mode
cols_to_fill = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd','SaleType']
for col in cols_to_fill:
    df[col] = df[col].fillna(df[col].mode()[0])

#Miscelleneous replacements
df['Functional'] = df['Functional'].fillna('Typ')

#Columns to drop
cols_to_drop = ['LotFrontage', 'Utilities', '1stFlrSF', 'GarageArea', 'GarageYrBlt']
for col in cols_to_drop:
    df = df.drop([col], axis=1)

#Check for missing data again
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(50)
# Analysing and normalising target variable
var = 'GrLivArea'
sns.distplot(df[var], fit=norm);
fig = plt.figure()
res = stats.probplot(df[var], plot=plt)

# Applying log transformation to resolve skewness
df[var] = np.log(df[var])
sns.distplot(df[var], fit=norm);
fig = plt.figure()
res = stats.probplot(df[var], plot=plt)

# Analysing and normalising target variable
var = 'TotalBsmtSF'
sns.distplot(df[var], fit=norm);
fig = plt.figure()
res = stats.probplot(df[var], plot=plt)

# Creating a new variable column for 'HasBsmt'
df['HasBsmt'] = 0
df.loc[df['TotalBsmtSF']>0, 'HasBsmt'] = 1

# Applying log transformation to resolve skewness
df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
sns.distplot(df[df[var]>0][var], fit=norm);
fig = plt.figure()
res = stats.probplot(df[df[var]>0][var], plot=plt)
df = df.drop(['HasBsmt'],axis=1)
#recasting numerical data that are actually categorical
cols_to_cast = ['MSSubClass']
for col in cols_to_cast:
    df[col] = df[col].astype(str)

#Label encoding for ordinal values
cols_to_label = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'OverallCond', 
        'YrSold', 'MoSold']

for col in cols_to_label:
    lbl = LabelEncoder()
    lbl.fit(list(df[col].values))
    df[col] = lbl.transform(list(df[col].values))

#OneHotEncoder/get_dummies for remaining categorical features
df = pd.get_dummies(df)
#STEP 4: XGBOOST MODELING WITH PARAMETER TUNING

#Creating train_test_split for cross validation
X = df.loc[df[outcome]>0]
X = X.drop([outcome], axis=1)
y = df[[outcome]]
y = y.drop(y.loc[y[outcome].isnull()].index, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=8)

#Creating DMatrices for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


#Setting initial parameters
params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':0.3,
    'subsample': 0.80,
    'colsample_bytree': 0.80,
    'reg_alpha': 0,
    'reg_lambda': 0,
    # Other parameters
    'objective':'reg:squarederror',
}


#Setting evaluation metrics - MAE from sklearn.metrics
params['eval_metric'] = "mae"

num_boost_round = 5000

#Begin training of XGB model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#replace num_boost_round with best iteration + 1
num_boost_round = model.best_iteration+1

#Establishing baseline MAE
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=8,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=50
)
cv_results
cv_results['test-mae-mean'].min()

#Parameter-tuning for max_depth & min_child_weight (First round)
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(0,12,2)
    for min_child_weight in range(0,12,2)
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


#Parameter-tuning for max_depth & min_child_weight (Second round)
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in [3,4,5]
    for min_child_weight in [3,4,5]
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


#Parameter-tuning for max_depth & min_child_weight (Third round)
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in [3]
    for min_child_weight in [i/10. for i in range(30,50,2)]
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#Updating max_depth and mind_child_weight parameters
params['max_depth'] = 3
params['min_child_weight'] = 3.2
#Recalibrating num_boost_round after parameter updates
num_boost_round = 5000

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#replace num_boost_round with best iteration + 1
num_boost_round = model.best_iteration+1


#Parameter-tuning for subsample & colsample (First round)
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(3,11)]
    for colsample in [i/10. for i in range(3,11)]
]

min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#Parameter-tuning for subsample & colsample (Second round)
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/100. for i in range(80,100)]
    for colsample in [i/100. for i in range(70,90)]
]

min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#Updating subsample and colsample parameters
params['subsample'] = 0.84
params['colsample'] = 0.71
#Recalibrating num_boost_round after parameter updates
num_boost_round = 5000

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#replace num_boost_round with best iteration + 1
num_boost_round = model.best_iteration+1

#Parameter-tuning for reg_alpha & reg_lambda
gridsearch_params = [
    (reg_alpha, reg_lambda)
    for reg_alpha in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    for reg_lambda in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
]

min_mae = float("Inf")
best_params = None

for reg_alpha, reg_lambda in gridsearch_params:
    print("CV with reg_alpha={}, reg_lambda={}".format(
                             reg_alpha,
                             reg_lambda))
    # We update our parameters
    params['reg_alpha'] = reg_alpha
    params['reg_lambda'] = reg_lambda
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (reg_alpha,reg_lambda)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#Updating reg_alpha and reg_lambda parameters
params['reg_alpha'] = 1e-05
params['reg_lambda'] = 0.001
#Resetting num_boost_round to 5000
num_boost_round = 5000

#Parameter-tuning for eta
min_mae = float("Inf")
best_params = None
for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
    print("CV with eta={}".format(eta))

    params['eta'] = eta
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=8,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=50
          )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
params['eta'] = 0.005
model = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

num_boost_round = model.best_iteration + 1
best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)

mean_absolute_error(best_model.predict(dtest), y_test)

testdf = df.loc[df[outcome].isnull()]
testdf = testdf.drop([outcome],axis=1)
sub = pd.DataFrame()
sub['Id'] = testdf['Id']
testdf = xgb.DMatrix(testdf)

y_pred = np.expm1(best_model.predict(testdf))
sub['SalePrice'] = y_pred

sub.to_csv('submission.csv', index=False)