! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import dabl



from collections import Counter



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

pd.options.display.max_columns = 8000
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data.head()
train_data.shape
train_data.isna().sum()
test_data.isna().sum()
# Remove outliers from OverallQual, GrLivArea and SalesPrice

train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index, inplace=True)

train_data.drop(train_data[(train_data['GrLivArea']>4500) & (train_data['SalePrice']<300000)].index, inplace=True)

train_data.reset_index(drop=True, inplace=True)
train_labels = train_data['SalePrice']

train_features = train_data.drop(['SalePrice'], axis=1)



data = pd.concat([train_features, test_data]).reset_index(drop=True)
# These columns have a lot of Null values, so we drop them

data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

data.head()
plt.style.use("fivethirtyeight")

plt.figure(figsize=(16, 9))

sns.distplot(data['MSSubClass'])

plt.xlabel("Type of Dwelling")

plt.ylabel("Count")

plt.title("Dwelling Type Count")

plt.show()
plt.style.use("ggplot")

plt.figure(figsize=(16, 9))

sns.countplot(data['MSZoning'])

plt.xlabel("Type of Zoning of the property")

plt.ylabel("Count")

plt.title("Zone Type Count")

plt.show()
plt.style.use("classic")

plt.figure(figsize=(16, 9))

sns.distplot(data['LotFrontage'])

plt.xlabel("Lot Frontage (in ft)")

plt.ylabel("Count")

plt.title("Lot Frontage Distribution")

plt.show()
plt.style.use("classic")

plt.figure(figsize=(16, 9))

sns.distplot(train_labels, color='red')

plt.xlabel("Price (in $)")

plt.ylabel("Count")

plt.title("Sales Price Distribution")

plt.show()
sns.pairplot(data.corr())
plt.figure(figsize=(16, 9))

sns.heatmap(data.corr())

plt.show()
dabl.plot(train_data, target_col='SalePrice')
train_labels = train_labels.apply(lambda x: np.log(1+x))
plt.style.use("classic")

plt.figure(figsize=(16, 9))

sns.distplot(train_labels, color='red')

plt.xlabel("Price (in $)")

plt.ylabel("Count")

plt.title("Sales Price Distribution")

plt.show()
data['MSSubClass'] = data['MSSubClass'].apply(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)



# the data description states that NA refers to typical ('Typ') values

data['Functional'] = data['Functional'].fillna('Typ')

# Replace the missing values in each of the columns below with their mode

data['Electrical'] = data['Electrical'].fillna("SBrkr")

data['KitchenQual'] = data['KitchenQual'].fillna("TA")

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



# Replacing the missing values with 0, since no garage = no cars in garage

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)

# Replacing the missing values with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data[col] = data[col].fillna('None')

# NaN values for these categorical basement features, means there's no basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')



# Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# We have no particular intuition around how to fill in the rest of the categorical features

# So we replace their missing values with None

objects = []

for i in data.columns:

    if data[i].dtype == object:

        objects.append(i)

data.update(data[objects].fillna('None'))



# And we do the same thing for numerical features, but this time with 0s

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in data.columns:

    if data[i].dtype in numeric_dtypes:

        numeric.append(i)

data.update(data[numeric].fillna(0))
data.isna().sum()
data = data.drop(['Id'], axis=1)
data.head()
# Make a list of all categorical columns

cat_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']



# Get the dummy variables from them

data = pd.get_dummies(data, columns=cat_cols)
# Recheck the shape of the data

data.shape
# Remove any repeated columns

data = data.iloc[:, ~data.columns.duplicated()]
# Identify the split percent and split the data

train = data[:len(train_labels)]

test = data[len(train_labels):]
train.head()
# Define some metrics



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, train=train, train_labels=train_labels):

    rmse = np.sqrt(-cross_val_score(model, train.values, train_labels.values, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
# First Make 5-Folds for cross validation

kf = KFold(n_splits=10, shuffle=True)
# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:squarederror',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)



# Ridge Regressor

ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)  



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)



# Stack up all the models above, optimized using xgboost

stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
# List of all regressors

regs = [(lightgbm, "Light Gradient Boosting Regressor"), (xgboost, "X-Gradient Boosting Regressor"), (ridge, "Ridge Regressor"), (svr, "Support Vector Regressor"), (gbr, "Gradient Boosting Regressor"), (rf, "Random Forest Regressor"), (stack_gen, "All Model Stacked")]



# We will store all the scores in here

cv_scores = {}



# Calculate CV-RMSE Scores for all regressors

for reg, reg_name in regs:

    sc = cv_rmse(reg)

    cv_scores[reg_name] = (sc.mean(), sc.std())

    print(f"Calculating CV-RMSE for {reg_name} ==> Score Mean: {sc.mean():.2f} | Score Std: {sc.std():.2f}")
# Now we fit all the above models and then get the final model which we will use to blend our predictions

for model, model_name in regs:

    print('='*40)

    print(f"Fitting {model_name}...")

    model.fit(train.values, train_labels.values)

    val_score_temp = model.score(train.values, train_labels.values)

    print(f"val_acc: {val_score_temp:.2f}")
def blended_predictions(X):

    return ((0.1 * regs[2][0].predict(X)) + \

            (0.2 * regs[3][0].predict(X)) + \

            (0.1 * regs[4][0].predict(X)) + \

            (0.1 * regs[1][0].predict(X)) + \

            (0.1 * regs[0][0].predict(X)) + \

            (0.05 * regs[5][0].predict(X)) + \

            (0.35 * regs[6][0].predict(np.array(X))))
blended_score = rmsle(train_labels.values, blended_predictions(train.values))

cv_scores['blended'] = (blended_score, 0)

print('RMSLE score on train data:')

print(blended_score)
# Plot the predictions for each model

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(cv_scores.keys()), y=[score for score, _ in cv_scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(cv_scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
# First we load the submission file

sub = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
# Now we make predictions on the test data

preds = blended_predictions(test.values)
sub['SalePrice'] = np.floor(np.expm1(preds))
sub.to_csv("submission_fixed.csv", index=False)