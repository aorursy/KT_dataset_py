# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the test data

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
for df in (train_data, test_data):

    df['Alley'].fillna('None', inplace=True)

    df['Fence'].fillna('None', inplace=True)

    df['MiscFeature'].fillna('None', inplace=True)

    

    df['BsmtFullBath'].fillna(0, inplace=True)

    df['BsmtHalfBath'].fillna(0, inplace=True)

    

    df['KitchenQual'].fillna(train_data['KitchenQual'].mode()[0], inplace=True)

    df['Functional'].fillna(train_data['Functional'].mode()[0], inplace=True)

    df['SaleType'].fillna(train_data['SaleType'].mode()[0], inplace=True)

    df['Utilities'].fillna(train_data['Utilities'].mode()[0], inplace=True)

    df['Exterior1st'].fillna(train_data['Exterior1st'].mode()[0], inplace=True)

    df['Exterior2nd'].fillna(train_data['Exterior2nd'].mode()[0], inplace=True)

    df['Electrical'].fillna(train_data['Electrical'].mode()[0], inplace=True)

    df['Utilities'].fillna(train_data['Utilities'].mode()[0], inplace=True)

    df['Exterior1st'].fillna(train_data['Exterior1st'].mode()[0], inplace=True)

    df['Exterior2nd'].fillna(train_data['Exterior2nd'].mode()[0], inplace=True)

    

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1']+df['BsmtFinSF2']+df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1']+df['BsmtFinSF2'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1']+df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF2']+df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF2'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(0, inplace=True)

    df[df['TotalBsmtSF']!=0]['BsmtQual'].fillna(train_data[train_data['TotalBsmtSF']!=0]['BsmtQual'].mode()[0], inplace=True)

    df['BsmtQual'].fillna('None', inplace=True)

    df[df['TotalBsmtSF']!=0]['BsmtCond'].fillna(train_data[train_data['TotalBsmtSF']!=0]['BsmtCond'].mode()[0], inplace=True)

    df['BsmtCond'].fillna('None', inplace=True)

    df[df['TotalBsmtSF']!=0]['BsmtExposure'].fillna(train_data[train_data['TotalBsmtSF']!=0]['BsmtExposure'].mode()[0], inplace=True)

    df['BsmtExposure'].fillna('None', inplace=True)

    

    df['BsmtFinSF1'].fillna(0, inplace=True)

    df[df['BsmtFinSF1']!=0]['BsmtFinType1'].fillna(train_data[train_data['BsmtFinSF1']!=0]['BsmtFinType1'].mode()[0], inplace=True)

    df['BsmtFinType1'].fillna('Unf', inplace=True)

    

    df['BsmtFinSF2'].fillna(0, inplace=True)

    df[df['BsmtFinSF2']!=0]['BsmtFinType2'].fillna(train_data[train_data['BsmtFinSF2']!=0]['BsmtFinType2'].mode()[0], inplace=True)

    df['BsmtFinType2'].fillna('Unf', inplace=True)

    

    df['BsmtUnfSF'].fillna(df['TotalBsmtSF']-df['BsmtFinSF1']-df['BsmtFinSF2'], inplace=True)

    df[df['BsmtUnfSF']<0]['BsmtUnfSF'] = 0

    

    df['MasVnrArea'].fillna(0, inplace=True)

    df[df['MasVnrArea']!=0]['MasVnrType'].fillna(train_data[train_data['MasVnrArea']!=0]['MasVnrType'].mode()[0], inplace=True)

    df['MasVnrType'].fillna('None', inplace=True)

    

    df['GarageArea'].fillna(0, inplace=True)

    df[df['GarageArea']!=0]['GarageType'].fillna(train_data[train_data['GarageArea']!=0]['GarageType'].mode()[0], inplace=True)

    df['GarageType'].fillna('None', inplace=True)

    df[df['GarageArea']!=0]['GarageFinish'].fillna(train_data[train_data['GarageArea']!=0]['GarageFinish'].mode()[0], inplace=True)

    df['GarageFinish'].fillna('None', inplace=True)

    df[df['GarageArea']!=0]['GarageQual'].fillna(train_data[train_data['GarageArea']!=0]['GarageQual'].mode()[0], inplace=True)

    df['GarageQual'].fillna('None', inplace=True)

    df[df['GarageArea']!=0]['GarageCond'].fillna(train_data[train_data['GarageArea']!=0]['GarageCond'].mode()[0], inplace=True)

    df['GarageCond'].fillna('None', inplace=True)

    

    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

    

    df['Fireplaces'].fillna(0, inplace=True)

    df[df['Fireplaces']!=0]['FireplaceQu'].fillna(train_data[train_data['Fireplaces']!=0]['FireplaceQu'].mode()[0], inplace=True)

    df['FireplaceQu'].fillna('None', inplace=True)

    

    df['PoolArea'].fillna(0, inplace=True)

    df[df['PoolArea']!=0]['PoolQC'].fillna(train_data[train_data['PoolArea']!=0]['PoolQC'].mode()[0], inplace=True)

    df['PoolQC'].fillna('None', inplace=True)
# Drop outliers identified in dataset exploration

train_data = train_data[(train_data['LotFrontage'] < 250)

                        & (train_data['LotArea'] < 100000)

                       & (train_data['BsmtFinSF1'] < 3000)

                       & (train_data['BsmtFinSF1'] < 3000)

                       & (train_data['1stFlrSF'] < 3500)

                       & (train_data['GrLivArea'] < 4000)

                       & (train_data['OpenPorchSF'] < 450)

                       & (train_data['EnclosedPorch'] < 450)

                       & (train_data['MiscVal'] < 3000)]
# Extract the target variable from the training data

train_y = train_data.SalePrice

train_data.drop('SalePrice', axis=1, inplace=True)
# The final competition score is based on the MSE between the log of the test predictions and the log of the true SalePrice.

# With that in mind, always train to fit logy.

train_logy = np.log(train_y)

# Predictions will need to be of y, however, so for the final test submission, take the exponent of its output.
# Separate the Id column from the predictive features

train_X = train_data.drop('Id', axis=1)

test_X = test_data.drop('Id', axis=1)
median_LotFrontages_byNeighborhood = train_X.groupby('Neighborhood')['LotFrontage'].median()



for features in [train_X, test_X]:

    # When LotFrontage is missing, impute median value of LotFrontage based on Neighborhood

    for neighborhood in median_LotFrontages_byNeighborhood.index:

        features[features['Neighborhood']==neighborhood]['LotFrontage'].fillna(median_LotFrontages_byNeighborhood[neighborhood])



    # Add useful features

    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

    features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                                   features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

    features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                                  features['EnclosedPorch'] + features['ScreenPorch'] +

                                  features['WoodDeckSF'])

    features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_X_numeric = train_X.select_dtypes(include=[np.number]).drop('MSSubClass', axis=1)

train_X_categorical = train_X.drop(train_X_numeric.columns, axis=1)



num_cols = train_X_numeric.columns

cat_cols = train_X_categorical.columns
print(num_cols)

print(cat_cols)
# Apply Box-Cox transformation on features with skewness > 0.5

# Lambda=0.30 minimizes the MSLE of a simple linear model



from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



skewed_feats = train_X[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)



for col in num_cols:

    if skewed_feats[col] > 0.5:

        train_X[col] = boxcox1p(train_X[col], 0.3)

        test_X[col] = boxcox1p(test_X[col], 0.3)
print(train_X.head())
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])



cat_pipeline = Pipeline([

    ('cat_nan_filler', SimpleImputer(strategy='constant', fill_value='not_in_data')),

    ('cat_onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor_pipeline = ColumnTransformer([

    ('num_pipeline', num_pipeline, num_cols),

    ('cat_pipeline', cat_pipeline, cat_cols)

])
train_X_preprocessed = preprocessor_pipeline.fit_transform(train_X)
xarray = train_X_preprocessed.toarray()

df = pd.DataFrame(xarray)



overfitted_features = []

for col in df.columns:

    counts = df[col].value_counts()

    zeros = counts.iloc[0]

    if zeros >= len(df)-1:

        overfitted_features.append(col)



df = df.drop(overfitted_features, axis=1)



print(f'Dropping {len(overfitted_features)} features with 1 or fewer filled entries')
train_X_preprocessed = df.values
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import LinearRegression

from mlxtend.regressor import StackingCVRegressor



from sklearn.model_selection import KFold, cross_val_score
def rmsle_cv(model, X, y):

    #add one line of code in order to shuffle the dataset before prior to cross validation strategy

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X)

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))

    print("RMSE ~> %s" % rmse)

    return(rmse)
names = ['ElasticNet', 'Ridge', 'Lasso', 'LGBM', 'XGBoost', 'ExtraTrees', 'RandomForest']

models = [ElasticNet(alpha=0.001, l1_ratio=0.55, max_iter=10000),

                   Ridge(alpha=20.0, max_iter=10000),

                   Lasso(alpha=0.0005, max_iter=10000),

                   LGBMRegressor(n_estimators=600, num_leaves=400, metric='huber',

                        min_child_samples=3, max_depth=3, learning_rate=0.07,

                        colsample_bytree=0.3, min_split_gain=0.003),

                   XGBRegressor(objective='reg:squarederror', gamma=0.02, min_child_samples=2,

                      learning_rate=0.1, colsample_bynode=0.1, max_depth=3, n_estimators=600),

                   ExtraTreesRegressor(n_estimators=300, criterion='mse', bootstrap=True,

                            min_samples_leaf=2, min_impurity_decrease=1e-5),

                   RandomForestRegressor(n_estimators=100, criterion='mse', bootstrap=False,

                               min_samples_split=3, max_features=0.3)]

print('---------------')

print('Model CV scores')

print('---------------')

for name,model in zip(names,models):

    print(f'{name}: ')

    score = rmsle_cv(model, train_X_preprocessed, train_logy)

    print("{:.4f} (+/-{:.4f})".format(score.mean(), score.std()))

    print()
# Best models from model examination

estimators = []

for i in range(10):

    best_models = [ElasticNet(alpha=0.001, l1_ratio=0.55, max_iter=10000),

#                    Ridge(alpha=20.0, max_iter=10000),

                   Lasso(alpha=0.0005, max_iter=10000),

                   LGBMRegressor(n_estimators=600, num_leaves=400, metric='huber',

                        min_child_samples=3, max_depth=3, learning_rate=0.07,

                        colsample_bytree=0.3, min_split_gain=0.003),

#                    XGBRegressor(objective='reg:squarederror', gamma=0.02, min_child_samples=2,

#                       learning_rate=0.1, colsample_bynode=0.1, max_depth=3, n_estimators=600),

#                    ExtraTreesRegressor(n_estimators=300, criterion='mse', bootstrap=True,

#                             min_samples_leaf=2, min_impurity_decrease=1e-5),

                   RandomForestRegressor(n_estimators=100, criterion='mse', bootstrap=False,

                               min_samples_split=3, max_features=0.3)

                  ]

    

    stack_regressor = StackingCVRegressor(regressors=best_models, meta_regressor=LinearRegression())

    estimators.append((str(i), stack_regressor))



from sklearn.ensemble import VotingRegressor

average_estimator = VotingRegressor(estimators)

average_estimator.fit(train_X_preprocessed, train_logy)
print('-------------------')

print('Average model score')

print('-------------------')

score = rmsle_cv(average_estimator, train_X_preprocessed, train_logy)

print("{:.4f} (+/-{:.4f})".format(score.mean(), score.std()))
stack_train_predictions = average_estimator.predict(train_X_preprocessed)



test_X_preprocessed = preprocessor_pipeline.transform(test_X)

test_X_preprocessed = pd.DataFrame(test_X_preprocessed.toarray()).drop(overfitted_features, axis=1).values

stack_test_predictions = average_estimator.predict(test_X_preprocessed)
import matplotlib.pyplot as plt



def plot_train_predictions(train_targets, train_predictions, model_name):

    plt.scatter(train_targets, train_targets, label='SalePrice')

    plt.scatter(train_targets, train_predictions, label=model_name)

    plt.xlabel('Actual log(SalePrice)')

    plt.ylabel('Predicted log(SalePrice)')

    plt.legend()

    plt.show()
plt.hist(train_logy, bins=20)

plt.xlabel('Train data log(SalePrice)')

plt.show()
plt.hist(stack_train_predictions, bins=20)

plt.xlabel('Train predictions log(SalePrice)')

plt.show()
plot_train_predictions(train_logy, stack_train_predictions, 'Stacked Models')
plt.hist(stack_test_predictions, bins=20)

plt.xlabel('Test predictions log(SalePrice)')

plt.show()
## Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': np.exp(stack_test_predictions)})



# # Blend with best submissions from other kernels

# supp1 = pd.read_csv("/kaggle/input/top-8-without-feature-engineering/submission.csv")

# supp2 = pd.read_csv("/kaggle/input/kernel1a1aa47c33/submission.csv")

# supp3 = pd.read_csv("/kaggle/input/house-pricing-elasticnet-ridge-lasso-xgb-top-7/submission.csv")

# output['SalePrice'] = (output['SalePrice']

#                       + supp1['SalePrice']

#                       + supp2['SalePrice']

#                       + supp3['SalePrice'])/4.0



output.to_csv('submission.csv', index=False)