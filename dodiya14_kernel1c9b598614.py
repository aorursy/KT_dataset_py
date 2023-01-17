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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import LinearRegression



import os

print(os.listdir("../input"))
train = pd.read_csv('/kaggle/input/house-price-predictioniiitb/train.csv')

test = pd.read_csv('/kaggle/input/house-price-predictioniiitb/test_toup.csv')
train.shape, test.shape
train.head()
test.head()
train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
train['SalePrice'].hist(bins = 50)
train['GrLivArea'].hist(bins = 50)
train = train[train.GrLivArea < 4000]

train.head()
train.shape
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice']

y.head()
train['SalePrice'].hist(bins = 50)
train_features = train.drop(['SalePrice'], axis=1)

test_features = test

features = pd.concat([train_features, test_features])
features.shape
# Since these column are actually a category , using a numerical number will lead the model to assume

# that it is numerical , so we convert to string .

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)







## Filling these columns With most suitable value for these columns 

features['Functional'] = features['Functional'].fillna('Typ') 

features['Electrical'] = features['Electrical'].fillna("SBrkr") 

features['KitchenQual'] = features['KitchenQual'].fillna("TA") 

features["PoolQC"] = features["PoolQC"].fillna("None")







## Filling these with MODE , i.e. , the most frequent value in these columns .

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

### Missing data in GarageYrBit most probably means missing Garage , so replace NaN with zero . 



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')



    

### Same with basement



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#MsSubclass is type of house and MSZone is type of area in which that house present

#Here we are assuming that for the same MSSubclass there is same MSZone
objects = []

others = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)

    else:

        others.append(i)

features.update(features[objects].fillna('None'))

print(objects)
# We are still filling up missing values 

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))

numerics[:]
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# Removing features that are not very useful . This can be understood only by doing proper EDA on data



features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)





# Adding new features . Make sure that you understand this. 



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])
## For ex, if PoolArea = 0 , Then HasPool = 0 too



features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
features.shape
final_features = pd.get_dummies(features).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape
features.describe()
overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.96:

        overfit.append(i)



overfit
X = X.drop(overfit, axis=1)

X_sub = X_sub.drop(overfit, axis=1)
X.shape, y.shape, X_sub.shape
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



def mse(pred,y):

    n = len(pred)

    val = 0

    for i in range(n):

        val+=(pred[i]-y[i])**2

    return val/n
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
xgboost = XGBRegressor(learning_rate=0.05,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
linreg = LinearRegression()
print('START Fit')



print("Linear Regression")

linreg_model = linreg.fit(X, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.5 * linreg_model.predict(X))+

            (0.25 * gbr_model_full_data.predict(X)) + 

            (0.15 * xgb_model_full_data.predict(X)) + 

            (0.1 * lgb_model_full_data.predict(X)))
print('Predict submission')

s_submission = pd.read_csv("/kaggle/input/house-price-predictioniiitb/samples submission.csv")

submission = pd.read_csv("/kaggle/input/house-price-predictioniiitb/samples submission.csv")

submission.iloc[:,1] = ((np.expm1(blend_models_predict(X_sub))))
submission['SalePrice'] = np.array([int(x) for x in submission['SalePrice']])
submission.to_csv("submission.csv", index=False)
mse(s_submission['SalePrice'],submission['SalePrice'])