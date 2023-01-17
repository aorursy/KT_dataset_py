# First, we need to import the needed packages.

import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.svm import SVC as svc

import sklearn.linear_model as sk

import numpy as np

import math

import matplotlib.pyplot as plt

import scipy.stats as sp

from scipy.special import boxcox1p

from sklearn.feature_selection import RFECV

import sklearn.metrics as mt

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv', index_col='Id')
# Set an array with 'No' feature names.

no_features = ['Alley', 'BsmtQual', 'BsmtCond',

               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

               'FireplaceQu', 'GarageType', 'GarageFinish',

               'GarageQual', 'GarageYrBlt', 'GarageCond', 'PoolQC',

               'Fence', 'MiscFeature']

# Fill NaNs with 'No' value.

train[no_features] = train[no_features].fillna('No')

train.loc[:, "LotFrontage_NA"] = train.LotFrontage.isnull() * 1

train.loc[:, "LotFrontage_NA"] = train.loc[:, "LotFrontage_NA"].astype("object")
def HasFeat(var):

    hasf = var

    hasf[hasf != 'No'] = 1

    hasf[hasf == 'No'] = 0

    return hasf

def feat_eng (tr):

    tr.GarageQual[tr['GarageQual'] == 'No'] = 0

    tr.GarageQual[tr['GarageQual'] == 'Po'] = 1

    tr.GarageQual[tr['GarageQual'] == 'Fa'] = 2

    tr.GarageQual[tr['GarageQual'] == 'TA'] = 3

    tr.GarageQual[tr['GarageQual'] == 'Gd'] = 4

    tr.GarageQual[tr['GarageQual'] == 'Ex'] = 5

    tr.GarageQual = tr.GarageQual.astype('float32')

    tr.GarageCond[tr['GarageCond'] == 'No'] = 0

    tr.GarageCond[tr['GarageCond'] == 'Po'] = 1

    tr.GarageCond[tr['GarageCond'] == 'Fa'] = 2

    tr.GarageCond[tr['GarageCond'] == 'TA'] = 3

    tr.GarageCond[tr['GarageCond'] == 'Gd'] = 4

    tr.GarageCond[tr['GarageCond'] == 'Ex'] = 5

    tr.GarageCond = tr.GarageCond.astype('float32')

    tr.GarageYrBlt[tr['GarageYrBlt'] == 'No'] = 0

    tr.GarageYrBlt = tr.GarageYrBlt.astype('float32')

    tr.Functional[tr['Functional'] == 'Typ'] = 7

    tr.Functional[tr['Functional'] == 'Min1'] = 6

    tr.Functional[tr['Functional'] == 'Min2'] = 5

    tr.Functional[tr['Functional'] == 'Mod'] = 4

    tr.Functional[tr['Functional'] == 'Maj1'] = 3

    tr.Functional[tr['Functional'] == 'Maj2'] = 2

    tr.Functional[tr['Functional'] == 'Sev'] = 1

    tr.Functional[tr['Functional'] == 'Sal'] = 0

    tr.Functional = tr.Functional.astype('float32')

    tr.Fence[tr['Fence'] == 'No'] = 0

    tr.Fence[tr['Fence'] == 'MnWw'] = 1

    tr.Fence[tr['Fence'] == 'GdWo'] = 2

    tr.Fence[tr['Fence'] == 'MnPrv'] = 3

    tr.Fence[tr['Fence'] == 'GdPrv'] = 4

    tr.Fence = tr.Fence.astype('float32')

    tr.KitchenQual[tr['KitchenQual'] == 'Po'] = 1

    tr.KitchenQual[tr['KitchenQual'] == 'Fa'] = 2

    tr.KitchenQual[tr['KitchenQual'] == 'TA'] = 3

    tr.KitchenQual[tr['KitchenQual'] == 'Gd'] = 4

    tr.KitchenQual[tr['KitchenQual'] == 'Ex'] = 5

    tr.KitchenQual = tr.KitchenQual.astype('float32')

    tr.HeatingQC[tr['HeatingQC'] == 'Po'] = 1

    tr.HeatingQC[tr['HeatingQC'] == 'Fa'] = 2

    tr.HeatingQC[tr['HeatingQC'] == 'TA'] = 3

    tr.HeatingQC[tr['HeatingQC'] == 'Gd'] = 4

    tr.HeatingQC[tr['HeatingQC'] == 'Ex'] = 5

    tr.HeatingQC = tr.HeatingQC.astype('float32')

    tr.ExterQual[tr['ExterQual'] == 'Po'] = 1

    tr.ExterQual[tr['ExterQual'] == 'Fa'] = 2

    tr.ExterQual[tr['ExterQual'] == 'TA'] = 3

    tr.ExterQual[tr['ExterQual'] == 'Gd'] = 4

    tr.ExterQual[tr['ExterQual'] == 'Ex'] = 5

    tr.ExterQual = tr.ExterQual.astype('float32')

    tr.BsmtQual[tr['BsmtQual'] == 'No'] = 0

    tr.BsmtQual[tr['BsmtQual'] == 'Po'] = 1

    tr.BsmtQual[tr['BsmtQual'] == 'Fa'] = 2

    tr.BsmtQual[tr['BsmtQual'] == 'TA'] = 3

    tr.BsmtQual[tr['BsmtQual'] == 'Gd'] = 4

    tr.BsmtQual[tr['BsmtQual'] == 'Ex'] = 5

    tr.BsmtQual = tr.BsmtQual.astype('float32')

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'No'] = 0

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'Unf'] = 1

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'LwQ'] = 2

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'Rec'] = 3

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'BLQ'] = 4

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'ALQ'] = 5

    tr.BsmtFinType1[tr['BsmtFinType1'] == 'GLQ'] = 6

    tr.BsmtFinType1 = tr.BsmtFinType1.astype('float32')

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'No'] = 0

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'Unf'] = 1

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'LwQ'] = 2

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'Rec'] = 3

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'BLQ'] = 4

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'ALQ'] = 5

    tr.BsmtFinType2[tr['BsmtFinType2'] == 'GLQ'] = 6

    tr.BsmtFinType2 = tr.BsmtFinType2.astype('float32')

    tr['TotalArea'] = tr['TotalBsmtSF'] + tr['1stFlrSF'] + tr['2ndFlrSF']

    tr['BsmtFinArea'] = (tr['BsmtFinSF1'] + tr['BsmtFinSF2'])/(tr['BsmtFinSF1'] + tr['BsmtFinSF2'] + tr['BsmtUnfSF'])

    tr['MSSubClass'] = tr['MSSubClass'].astype('object')

    return tr



train = feat_eng(train)
# Plotting GrLivArea.

plt.scatter(train['GrLivArea'], train['SalePrice'])

# Plot only the two outliers.

plt.scatter([4676, 5642], [184750,160000], color="red")

# Show plot.

plt.show()
# Removing the outliers.

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Plotting again.

plt.scatter(train['GrLivArea'], train['SalePrice'])

plt.show()
# We need to get all the numerical features from the train dataset.

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num = train.select_dtypes(include=numerics)



# Now we create a Series with the skewness and plot in an ordered line graph.

skew = pd.DataFrame(sp.skew(num), num.keys())

# Sorting values.

skew=skew.sort_values(by=0, ascending=False)

# Setting the size.

plt.figure(figsize=(15,7))

# Plotting the skew.

plt.plot(skew)

# Rotating the features labels.

plt.xticks(rotation=45, fontsize=8)

# Adding labels.

plt.xlabel('Numerical Features', fontsize=10)

plt.ylabel('Skew', fontsize=10)

# Adding limit lines.

plt.axhline(y=1, c='red')

plt.axhline(y=-1, c='red', )

plt.show()

# Fist we define a function to indicate what variables will be transformed.

def boxcox(a, f):

    if sp.skew(a) not in range(-1,1) or sp.kurtosis(a) not in range(-1,1):

        a = boxcox1p(a, 0.05)

        print(f, 'transformed.')

        return a

# Here we create an array of feature keys for transforming later.

transformed_feats = []



# For loop with substitutions, transformations and new features.

for f in train.keys():

    # Categorical features are recognised as objects by read_csv.

    if train[f].dtype == object:

        train[f].fillna(value=train[f].value_counts().idxmax())

        train[f] = train[f].astype('category')

    else:

        # Numerical features.

        train[f] = train[f].fillna(value=np.mean(train[f]))

        # The target feature will be transformed later so we must exclude from this loop for now.

        if f != 'SalePrice':

            train[f] = boxcox(train[f], f)

            train[f + 'Sq'] = (train[f] ** 2)

            train[f + 'Cub'] = (train[f] ** 3)

            transformed_feats.append(f)
# Getting the numerical features.

num = train[skew.index].select_dtypes(include=numerics)

# Creating a Series with skewness values for each feature.

skew = pd.DataFrame(sp.skew(num), num.keys())

skew=skew.sort_values(by=0, ascending=False)

# Plotting in a line graph.

plt.figure(figsize=(15,7))

plt.plot(skew)

# Rotating the features names.

plt.xticks(rotation=45, fontsize=8)

# Adding labels.

plt.xlabel('Numerical Features', fontsize=10)

plt.ylabel('Skew', fontsize=10)

# Adding limits.

plt.axhline(y=1, c='red')

plt.axhline(y=-1, c='red', )

# Showing the graph.

plt.show()
cols = train.columns

train = pd.get_dummies(train, drop_first=True)
# Saving all features for future comparison.

all_features = train.keys()

# Removing features.

train = train.drop(train.loc[:,(train==0).sum()>=1444],axis=1)

train = train.drop(train.loc[:,(train==1).sum()>=1444],axis=1) 

# Getting and printing the remaining features.

remain_features = train.keys()

remov_features = [st for st in all_features if st not in remain_features]

print(len(remov_features), 'features were removed:', remov_features)
# Plot histogram of SalePrice.

plt.hist(train['SalePrice'])

plt.show()

#Plot the QQ-plot

sp.probplot(train['SalePrice'], plot=plt)

plt.show()
# Transformation

y = np.log(train['SalePrice'].values)



# Let's plot again to see the result.

plt.hist(y)

plt.show()

sp.probplot(y, plot=plt)

plt.show()
X = train.drop('SalePrice', axis=1)
def scorer(estimator, X, y):

    y_new = estimator.predict(X)   

    return np.sqrt(mt.mean_squared_error(y, y_new))

est = sk.ElasticNet(l1_ratio=0, alpha=0.017, random_state=1)

fsel = RFECV(est, step=1, cv=15, n_jobs=-1, scoring=scorer)

fsel = fsel.fit(X, y)

important_feat = list(X.loc[:, fsel.ranking_<=130].columns)

X = train.loc[:, important_feat].values

print("Most important features:", important_feat)
def cv_train(X, y, k):

    param = {'learning_rate' : [0.1],

             'n_estimators' : [600],

             'max_depth' : [2]}

    gbm = GradientBoostingRegressor(loss='huber', random_state=1)

    cv1 = GridSearchCV(gbm, param, cv=k, scoring='neg_mean_squared_error')

    cv1.fit(X, y)

    lasso = sk.ElasticNet(random_state=1)

    param = {'l1_ratio' : [0],

             'alpha' : [0.017]}

    cv2 = GridSearchCV(lasso, param, cv=k, scoring='neg_mean_squared_error')

    cv2.fit(X, y)

    print('GBM:', np.sqrt(cv1.best_score_*-1),

          'Lasso:', np.sqrt(cv2.best_score_*-1),

          'Mean:', np.sqrt(((cv1.best_score_+cv2.best_score_)/2)*-1),      

          'Pond Mean (0.8):', np.sqrt(((cv1.best_score_*0.2)+(cv2.best_score_*(0.8)))*-1))

    return cv1, cv2
cv1, cv2 = cv_train(X, y, 20)
loss1 = (y-cv1.predict(X))**2

loss2 = (y-cv2.predict(X))**2

plt.scatter(y, loss1)

plt.scatter(y, loss2)

plt.legend(['GBM', 'Lasso'])

plt.show()



#Plot the QQ-plot

sp.probplot(loss1, plot=plt)

plt.show()

sp.probplot(loss2, plot=plt)

plt.show()
test = pd.read_csv('../input/test.csv')

Id = test.loc[:, "Id"]
test[no_features] = test[no_features].fillna('No')

test.loc[:, "LotFrontage_NA"] = test.LotFrontage.isnull() * 1

test.loc[:, "LotFrontage_NA"] = test.loc[:, "LotFrontage_NA"].astype("object")



test = feat_eng(test)



for f in test.keys():

    # Categorical features are recognised as objects by read_csv.

    if test[f].dtype == object:

        test[f].fillna(value=test[f].value_counts().idxmax())

        test[f] = test[f].astype('category')

    else:

        # Numerical features.

        test[f] = test[f].fillna(value=np.mean(test[f]))

        # The target feature will be transformed later so we must exclude from this loop for now.

        if f in transformed_feats and f != 'SalePrice':

            test[f] = boxcox(test[f], f)

            test[f + 'Sq'] = (test[f] ** 2)

            test[f + 'Cub'] = (test[f] ** 3)            



test = pd.get_dummies(test, drop_first=True)

test = test.loc[:, important_feat]

print(test.loc[:, np.sum(test.isnull()) == len(test)].columns)

for c in list(test.loc[:, np.sum(test.isnull()) == len(test)].columns):

    test.loc[:, c] = 0

print("NAs:", np.sum(np.sum(test.isnull())))
pred = ((math.e**cv2.predict(test))*(0.8)) + ((math.e**cv1.predict(test))*0.2)

sub = pd.read_csv('../input/sample_submission.csv')

pd.DataFrame({"Id": Id.values, "SalePrice": pred}).to_csv("submission.csv", index=False)