import math
import pandas as pd
import numpy as np
from scipy.stats import stats, norm, skew
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.special import boxcox1p
import lightgbm as lgb
import xgboost as xgb

%matplotlib inline
np.random.seed(2)
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.shape)
print(test.shape)
train = train.drop('Id', axis=1)
train.columns

cols_with_none_as_nan = [
    "PoolQC", 
    "MiscFeature", 
    "Alley", 
    "Fence", 
    "FireplaceQu", 
    'GarageType', 
    'GarageFinish', 
    'GarageQual', 
    'GarageCond', 
    'BsmtQual', 
    'BsmtCond', 
    'BsmtExposure', 
    'Electrical', 
    'BsmtFinType1', 
    'BsmtFinType2', 
    "MasVnrType",
    'MSZoning',
    'Utilities',
    'Exterior1st',
    'Exterior2nd',
    'KitchenQual',
    'Functional',
    'SaleType']


o = train.dtypes[train.dtypes==object].index
print(o)

# fill missing text fields with a default string
object_columns = train[o]
test_object_columns = test[o]

# for these colunms the string 'None' will be inserted in place of nan
for col in cols_with_none_as_nan:
    object_columns.loc[:, col] = object_columns.loc[:, col].fillna('None')
    test_object_columns.loc[:, col] = test_object_columns.loc[:, col].fillna('None')

remaining_fix = object_columns.isnull().sum()
print('Fixes remaining on train set\n', remaining_fix[remaining_fix>0])

remaining_fix = test_object_columns.isnull().sum()
print('Fixes remaining on test set\n',remaining_fix[remaining_fix>0])
# Base scale, taken from 'OverallCond' feature
#   10   Very Excellent
#   9    Excellent
#   8    Very Good
#   7    Good
#   6    Above Average   
#   5    Average
#   4    Below Average   
#   3    Fair
#   2    Poor
#   1    Very Poor

#   Ex   Excellent (100+ inches) 
#   Gd   Good (90-99 inches)
#   TA   Typical (80-89 inches)
#   Fa   Fair (70-79 inches)
#   Po   Poor (&lt;70 inches
#   NA   No Basement

quals_mapping = {
    "Ex": 9,
    "Gd": 7,
    "Av": 5,
    "TA": 5,
    "Fa": 3,
    "Po": 2,
    "NA": 0,
    "None": 0
}
quals_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', ]

for c in object_columns.columns:
    if c in quals_columns:
        vals = np.unique(object_columns[c])
        for v in quals_mapping:
            object_columns.loc[:,c] = object_columns[c].replace(to_replace=v, value=quals_mapping[v])
            test_object_columns.loc[:,c] = test_object_columns[c].replace(to_replace=v, value=quals_mapping[v])

numeric_columns = train.select_dtypes(include=[int, float])

remaining_fix = numeric_columns.isnull().sum()
print('Fixes remaining on train set\n',remaining_fix[remaining_fix>0])

test_numeric_columns = test.select_dtypes(include=[int, float])

remaining_fix = test_numeric_columns.isnull().sum()
print('Fixes remaining on test set\n',remaining_fix[remaining_fix>0])
cols_with_zero_as_nan = ['MasVnrArea', 
                         'GarageYrBlt',
                         'BsmtFinSF1',
                         'BsmtFinSF2',
                         'BsmtUnfSF',
                         'TotalBsmtSF',
                         'BsmtFullBath',
                         'BsmtHalfBath',
                         'GarageCars',
                         'GarageArea',
                        ]
cols_with_mean_as_nan = ['LotFrontage']

# for these colunms the mean will be inserted in place of nan
for col in cols_with_mean_as_nan:
    numeric_columns.loc[:, col] = numeric_columns.loc[:, col].fillna(numeric_columns[col].mean())
    test_numeric_columns.loc[:, col] = test_numeric_columns.loc[:, col].fillna(test_numeric_columns[col].mean())

# for these colunms a zero will be inserted in place of nan
for col in cols_with_zero_as_nan:
    numeric_columns.loc[:, col] = numeric_columns.loc[:, col].fillna(0)
    test_numeric_columns.loc[:, col] = test_numeric_columns.loc[:, col].fillna(0)


remaining_fix = numeric_columns.isnull().sum()
print('Fixes remaining on train set\n',remaining_fix[remaining_fix>0])

remaining_fix = test_numeric_columns.isnull().sum()
print('Fixes remaining on test set\n',remaining_fix[remaining_fix>0])
plt.figure(figsize=(10,5))
sns.scatterplot(x=numeric_columns.loc[:,'OverallQual'], y=np.log1p(train['SalePrice']))
plt.show()

for d in numeric_columns[numeric_columns['OverallQual'] == 4].index:
    if np.log1p(numeric_columns.loc[d,'SalePrice']) < 10.8:
        numeric_columns = numeric_columns.drop(d, axis=0)
        object_columns = object_columns.drop(d, axis=0)

plt.figure(figsize=(10,5))
sns.scatterplot(x=numeric_columns.loc[:,'OverallQual'], y=np.log1p(train['SalePrice']))
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x=numeric_columns['GrLivArea'], y=np.log1p(train['SalePrice']))
plt.show()

for d in numeric_columns[numeric_columns['GrLivArea'] > 4000].index:
    numeric_columns = numeric_columns.drop(d, axis=0)
    object_columns = object_columns.drop(d, axis=0)

plt.figure(figsize=(10,5))
sns.scatterplot(x=numeric_columns['GrLivArea'], y=np.log1p(train['SalePrice']))
plt.show()


strings = np.array([])
for c in object_columns.columns:
    if c not in quals_columns:
        strings = np.append(strings, pd.unique(object_columns[c].values))

for c in test_object_columns.columns:
    if c not in quals_columns:
        strings = np.append(strings, pd.unique(test_object_columns[c].values))

print(len(strings), 'distinct labels generated')

labeler = LabelEncoder()
labeler.fit(strings)

for c in object_columns.columns:
    if c not in quals_columns:
        object_columns.loc[:,c] = labeler.transform(object_columns.loc[:,c])
        test_object_columns.loc[:,c] = labeler.transform(test_object_columns.loc[:,c])

object_columns[0:5]
# final train dataset
train_ds = object_columns.join(numeric_columns)
train_ds["SalePrice"] = np.log1p(train_ds["SalePrice"])

# test dataset
test_ds = test_object_columns.join(test_numeric_columns)


# keeping only the most promising features
corrmat = train_ds.corr()['SalePrice']
best_columns = corrmat[abs(corrmat) > 0.01].index

print('Keeping only ', len(best_columns), 'features out of ', len(train_ds.columns))

train_ds = train_ds[best_columns]
test_ds = test_ds[best_columns.drop('SalePrice')]

X_train = (train_ds.values[:,:-1])
y_train = np.asarray([[t] for t in (train_ds.values[:,-1])])
X_test = test_ds.values

print('Training set features shape', X_train.shape)
print('Training set labels shape', y_train.shape)
print('Test set shape', test_ds.shape)

print(test_ds.dtypes[test_ds.dtypes==object])

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X_train)
X_train = pd.DataFrame(transformer.transform(X_train))
X_test = pd.DataFrame(transformer.transform(X_test))

print('Training set features shape', X_train.shape)
print('Training set labels shape', y_train.shape)

def train_model(depth, learning_rate, n_estimators, model_type='xgb'):
    if model_type == 'xgb':
        print('XGBRegressor')
        model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=depth, 
                                 n_estimators=n_estimators,
                                 silent=1,
                                 random_state =7, nthread = -1)
    if model_type == 'lgb':
        print('LGBMRegressor')
        model = lgb.LGBMRegressor(learning_rate=learning_rate, max_depth=depth, 
                                 n_estimators=n_estimators,
                                 silent=1,
                                 random_state =7, nthread = -1)
    if model_type == 'lasso':
        print('Lasso')
        model = Lasso(alpha =0.0006)
        
    if model_type == 'elastic':
        print('ElasticNet')
        model = ElasticNet(alpha=0.0012, l1_ratio=.5)
        
    score = rmsle_cv(model)
    print(model_type, " score: depth={:d} lr={:.2f} est={:d} -> mean:{:.5f} std:{:.4f}".format(depth, learning_rate, n_estimators, score.mean(), score.std()))
    return score

#Validation function
n_folds = 10

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train.flatten(), scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

depth = 0
learning_rate = 0
n_estimators = 0

score = train_model(depth, learning_rate/100, n_estimators, model_type='lasso')

# train with the best parameters
model_lasso = Lasso(alpha=0.0006)

score = rmsle_cv(model_lasso)
print("model_lr score: depth={:d} lr={:.2f} est={:d} -> mean:{:.5f} std:{:.4f}".format(depth, learning_rate, n_estimators, score.mean(), score.std()))
model_lasso.fit(X_train, y_train)
y_pred = model_lasso.predict(X_train)
print('RMSLE LR = ', rmsle(y_train, y_pred))

plt.figure(figsize=(8, 8))
y_pred = model_lasso.predict(X_train)
plt.scatter(np.expm1(y_train), np.expm1(y_pred))
plt.xlim(0, 600_000)
plt.ylim(0, 600_000)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.grid()
plt.plot([(0, 0), (1_000_000, 1_000_000)], [(0, 0), (1_000_000, 1_000_000)])
plt.show()

score = train_model(depth, learning_rate/100, n_estimators, model_type='elastic')

# train with the best parameters
model_elastic = ElasticNet(alpha=0.0005, l1_ratio=.5)

score = rmsle_cv(model_elastic)
print("model_elastic score: depth={:d} lr={:.2f} est={:d} -> mean:{:.5f} std:{:.4f}".format(depth, learning_rate, n_estimators, score.mean(), score.std()))
model_elastic.fit(X_train, y_train)
y_pred = model_elastic.predict(X_train)
print('RMSLE model_elastic = ', rmsle(y_train, y_pred))

plt.figure(figsize=(8, 8))
y_pred = model_elastic.predict(X_train)
plt.scatter(np.expm1(y_train), np.expm1(y_pred))
plt.xlim(0, 600_000)
plt.ylim(0, 600_000)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.grid()
plt.plot([(0, 0), (1_000_000, 1_000_000)], [(0, 0), (1_000_000, 1_000_000)])
plt.show()


result = []
for depth in range(3, 5):
    for learning_rate in range(2, 3, 1):
        for n_estimators in range(1200, 2000, 200):
            score = train_model(depth, learning_rate/100, n_estimators, model_type='xgb')
            result.append([depth, learning_rate/100, n_estimators, score.mean(), score.std()])

# check results
result = pd.DataFrame(result, columns=['depth', 'learning_rate', 'n_estimators', 'score_mean', 'score_std'])
result.describe()

best = np.argmin(result['score_mean'].values)
print('Best params = \n', result.iloc[best])

# train with the best parameters
depth = int(result.iloc[best]['depth'])
learning_rate = result.iloc[best]['learning_rate']
n_estimators = int(result.iloc[best]['n_estimators'])

model_xgb = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=depth, 
                         n_estimators=n_estimators,
                         silent=1,
                         random_state =7, nthread = -1)

score = rmsle_cv(model_xgb)
print("Xgboost score: depth={:d} lr={:.2f} est={:d} -> mean:{:.5f} std:{:.4f}".format(depth, learning_rate, n_estimators, score.mean(), score.std()))
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_train)
print('RMSLE XGB = ', rmsle(y_train, y_pred))

plt.figure(figsize=(8, 8))
y_pred = model_xgb.predict(X_train)
plt.scatter(np.expm1(y_train), np.expm1(y_pred))
plt.xlim(0, 600_000)
plt.ylim(0, 600_000)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.grid()
plt.plot([(0, 0), (1_000_000, 1_000_000)], [(0, 0), (1_000_000, 1_000_000)])
plt.show()

result = []
for depth in range(3, 5):
    for learning_rate in range(2, 4, 1):
        for n_estimators in range(1200, 2000, 200):
            score = train_model(depth, learning_rate/100, n_estimators, model_type='lgb')
            result.append([depth, learning_rate/100, n_estimators, score.mean(), score.std()])

# check results
result = pd.DataFrame(result, columns=['depth', 'learning_rate', 'n_estimators', 'score_mean', 'score_std'])
result.describe()

best = np.argmin(result['score_mean'].values)
print('Best params = \n', result.iloc[best])

# train with the best parameters
depth = int(result.iloc[best]['depth'])
learning_rate = result.iloc[best]['learning_rate']
n_estimators = int(result.iloc[best]['n_estimators'])

model_lgb = lgb.LGBMRegressor(learning_rate=learning_rate, max_depth=depth, 
                         n_estimators=n_estimators,
                         silent=1,
                         random_state =7, nthread = -1)


score = rmsle_cv(model_lgb)
print("lgb score: depth={:d} lr={:.2f} est={:d} -> mean:{:.5f} std:{:.4f}".format(depth, learning_rate, n_estimators, score.mean(), score.std()))

model_lgb.fit(X_train, y_train.flatten())
y_pred = model_lgb.predict(X_train)

print('RMSLE LGB = ', rmsle(y_train, y_pred))


plt.figure(figsize=(8, 8))
plt.scatter(np.expm1(y_train), np.expm1(y_pred))
plt.xlim(0, 600_000)
plt.ylim(0, 600_000)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (1_000_000, 1_000_000)], [(0, 0), (1_000_000, 1_000_000)])
plt.show()


avg_predicts = np.column_stack([
    model_xgb.predict(X_train),
    model_lgb.predict(X_train),
    model_lasso.predict(X_train),
    model_elastic.predict(X_train)
])

avg_predict = [np.mean(p) for p in avg_predicts]
y_pred = avg_predict

plt.figure(figsize=(8, 8))
plt.scatter(np.expm1(y_train), np.expm1(y_pred))
plt.xlim(0, 600_000)
plt.ylim(0, 600_000)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (1_000_000, 1_000_000)], [(0, 0), (1_000_000, 1_000_000)])
plt.show()

print('RMSLE LGB = ', rmsle(y_train, y_pred))


# Submission

avg_predicts = np.column_stack([
    model_xgb.predict(X_test),
    model_lgb.predict(X_test),
    model_lasso.predict(X_test),
    model_elastic.predict(X_test)
])


avg_predict = [np.mean(p) for p in avg_predicts]

subm_predict = np.expm1(avg_predict)

dsubm_predict = pd.DataFrame(subm_predict)
dsubm_predict['Id'] = test.values[:,0]
dsubm_predict.columns = ['SalePrice', 'Id']

dsubm_predict.to_csv('submission.csv', index=False)
dsubm_predict[0:10]
