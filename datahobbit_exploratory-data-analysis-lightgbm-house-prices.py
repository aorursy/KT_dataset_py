import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("Train Shape: " + str(train.shape))
print("Test Shape: " + str(test.shape))
print(list(train))
unique_vals_per_col = train.T.apply(lambda x: x.nunique(), axis=1)
print(unique_vals_per_col.head(5))
import matplotlib.pyplot as plt

# a scatter plot comparing num_children and num_pets
train.plot(kind='scatter',x='GrLivArea',y='SalePrice',color='green')
plt.show()
plt.clf()
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train.groupby('Neighborhood')['SalePrice'].mean().plot(kind='bar')
plt.show()
plt.clf()
train.groupby('Neighborhood')['SalePrice'].count().plot(kind='bar')
plt.show()
plt.clf()
train.groupby('HouseStyle')['SalePrice'].mean().plot(kind='bar')
plt.show()
plt.clf()
train.groupby('KitchenQual')['SalePrice'].mean().plot(kind='bar')
plt.show()
plt.clf()
y = train['SalePrice']
train.drop("SalePrice", axis = 1, inplace = True)
from sklearn import preprocessing
#from sklearn.preprocessing import LabelEncoder
import datetime

ntrain = train.shape[0]

all_data = pd.concat((train, test)).reset_index(drop=True)
print("all_data size is : {}".format(all_data.shape))

def ordinal_encode(df, col, order_list):
    df[col] = df[col].astype('category', ordered=True, categories=order_list).cat.codes
    return df

def label_encode(df, col):
    for c in col:
        #print(str(c))
        encoder = preprocessing.LabelEncoder()
        df[c] = encoder.fit_transform(df[c].astype(str))
    return df 

def split_all_data(all_data, ntrain):
    print(('Split all_data back to train and test: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
    train_df = all_data[:ntrain]
    test_df = all_data[ntrain:]
    return train_df, test_df

"""
NOW START ENCODING 1. ORDINALS
"""
print(('Ordinal Encoding: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
order_list = ['Ex', 'Gd', 'TA', 'Fa', 'Po'] #This applies to a few different columns
cols = ['KitchenQual', 'ExterQual', 'ExterCond', 'HeatingQC']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)

order_list = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'] #This applies to a few different columns
cols = ['BsmtQual', 'BsmtCond']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)

order_list = ['Gd', 'Av', 'Mn', 'No', 'NA']
cols = ['BsmtExposure', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)
    
order_list = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
cols = ['Functional']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)
    
order_list = ['Fin', 'RFn', 'Unf', 'NA']
cols = ['GarageFinish']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)
    
order_list = ['Ex', 'Gd', 'TA', 'Fa', 'NA'] 
cols = ['PoolQC']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)

"""
ENCODE 2. NON-ORDINAL LABELS
"""
print(('Label Encoding: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
cols_to_label_encode = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 
                       'Electrical', 'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
all_data = label_encode(all_data, cols_to_label_encode)

train, test = split_all_data(all_data, ntrain)

print("Train Shape: " + str(train.shape))
print("Test Shape: " + str(test.shape))

    
def score_transformer(y):
    y = np.log(y)
    
    return y

y = score_transformer(y)
#Quicker to calculate once for train and test for values where this is appropriate
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data['fe.sum.GrLivArea_BsmtFinSF1_BsmtFinSF2'] = all_data['GrLivArea'] + all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] 
all_data['fe.sum.OverallQual_Overall_Cond'] = all_data['OverallQual'] + all_data['OverallCond']
all_data['fe.mult.OverallQual_Overall_Cond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['fe.sum.KitchenQual_ExterQual'] = all_data['KitchenQual'] + all_data['ExterQual']
all_data['fe.mult.OverallQual_Overall_Cond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['fe.ratio.1stFlrSF_2ndFlrSF'] = all_data['1stFlrSF'] / all_data['2ndFlrSF']
all_data['fe.ratio.BedroomAbvGr_GrLivArea'] = all_data['BedroomAbvGr'] / all_data['GrLivArea']


train_features = list(all_data)
#Id should be removed for modelling
train_features = [e for e in train_features if e not in ('ExterQual', 'Condition2', 'GarageCond', 'Street', 'Alley', 'PoolArea', 'PoolQC', 'Utilities', 
                                                         'GarageQual', 'MiscVal', 'MiscFeature')]

train, test = split_all_data(all_data, ntrain)

train_features.remove('Id')

#remove highly correlated variables
#train_features.remove('GarageFinish')
#train_features.remove('GarageArea')
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

nfolds=5
kf = KFold(n_splits=nfolds, shuffle=True, random_state=37) #33 originally
y_valid_pred = 0*y
y_valid_pred_cat = 0*y
fold_scores = [0] * nfolds
fold_scores_cat = [0] * nfolds

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
sub_reg_preds_cat = np.zeros(test.shape[0])


for fold_, (train_index, val_index) in enumerate(kf.split(train, y)):
    trn_x, trn_y = train[train_features].iloc[train_index], y.iloc[train_index]
    val_x, val_y = train[train_features].iloc[val_index], y.iloc[val_index]
    
    reg = LGBMRegressor(
        num_leaves=15,
        max_depth=3,
        min_child_weight=50,
        learning_rate=0.04,
        n_estimators=1000,
        #min_split_gain=0.01,
        #gamma=100,
        reg_alpha=0.01,
        reg_lambda=5,
        subsample=1,
        colsample_bytree=0.21,
        random_state=2
    )
    reg.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=20,
        verbose=100,
        eval_metric='rmse'
    )    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')

    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    y_valid_pred.iloc[val_index] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    y_valid_pred[y_valid_pred < 0] = 0
    fold_score = reg.best_score_['valid_0']['rmse']
    fold_scores[fold_] = fold_score
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += _preds / nfolds
    
print("LightGBM CV RMSE: " + str(mean_squared_error(y, y_valid_pred) ** .5))
print("LightGBM CV standard deviation: " + str(np.std(fold_scores)))
   

import seaborn as sns
import warnings
#cat_rgr.fit(X_train, y_train, eval_set=(X_valid, y_valid), logging_level='Verbose', plot=False)
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(10, 14))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
sub_reg_preds = np.exp(sub_reg_preds)
test.is_copy = False #disable the SettingWithCopyWarning
test.loc[:,'SalePrice'] = sub_reg_preds
test[['Id', 'SalePrice']].to_csv("tutorial_sub.csv", float_format='%.8f', index=False)

