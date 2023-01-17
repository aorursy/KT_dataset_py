# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

raw_train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_sub_df = pd.read_csv('../input/sample_submission.csv')
desc_lines = []
with open('../input/data_description.txt', 'r') as f:
    desc_lines = f.read().splitlines()

    
    
# sns.pairplot(raw_train_df[['SalePrice','GrLivArea']], height=3.5)
print(raw_train_df[(raw_train_df['OverallQual']<5) & (raw_train_df['SalePrice']>200000)].shape)
print(raw_train_df[(raw_train_df['GrLivArea']>4000) & (raw_train_df['SalePrice']<300000)].shape)
# As suggested by many participants, we remove several outliers
raw_train_df.drop(raw_train_df[(raw_train_df['OverallQual']<5) & (raw_train_df['SalePrice']>200000)].index, inplace=True)
raw_train_df.drop(raw_train_df[(raw_train_df['GrLivArea']>4000) & (raw_train_df['SalePrice']<300000)].index, inplace=True)

# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
raw_train_df['MSSubClass'] = raw_train_df['MSSubClass'].apply(str)
raw_train_df['YrSold'] = raw_train_df['YrSold'].astype(str)
raw_train_df['MoSold'] = raw_train_df['MoSold'].astype(str)
print('\n'.join(desc_lines))

target_col = 'SalePrice'
# TODO: handle nan in real valued columns
skip_cols = ['Id']#,'GarageYrBlt', 'MasVnrArea', 'LotFrontage']


raw_categorical_features = {
    'Utilities': {
        'AllPub': 4,
        'NoSewr': 3,
        'NoSeWa': 2,
        'ELO': 1
    },
    'ExterQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1
    },
    'ExterCond': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1
    },
    'BsmtQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': float('nan')
    },
    'BsmtCond': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': float('nan')
    },
    'BsmtExposure': {
        'Gd': 5,
        'Av': 4,
        'Mn': 3,
        'No': 2,
        'NA': float('nan'),
    },
    'BsmtFinType1': {
        'GLQ': 5,
        'ALQ': 4,
        'BLQ': 3,
        'Rec': 2,
        'LwQ': 1,
        'Unf': float('nan'),
        'NA': float('nan')
    },
    'HeatingQC': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
    },
    'Electrical': {
        'SBrkr': 5,
        'FuseA': 4,
        'FuseF': 3,
        'FuseP': 2,
        'Mix': 1
    },
    'KitchenQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
    },
    'Functional': {
        'Typ': 8,
        'Min1': 7,
        'Min2': 6,
        'Mod': 5,
        'Maj1': 4,
        'Maj2': 3,
        'Sev': 2,
        'Sal': 1,
    },
    'FireplaceQu': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': float('nan')
    },
    'GarageFinish': {
        'Fin': 4,
        'RFn': 3,
        'Unf': 2,
        'NA': float('nan'),
    },
    'GarageQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': float('nan')
    },
    'GarageCond': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': float('nan')
    },
    'PavedDrive': {
        'Y': 3,
        'P': 2,
        'N': 1
    },
    'PoolQC': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'NA': float('nan')
    },
}

def get_skew_reduction_columns(train_df):
    """
    Since we are dealing with large numbers, taking log reduces the skew and therefore makes it more 
    like normal distribution
    """
    train_df = train_df[value_cols + [target_col]].fillna(train_df[value_cols].mean())
    raw_skew = train_df.apply(skew)
    raw_skew = raw_skew[raw_skew.abs() > 0.5]
    large_skew_cols = raw_skew.index.tolist()
    train_df = train_df[large_skew_cols]
    mask = train_df !=0
    train_df[mask] = train_df[mask].applymap(np.log)
    improved_skew = train_df.apply(skew)
    raw_skew = raw_skew[improved_skew.index]
    diff = (raw_skew - improved_skew)/raw_skew.abs()
#     NOTE: this 0.5 is pretty important hyperparameter. It should have been better if it was picked after
# k fold validation rather than manual selection.
    return diff[diff > 0.3].index.tolist()
    
def preprocess_categorical_features(df, categorical_features):
    """
    There are several features which are categorical and whose values have
    a predefined performance bias.
    Excellent basement will have higher sale price than poor basement.
    For such variables we need not go with one hot encoding and can use an integer.
    """
    df = df.copy()
    for column in categorical_features:
        df.loc[:, column] = df[column].map(categorical_features[column])
    #test
    columns = list(categorical_features.keys())
    assert not df[columns].applymap(lambda x: isinstance(x, str)).any().any()
    
    return df

def replace_nan_in_categorical_features(df, categorical_features):
    """
    It computes the average value of SalePrice when the category is absent. It then finds a category such that
    SalePrice is closest to it. nan is replaced by the value of the closest category found.
    """
    categorical_features=categorical_features.copy()
    for column in categorical_features:
        nan_categories = []
        for category, value in categorical_features[column].items():
            if np.isnan(value):
                nan_categories.append(category)
#         print('Attempting to replace NA values in', nan_categories)
        if not nan_categories:
            continue
        
        avg_price = df[df[column]=='NA'][target_col].mean()
        print('nan column count for ', column, ': ', df[df[column]=='NA'].shape, avg_price)
        category_means = df[df[column] != 'NA'].groupby(column)[target_col].mean()
        print(category_means)
        closest_category = (
            category_means - avg_price).abs().sort_values().index[0]
        
        for nan_category in nan_categories:
            categorical_features[column][nan_category] = categorical_features[column][closest_category]
        # test
    for column in categorical_features:
        for category, value in categorical_features[column].items():
            assert not np.isnan(value)
    
    return categorical_features

trainable_cols = list(set(raw_train_df.columns) - set(skip_cols + [target_col]))
categorical_cols = []
value_cols = []
for col in trainable_cols:
    val =raw_train_df[col].dropna().iloc[0]
    if isinstance(val, np.int64) or isinstance(val, float) or isinstance(val, np.int64):
        value_cols.append(col)
    elif isinstance(val, str):
        categorical_cols.append(col)
    else:
        raise Exception('Unhandled type', type(val))
    

import seaborn as sns
var = 'FullBath'
data = pd.concat([raw_train_df['SalePrice'], raw_train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis();
log_cols = get_skew_reduction_columns(raw_train_df)
print('Features which needs to be in log scale', log_cols)
assert target_col in log_cols
train_log_df = raw_train_df[log_cols].copy()
non_zero_mask_train = train_log_df > 0
train_log_df[non_zero_mask_train] = train_log_df[non_zero_mask_train].applymap(np.log)
raw_train_df.loc[:, log_cols] = train_log_df[log_cols].copy()
del train_log_df

test_log_cols = list(set(log_cols) - set([target_col]))
test_log_df = test_df[test_log_cols].copy()
non_zero_mask_test = test_log_df > 0
test_log_df[non_zero_mask_test]=test_log_df[non_zero_mask_test].applymap(np.log)
test_df.loc[:,test_log_cols] = test_log_df[test_log_cols].copy()
categorical_value_cols = list(raw_categorical_features.keys())
one_hot_enc_cols = list(set(categorical_cols) - set(categorical_value_cols))
value_cols = list(set(value_cols + list(categorical_value_cols)))

raw_train_df.loc[:,categorical_cols]= raw_train_df[categorical_cols].fillna('NA')
test_df.loc[:,categorical_cols]= test_df[categorical_cols].fillna('NA')

raw_train_df = raw_train_df[trainable_cols + [target_col]]
test_df = test_df[trainable_cols + ['Id']]
# filtr =np.random.rand(raw_train_df.shape[0]) > 0.2
# train_df = raw_train_df[filtr].copy()
# val_df = raw_train_df[~filtr].copy()
# print('Train size', train_df.shape, 'Validation size', val_df.shape)
# # this makes sure we do it on train_df and not the whole of data.
# replace_nan_in_categorical_features(train_df)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(raw_train_df[one_hot_enc_cols])


def get_X(df, encoder, categorical_features):
    df = preprocess_categorical_features(df, categorical_features)    
    encoded_df = encoder.transform(df[one_hot_enc_cols])
    X = np.c_[encoded_df, df[value_cols].fillna(method='ffill').fillna(
        method='bfill').values]
    return X



# X_train = get_X(train_df.copy(), enc)
# y_train = train_df[target_col].values

# X_val = get_X(val_df.copy(), enc)
# y_val = val_df[target_col].values


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


def _stat(pred, actual):
    return np.round(sqrt(mean_squared_error(pred, actual)), 2)


def get_stats(pred_train, pred_val):

    train_stat = _stat(pred_train, y_train)
    val_stat = _stat(pred_val, y_val)
    return (train_stat, val_stat)


from sklearn.model_selection import KFold
nF = 15
kf = KFold(n_splits=nF, random_state=955, shuffle=True)
kfold_train= []
kfold_val= []
ifold = 0
for train_index, test_index in kf.split(raw_train_df.values):
    print('Kfold ',ifold)
    ifold = ifold + 1
    train_df = raw_train_df.iloc[train_index].copy()   
    categorical_features_kfold = replace_nan_in_categorical_features(train_df,
                                                               raw_categorical_features)
    X_train = get_X(train_df, enc, categorical_features_kfold)
    X_val = get_X(raw_train_df.iloc[test_index].copy(), enc, categorical_features_kfold)
    y_train=raw_train_df.iloc[train_index][target_col].values
    y_val =raw_train_df.iloc[test_index][target_col].values
    print ('Train', X_train.shape)
    print ('Test', X_val.shape)

    train_perf = []
    val_perf = []
    alphas = [0,0.000001, 0.000005, 0.00001,0.00004, 0.00006,0.00008, .0001,.0002]# 0.0001,0.00001, 0]
    for alpha in alphas:
        lr = Lasso(alpha=alpha, normalize=True, copy_X =True)
        lr.fit(X_train, y_train)
    #     print(lr.coef_)
        prediction_train = lr.predict(X_train)
        prediction_val = lr.predict(X_val)

        (train_corr, val_corr) = get_stats(prediction_train, prediction_val)
        train_perf.append(train_corr)
        val_perf.append(val_corr)
    kfold_train.append(train_perf)
    kfold_val.append(val_perf)

train_perf = np.mean(np.array(kfold_train), axis=0)
val_perf = np.mean(np.array(kfold_val), axis=0)
plt.figure(figsize=(10,10))

ax = plt.gca()
ax.plot(alphas, train_perf, label='train')
ax.plot(alphas, val_perf, label='validation')
ax.legend()

best_alpha = 0.00005
models = []
prediction_actual_test = 0
ifold=0
for train_index, test_index in kf.split(raw_train_df.values):
    print('Kfold ',ifold)
    ifold = ifold + 1
    train_df = raw_train_df.iloc[train_index].copy()   
    categorical_features_kfold = replace_nan_in_categorical_features(train_df,
                                                               raw_categorical_features)
    X_train = get_X(train_df, enc, categorical_features_kfold)    
    y_train=raw_train_df.iloc[train_index][target_col].values


    lr = Lasso(alpha=best_alpha, normalize=True, copy_X=True)
    #lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    models.append(lr)
    X_actual_test = get_X(test_df, enc, categorical_features_kfold)
    
    prediction_actual_test = prediction_actual_test + lr.predict(X_actual_test)

prediction_actual_test = prediction_actual_test /len(models)

column_names = []
for col in one_hot_enc_cols:
    num_col_types = raw_train_df[col].unique()
    column_names += [col + '_{}'.format(i) for i in num_col_types]
column_names += value_cols
final_train_df = pd.DataFrame(X_train, columns=column_names)
final_train_df['Error'] = y_train - lr.predict(X_train)
final_train_df[final_train_df['Error'].abs() < 0.1].shape
final_train_df[final_train_df['Error'].abs() > 0.15].shape
max_err_df = pd.DataFrame(index=final_train_df.columns, columns=['category', 'error', 'count'])
max_err_df.index.name = 'features'
max_err_df.name='error'
for col in final_train_df:
    if col == 'Error':
        continue
    if len(final_train_df[col].unique())< 10:
        mn =final_train_df[[col,'Error']].groupby(col).agg(['mean', 'count']).abs()['Error']
        
        max_err_df.loc[col,:] = mn.reset_index().sort_values('mean').values[-1:]
# max_err_df.plot()
max_err_df[max_err_df['count']> 10].sort_values('error').tail(10)
max_err_df[max_err_df['count']> 10].sort_values('error').tail(10)
# tmp_df= max_err_df.dropna().sort_values().tail(15)
# fig, ax = plt.subplots()
# tmp_df.plot( ax=ax) 
# print(tmp_df.tail(10))
# del tmp_df
# data_df.head()
# sns.set()
# # 'RoofMatl',
# bad_cols= ['BedroomAbvGr','PoolQC','Condition2','OverallCond','Utilities','BsmtFullBath','RoofMatl','Functional',
#            'PoolArea',  target_col] 
# data_df = raw_train_df[bad_cols].copy()
# # bad_cat_cols = list(set(bad_cols).intersection(set(one_hot_enc_cols)))
# # data_df[bad_cat_cols]=data_df[bad_cat_cols].astype('category')
# # data_df.loc[:,bad_cat_cols] = data_df[bad_cat_cols].apply(lambda x: x.cat.codes)
# sns.pairplot(data_df, height=3.5,x_vars=target_col, y_vars=bad_cols[:-1])
# from xgboost import XGBRegressor
# xgb_train_perf = []
# xgb_val_perf = []
# max_depths=[1,2, 3,4,5]
# # lrs = [0.1, 0.15, 0.2, 0.3, 0.4,0.5]
# # reg_alpha (float (xgb's alpha)) – L1 regularization term on weights
# # reg_lambda (float (xgb's lambda)) – L2 regularization term on weights
# reg_alphas= [10, 50, 80, 100, 120, 150]
# for max_depth in max_depths:
#     for reg_alpha in reg_alphas:
#         my_model = XGBRegressor(learning_rate=0.15, max_depth=max_depth, 
#                                 reg_lambda=0, reg_alpha=0.1, n_estimators=reg_alpha)
#         # Add silent=True to avoid printing out updates with each cycle
#         my_model.fit(X_train, y_train, verbose=True)
#         xgb_val_predict = my_model.predict(X_val)
#         xgb_train_predict = my_model.predict(X_train)
#         (train_corr, val_corr) = get_stats(xgb_train_predict, xgb_val_predict)
#         xgb_train_perf.append(train_corr)
#         xgb_val_perf.append(val_corr)
# # from sklearn.preprocessing import Normalizer
# # final_train_df.loc[:,:] = Normalizer().fit_transform(final_train_df)
# # final_train_df =final_train_df.sort_values('Error')
# # # plt.figure(figsize=(20,10))
# # err_df=final_train_df.corr()['Error'].sort_values().dropna().drop('Error')
# xgb_val_df = pd.DataFrame(np.array(xgb_val_perf).reshape(len(max_depths),len(reg_alphas)),
#                           columns=reg_alphas, index=max_depths)

# xgb_train_df = pd.DataFrame(np.array(xgb_train_perf).reshape(len(max_depths),len(reg_alphas)),
#                           columns=reg_alphas, index=max_depths)

# for lr in reg_alphas:
#     pd.concat([xgb_val_df[lr].to_frame('val'), xgb_train_df[lr].to_frame('train')],axis=1).plot(title='Lr: ' + str(lr))
# my_model = XGBRegressor(learning_rate=0.15, max_depth=3, 
#                                 reg_lambda=0, reg_alpha=0.1, n_estimators=100)
# #lr = LinearRegression(normalize=True)
# my_model.fit(X_train, y_train)
# X_actual_test = get_X(test_df, enc)
# prediction_actual_test = my_model.predict(X_actual_test)

output_df = pd.DataFrame(
    np.c_[test_df.Id.values, prediction_actual_test],
    columns=['Id', 'SalePrice'])
output_df.loc[:, target_col] = output_df[target_col].apply(np.exp)
output_df.loc[:, 'Id'] = output_df.Id.apply(int)
output_df.to_csv('housing_prices_output.csv', index=False)

