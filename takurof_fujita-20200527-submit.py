import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import StratifiedKFold, KFold



from lightgbm import LGBMRegressor

import lightgbm as lgb



from tqdm import tqdm_notebook as tqdm



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss



pd.options.display.float_format = '{:.2f}'.format
df_train = pd.read_csv('../input/exam-for-students20200527/train.csv', index_col=0)

df_test = pd.read_csv('../input/exam-for-students20200527/test.csv', index_col=0)
df_test.head(30)
df_train.describe()
df_train = df_train[(df_train['TradePrice'] > 200000) & (df_train['TradePrice'] < 500000000)]
#df_train = df_train.sort_values('TradePrice', ascending=False)
y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'], axis=1)



X_test = df_test
X_train = X_train.drop(['Prefecture','Municipality','DistrictName','NearestStation','TimeToNearestStation','MaxTimeToNearestStation'],axis=1)

X_test = X_test.drop(['Prefecture','Municipality','DistrictName','NearestStation','TimeToNearestStation','MaxTimeToNearestStation'],axis=1)
cats = []

nums = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

    elif col == 'Year' or col == 'Quarter':

        cats.append(col)

    elif col != 'AreaIsGreaterFlag' and col != 'FrontageIsGreaterFlag' and col != 'TotalFloorAreaIsGreaterFlag':

        nums.append(col)
nums
# ##欠損値処理

for num in nums:

#     X_train[num + '_NA'] = 1

#     X_train[num + '_NA'] = X_train[num + '_NA'].where(X_train[num].isnull(), 0)

#     X_test[num + '_NA'] = 1

#     X_test[num + '_NA'] = X_test[num + '_NA'].where(X_test[num].isnull(), 0)   

    if num == 'MinTimeToNearestStation' or 'BuildingYear':

        X_train[num].fillna(0, inplace=True)

        X_test[num].fillna(0, inplace=True)

    else:

        X_train[num].fillna(X_train[num].median(), inplace=True)

        X_test[num].fillna(X_train[num].median(), inplace=True)
# #ランクガウス

# from sklearn.preprocessing import quantile_transform



# X_all = pd.concat([X_train, X_test], axis=0)

# X_all[nums] = quantile_transform(X_all[nums], n_quantiles=100, random_state=0, output_distribution='normal')



# X_train = X_all.iloc[:X_train.shape[0], :]

# X_test = X_all.iloc[X_train.shape[0]:, :]
for col in cats:

    X_train[col + '_NA'] = 1

    X_train[col + '_NA'] = X_train[col + '_NA'].where(X_train[col].isnull(), 0)

    X_test[col + '_NA'] = 1

    X_test[col + '_NA'] = X_test[col + '_NA'].where(X_test[col].isnull(), 0)   

    X_train[col].fillna("NULL", inplace=True)

    X_test[col].fillna("NULL", inplace=True)

    

#     summary = X_train[col].value_counts()

#     X_train[col] = X_train[col].map(summary)

#     X_test[col] = X_test[col].map(summary)
target = 'TradePrice'



for col in cats:

    X_temp = pd.concat([X_train, y_train], axis=1)



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    enc_test = X_test[col].map(summary) 

    

    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

    

    X_train[col] = enc_train

    X_test[col] = enc_test
#カテゴリ欠損値処理

for col in cats:

    X_train[col].fillna(0, inplace=True)

    X_test[col].fillna(0, inplace=True)
#層化抽出

scores = []



y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]

    

    clf = LGBMRegressor() 

    

    clf.fit(X_train_, np.log1p(y_train_))

    y_pred = np.expm1(clf.predict(X_val))

    score = mean_squared_log_error(y_val, y_pred)**0.5

    scores.append(score)

       

    y_pred = np.expm1(clf.predict(X_test))

    

    y_pred_test += y_pred
print(score)
y_pred_test /= 5
submission = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv', index_col=0)



submission.TradePrice = y_pred_test

submission.to_csv('submission.csv')