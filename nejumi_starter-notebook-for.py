# まずは必要なライブラリをimportしよう。

import numpy as np

import pandas as pd

pd.options.display.max_columns = 200



import lightgbm as lgb

from lightgbm import LGBMRegressor



from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error, mean_squared_log_error



from tqdm import tqdm_notebook as tqdm
# csvファイルを読み込みます。

df_train = pd.read_csv('../input/machine-learning-homework/train.csv', index_col=0)

df_test = pd.read_csv('../input/machine-learning-homework/test.csv', index_col=0)
# 行数と列数を確認する。

df_train.shape, df_test.shape
# 先頭5行を確認してみよう。

df_train.head()
# 先頭5行を確認してみよう。

df_test.head()
# 都道府県別の出現頻度を確認してみよう。

df_train.Prefecture.value_counts()
df_test.Prefecture.value_counts()
# データを特徴量とターゲットに分割しておきます。

y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'], axis=1)



X_test = df_test.copy()
# 最寄り駅までの所要時間がカテゴリになっているので、数値に変換してみる。

replace_dict = {

    '30-60minutes':30, '1H-1H30':60, '2H-':120, '1H30-2H':90}

X_train.TimeToNearestStation.replace(replace_dict, inplace=True)

X_test.TimeToNearestStation.replace(replace_dict, inplace=True)



X_train.TimeToNearestStation = X_train.TimeToNearestStation.astype(float)

X_test.TimeToNearestStation = X_test.TimeToNearestStation.astype(float)
# dtypeがobjectのものを雑にLabelEncodingしておく。同時に欠損を埋めておく。

X_concat = pd.concat([X_train, X_test])



for col in X_concat.columns:

    if (X_concat[col].dtype == 'object'):

        le = LabelEncoder()

        X_concat[col] = le.fit_transform(X_concat[col].fillna('NaN')) # カテゴリの欠損をNaNという値で埋めておく

        

X_train = X_concat[X_concat.index.isin(X_train.index)].fillna(-99999) # 数値の欠損を-99999で埋めておく

X_test = X_concat[~X_concat.index.isin(X_train.index)].fillna(-99999)
# 処理後についても先頭5行をみてみよう。

X_train.head()
X_test.head()
groups = X_train.Prefecture.values

X_train.drop(['Prefecture'], axis=1, inplace=True)

X_test.drop(['Prefecture'], axis=1, inplace=True)
# 都道府県で区切って交差検定を行い、予測精度を見積もる。

# 同時にテストデータに対してcv averagingを行い、予測値を得る。

n_fold = 5

cv = GroupKFold(n_splits=n_fold)



y_pred_train = np.zeros(len(X_train))

y_pred_test = np.zeros(len(X_test))

scores = []



for i, (train_index, val_index) in enumerate(cv.split(X_train, y_train, groups)):

    X_train_, y_train_ = X_train.iloc[train_index], y_train.iloc[train_index]

    X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]

    

    model = HistGradientBoostingRegressor(learning_rate=0.05, random_state=71, max_iter=500)

    model.fit(X_train_, np.log1p(y_train_))

    y_pred_val = np.expm1(model.predict(X_val))

    y_pred_test += np.expm1(model.predict(X_test))/n_fold

    

    y_pred_train[val_index] = y_pred_val

    score = mean_squared_log_error(y_val, y_pred_val)**0.5

    scores.append(score)

    

    print("Fold%d RMSLE: %f"%(i, score))

    

print("Overall RMSLE: %f±%f"%(np.mean(scores), np.std(scores)))
df_sub = pd.read_csv('../input/machine-learning-homework/sample_submission.csv', index_col=0)

df_sub.TradePrice = y_pred_test

df_sub.to_csv('submission.csv')