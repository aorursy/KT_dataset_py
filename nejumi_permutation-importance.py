import numpy as np

import pandas as pd

from pandas import DataFrame

pd.options.display.max_columns = 200



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.inspection import permutation_importance # 最新のscikit-learnではPermutation Importanceが追加されている。



from tqdm import tqdm_notebook as tqdm
# csvファイルを読み込みます。

df_train = pd.read_csv('../input/machine-learning-homework/train.csv', index_col=0)

df_test = pd.read_csv('../input/machine-learning-homework/test.csv', index_col=0)
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
X_concat = pd.concat([X_train, X_test])



for col in X_concat.columns:

    if (X_concat[col].dtype == 'object'):

        le = LabelEncoder()

        X_concat[col] = le.fit_transform(X_concat[col].fillna('NaN')) # カテゴリの欠損をNaNという値で埋めておく

        

X_train = X_concat.iloc[:len(X_train)].fillna(-99999) # 数値の欠損を-99999で埋めておく

X_test = X_concat.iloc[len(X_train):].fillna(-99999)
# 埼玉的な要素の高い地域を検定パートにする。仮想埼玉！

val_area = ['Nerima Ward', 'Kita Ward', 'Itabashi Ward']



X_val = X_train[df_train.Municipality.isin(val_area)]

y_val = y_train[df_train.Municipality.isin(val_area)]



X_train_ = X_train[~df_train.Municipality.isin(val_area)]

y_train_ = y_train[~df_train.Municipality.isin(val_area)]
# モデルを学習データに対してfitして、検定データに対してPermutaiton Importanceを計算する。

model = HistGradientBoostingRegressor(learning_rate=0.05, random_state=71, max_iter=500)

model.fit(X_train_, np.log1p(y_train_))

result = permutation_importance(model, X_val, np.log1p(y_val), n_repeats=10,

                                random_state=71)



perm_sorted_idx = result.importances_mean.argsort()
# Permutaiton Importanceを可視化してみる

num_features = len(X_val.columns)



plt.figure(figsize=[8,15])

plt.title('Permutation Importance')

plt.barh(range(num_features), result['importances_mean'][perm_sorted_idx], xerr=result['importances_std'][perm_sorted_idx])

plt.yticks(range(num_features), X_val.columns[perm_sorted_idx])

plt.show()