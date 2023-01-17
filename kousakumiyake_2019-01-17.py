import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
df_train = pd.read_csv('../input/train_V2.csv')
df_test  = pd.read_csv('../input/test_V2.csv')

df_train_lgbm = df_train.copy()
df_test_lgbm = df_test.copy()

df_train_lgbm = df_train_lgbm.drop(["Id","groupId", "matchId", "matchType"], axis=1)
df_test_lgbm = df_test_lgbm.drop(["Id","groupId", "matchId", "matchType"], axis=1)
df_train_lgbm = df_train_lgbm.dropna(axis=0) #欠損値補完
fcol = [c for c in df_train_lgbm if c not in ["winPlacePerc"]]

y = df_train_lgbm["winPlacePerc"].values
X = df_train_lgbm[fcol].values
#ホールドアウト
X_train, X_test, y_train, y_test = train_test_split(X, y)
#データセット作成
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
lgbm_params = {
        # 回帰問題
        'objective': 'regression',
        # RMSE (平均二乗誤差平方根) の最小化を目指す
        'metric': 'mean_absolute_error',
    }

# 上記のパラメータでモデルを学習する
model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

# テストデータを予測する
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# RMSE を計算する
mae = mean_absolute_error(y_test, y_pred)
print(mae)
tests_x = df_test_lgbm.values
result = model.predict(tests_x, num_iteration=model.best_iteration)
df1 = df_test["Id"]
df2 = pd.DataFrame(data=result, columns=["winPlacePerc"])
for c in range(len(df2)):
    if df2.iat[c,0] < 0:
        df2.iat[c,0] = 0
    elif df2.iat[c,0] >1:
        df2.iat[c,0] = 1
submission_file = pd.concat([df1,df2], axis = 1)
submission_file.to_csv('submission_file.csv',index=False)