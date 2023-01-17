# No.1
# ライブラリのインポート
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Jupyter Notebookの中でインライン表示する場合の設定（これが無いと別ウィンドウでグラフが開く）
%matplotlib inline

# データの読み込み
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#　No.6
# 文字列をラベル化した数値に変換する為のライブラリをインポート
from sklearn.preprocessing import LabelEncoder

# データタイプがobjectの列の値をラベル化した数値に変換
for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
# No.10
# keep ID for submission
train_ID = train['Id']
test_ID = test['Id']

# split data for training
y_train = train['SalePrice']
X_train = train.drop(['Id','SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

# dealing with missing data
Xmat = pd.concat([X_train, X_test])
# 欠損値の多いカラムを削除
Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)
# 欠損値の少ないカラムのNaNは中央値(median)で埋める
Xmat = Xmat.fillna(Xmat.median())

# check whether there are still nan
Xmat_nan = Xmat.isnull().sum()
Xmat_nan = Xmat_nan[Xmat_nan > 0]
Xmat_nan
# No.11
Xmat["TotalSF"] = Xmat["TotalBsmtSF"] + Xmat["1stFlrSF"] + Xmat["2ndFlrSF"]
# No.15
# 対数計算を実施
# 数字のばらつき、偏りを小さくする
y_train = np.log(y_train)

sns.distplot(y_train)
plt.show()
# No.16
# trainデータとtestデータを含んでいるXmatを、再度trainデータとtestデータに分割
X_train = Xmat.iloc[:train.shape[0],:]
X_test = Xmat.iloc[train.shape[0]:,:]

# ランダムフォレストをインポート
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print("Training done using Random Forest")

# np.argsort()はソート結果の配列のインデックスを返す。引数の頭に"-"をつけると降順。
# つまり"-rf.feature_importances_"を引数にする事で重要度の高い順にソートした配列のインデックスを返す。
ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()
# No.17
# use the top 30 features only
X_train = X_train.iloc[:,ranking[:30]]
X_test = X_test.iloc[:,ranking[:30]]

# interaction between the top 2
X_train["Interaction"] = X_train["TotalSF"] * X_train["OverallQual"]
X_test["Interaction"] = X_test["TotalSF"] * X_test["OverallQual"]
# No.18
# z-scoreにて標準化
# (値 - 平均) / 標準偏差
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()
# No.19
# relation to the target
fig = plt.figure(figsize=(12,7))
for i in np.arange(30):
    ax = fig.add_subplot(5,6,i+1)
    sns.regplot(x=X_train.iloc[:,i], y=y_train)

plt.tight_layout()
plt.show()
# No.20
# outlier deletion
Xmat = X_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(index = Xmat[(Xmat['TotalSF'] > 5) & (Xmat['SalePrice'] < 12.5)].index)
Xmat = Xmat.drop(index = Xmat[(Xmat['GrLivArea'] > 5) & (Xmat['SalePrice'] < 13)].index)

# recover
y_train = Xmat['SalePrice']
X_train = Xmat.drop(['SalePrice'], axis=1)
intergrate_result = pd.DataFrame(index = [x for x in test.index], columns = ['liner', 'lasso', 'ridge', 'xgb', 'EN', 'LGB'])
train_result = pd.DataFrame(index = [x for x in X_train.index], columns = ['liner', 'lasso', 'ridge', 'xgb', 'EN', 'LGB'])
# アルゴリズムに線形回帰(Linear Regression)を採用
slr = LinearRegression()

# fit関数で学習開始
slr.fit(X_train,y_train)
train_result['liner'] = np.exp(slr.predict(X_train))
intergrate_result['liner'] = np.exp(slr.predict(X_test))
submission = np.exp(slr.predict(X_test))
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = submission
sub.to_csv('submission.csv',index=False)
