# ライブラリのインポート
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.linear_model import LinearRegression

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# 文字列をラベル化した数値に変換する為のライブラリをインポート
from sklearn.preprocessing import LabelEncoder

#IDの一時保存
test_ID = test['PassengerId']


###########################
### ラベル化
###########################
# データタイプがobjectの列の値をラベル化した数値に変換
lbl = LabelEncoder()

lbl.fit(list(train['Sex'].values) + list(test['Sex'].values))
train['Sex'] = lbl.transform(list(train['Sex'].values))
test['Sex'] = lbl.transform(list(test['Sex'].values))

lbl.fit(list(train['Ticket'].values) + list(test['Ticket'].values))
train['Ticket'] = lbl.transform(list(train['Ticket'].values))
test['Ticket'] = lbl.transform(list(test['Ticket'].values))

lbl.fit(list(train['Cabin'].values) + list(test['Cabin'].values))
train['Cabin'] = lbl.transform(list(train['Cabin'].values))
test['Cabin'] = lbl.transform(list(test['Cabin'].values))

lbl.fit(list(train['Embarked'].values) + list(test['Embarked'].values))
train['Embarked'] = lbl.transform(list(train['Embarked'].values))
test['Embarked'] = lbl.transform(list(test['Embarked'].values))


###########################
### FareのNaNを埋める
###########################
# NaNの処理
# 学習用にデータを分ける
y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)

# 欠損データの処理(Fareの欠損処理)
Xmat = pd.concat([X_train, test])
Xmat['Fare'] = Xmat['Fare'].fillna(Xmat['Fare'].mean())

###########################
### 下処理
###########################
# 年齢の処理
X_train_age = Xmat.query('Age >= 0')
X_test_age = Xmat[Xmat['Age'].isnull()]

# IDの一時保存
train_age_ID = X_train_age['PassengerId']
test_age_ID = X_test_age['PassengerId']

# Ageの一時保存
y_train_age_store = X_train_age['Age']
y_train_age = np.log(y_train_age_store)

X_test_age = X_test_age.drop(['PassengerId', 'Age', 'Name'], axis=1)
X_train_age = X_train_age.drop(['PassengerId', 'Age', 'Name'], axis=1)
X_train_age_std = (X_train_age - X_train_age.mean()) / X_train_age.std()
X_test_age_std = (X_test_age - X_test_age.mean()) / X_test_age.std()


###########################
### 予測
###########################
# 線形回帰
slr = LinearRegression()
slr.fit(X_train_age_std,y_train_age)
y_test_age_slrpred = np.exp(slr.predict(X_test_age_std))

#boost
mod = xgb.XGBRegressor()
mod.fit(X_train_age_std,y_train_age)
y_test_age_xgbpred = np.exp(mod.predict(X_test_age_std))

y_test_age_pred = 0.7*y_test_age_slrpred + 0.3*y_test_age_xgbpred 


###########################
### 後処理
###########################
X_train_age = pd.concat([train_age_ID, y_train_age_store, X_train_age], axis=1)

X_test_age = pd.concat([test_age_ID, X_test_age], axis=1).reset_index(drop=True)
y_test_age_pred = pd.DataFrame(y_test_age_pred).rename(columns={0 : 'Age'})
X_test_age = pd.concat([X_test_age, y_test_age_pred], axis=1)

Xmat = pd.concat([X_train_age, X_test_age])
Xmat = Xmat.sort_values(by='PassengerId')

X_train = Xmat.iloc[:train.shape[0],:]
X_test = Xmat.iloc[train.shape[0]:,:]

###########################
### 予測
###########################
#標準化
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# ロジスティック回帰
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_test_pred = clf.predict(X_test)


###########################
### 出力
###########################
# submission
submission = pd.DataFrame({
    "PassengerId": test_ID,
    "Survived": y_test_pred
})
submission.to_csv('submission.csv', index=False)
