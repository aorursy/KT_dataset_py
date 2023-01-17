# ライブラリのインポート
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
%matplotlib inline

# 他のやつ
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# データの読み込み
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# IDの一時保存
train_ID = train['PassengerId']
test_ID = test['PassengerId']

# 下処理関数
def preprocess(df):
    # NameからMr/Mrs/Miss/Masterを取り出し
    df["Mr"] = df["Name"].apply(lambda x: x.count("Mr."))
    df["Mrs"] = df["Name"].apply(lambda x: x.count("Mrs."))
    df["Miss"] = df["Name"].apply(lambda x: x.count("Miss."))
    df["Master"] = df["Name"].apply(lambda x: x.count("Master."))
    # 家族構成
    df['NumFamily'] = df['SibSp'] + df['Parch']+1
    df['IsAlone'] = [(1 if x == 1 else 0) for x in df.NumFamily]
    # いらないカラムの削除
    df = df.drop(['PassengerId','NumFamily','Name', 'Sex'], axis=1)
    return df

train = preprocess(train)
test = preprocess(test)

train.dtypes
# 文字列をラベル化した数値に変換する為のライブラリをインポート
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()

lbl.fit(list(train['Ticket'].values) + list(test['Ticket'].values))
train['Ticket'] = lbl.transform(list(train['Ticket'].values))
test['Ticket'] = lbl.transform(list(test['Ticket'].values))

lbl.fit(list(train['Cabin'].values) + list(test['Cabin'].values))
train['Cabin'] = lbl.transform(list(train['Cabin'].values))
test['Cabin'] = lbl.transform(list(test['Cabin'].values))

lbl.fit(list(train['Embarked'].values) + list(test['Embarked'].values))
train['Embarked'] = lbl.transform(list(train['Embarked'].values))
test['Embarked'] = lbl.transform(list(test['Embarked'].values))

train.dtypes
train_nan = train.isnull().sum()
train_nan = train_nan[train_nan > 0]
train_nan
test_nan = test.isnull().sum()
test_nan = test_nan[test_nan > 0]
test_nan
# 学習用にデータを分ける
y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)

# 欠損データの処理
Xmat = pd.concat([X_train, test])
Xmat = Xmat.fillna(Xmat.median())

# びにんぐ
Xmat['FareBin'] = pd.cut(Xmat.Fare, 5, labels=False)
Xmat['AgeBin'] = pd.cut(Xmat.Age, 5, labels=False)
Xmat = Xmat.drop(['Fare', 'Age'], axis=1)
Xmat
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
# 正規化
X_train = X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
X_test = X_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# ロジスティック回帰
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_test_pred_log = clf.predict(X_test)

# 違うやつ達
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_test_pred_dec = clf.predict(X_test)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_test_pred_ran = clf.predict(X_test)

clf = MLPClassifier()
clf.fit(X_train,y_train)
y_test_pred_mlp = clf.predict(X_test)

clf = GradientBoostingClassifier()
clf.fit(X_train,y_train)
y_test_pred_gra = clf.predict(X_test)

clf = SVC()
clf.fit(X_train,y_train)
y_test_pred_svc = clf.predict(X_test)

#y_test_pred_sum = (y_test_pred_log + y_test_pred_dec + y_test_pred_ran + y_test_pred_mlp + y_test_pred_gra)
#y_test_pred = list(map(lambda x: 1 if 3 <= x else 0, y_test_pred_sum))
#y_test_pred_sum = (y_test_pred_log + y_test_pred_ran + y_test_pred_gra)
#y_test_pred = list(map(lambda x: 1 if 2 <= x else 0, y_test_pred_sum))

# submission
submission = pd.DataFrame({
    "PassengerId": test_ID,
    "Survived": y_test_pred_log
})
submission.to_csv('submission.csv', index=False)
submission
