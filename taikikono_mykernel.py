import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
%matplotlib inline
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.describe()
test_data.describe()
train_data.info()
train_data.corr()
train_data['Pclass'].value_counts()
train_data['Sex'].value_counts()
train_data['Embarked'].value_counts()
def missing_values_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val / len(df)
    missing_values_table = pd.concat([null_val, percent], axis=1)
    return missing_values_table.rename(columns= { 0: '欠損数', 1: '%'})
missing_values_table(train_data)
missing_values_table(test_data)
# 検証1
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# カテゴリデータの扱い　LabelEncoder
# スコア　0.75119
train1 = train_data.copy()
test1 = test_data.copy()

train1['Age'] = train1['Age'].fillna(train1['Age'].median())
train1['Embarked'] = train1['Embarked'].fillna('S')
train1 = train1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test1['Age'] = test1['Age'].fillna(test1['Age'].median())
test1['Fare'] = test1['Fare'].fillna(test1['Fare'].median())
test1 = test1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 検証1
train1['Sex'] = le.fit_transform(train1['Sex'])
train1['Embarked'] = le.fit_transform(train1['Embarked'])
test1['Sex'] = le.fit_transform(test1['Sex'])
test1['Embarked'] = le.fit_transform(test1['Embarked'])
# 検証1
missing_values_table(train1)
missing_values_table(test1)
# 検証1
train1_y = train1["Survived"]
train1_X = train1.drop('Survived', axis=1)
test1_X = test1

lr.fit(train1_X, train1_y)

Y_pred1 = lr.predict(test1_X)
print(round(lr.score(train1_X, train1_y) * 100, 2))
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred})
# submission.to_csv('submission.csv', index=False)
# 検証2
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# カテゴリデータの扱い　LabelEncoder
# ★相関のあるFareとPclassの積を新しい列に追加
# スコア　0.77033
train2 = train_data.copy()
test2 = test_data.copy()

train2['Age'] = train2['Age'].fillna(train2['Age'].median())
train2['Embarked'] = train2['Embarked'].fillna('S')
train2 = train2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test2['Age'] = test2['Age'].fillna(test2['Age'].median())
test2['Fare'] = test2['Fare'].fillna(test2['Fare'].median())
test2 = test2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 検証2
train2['Sex'] = le.fit_transform(train2['Sex'])
train2['Embarked'] = le.fit_transform(train2['Embarked'])
test2['Sex'] = le.fit_transform(test2['Sex'])
test2['Embarked'] = le.fit_transform(test2['Embarked'])
sns.heatmap(train2.corr())
train_data.groupby(['Sex', 'Survived'])['Survived'].count()
sns.countplot('Pclass',hue='Survived',data=train_data)
plt.hist(train_data['Fare'])
train2['Fare_Pclass'] = train2['Fare'] * train2['Pclass']
train2 = train2.drop(['Fare', 'Pclass'], axis=1)

test2['Fare_Pclass'] = test2['Fare'] * test2['Pclass']
test2 = test2.drop(['Fare', 'Pclass'], axis=1)
# 検証2
train2_y = train2["Survived"]
train2_X = train2.drop('Survived', axis=1)
test2_X = test2

lr.fit(train2_X, train2_y)

Y_pred2 = lr.predict(test2_X)
print(round(lr.score(train2_X, train2_y) * 100, 2))
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred2})
# submission.to_csv('submission.csv', index=False)
# 検証3
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# カテゴリデータの扱い　LabelEncoder
# ★ランダムフォレストで特徴量の重要度上位4つのみで学習した場合
# スコア　0.75598
train3 = train_data.copy()
test3 = test_data.copy()

train3['Age'] = train3['Age'].fillna(train2['Age'].median())
train3['Embarked'] = train3['Embarked'].fillna('S')
train3 = train3.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test3['Age'] = test3['Age'].fillna(test3['Age'].median())
test3['Fare'] = test3['Fare'].fillna(test3['Fare'].median())
test3 = test3.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 検証3
train3['Sex'] = le.fit_transform(train3['Sex'])
train3['Embarked'] = le.fit_transform(train3['Embarked'])
test3['Sex'] = le.fit_transform(test3['Sex'])
test3['Embarked'] = le.fit_transform(test3['Embarked'])
train3['Fare_Pclass'] = train3['Fare'] * train3['Pclass']
test3['Fare_Pclass'] = test3['Fare'] * test3['Pclass']
# 検証3
train3_y = train3["Survived"]
train3_X = train3.drop('Survived', axis=1)
test3_X = test3
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(train3_X, train3_y)
fti = rfc.feature_importances_

print('Feature Importances:')
for i, f in enumerate(train3_X.columns):
    print('\t{0:20s} : {1:>.6f}'.format(f, fti[i]))
train3_X_selected = train3_X[['Sex', 'Age', 'Fare']]
test3_X_selected = test3_X[['Sex', 'Age', 'Fare']]

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(train3_X_selected, train3_y)
Y_pred3 = rfc.predict(test3_X_selected)

# lr.fit(train3_X_selected, train3_y)
# Y_pred3 = lr.predict(test3_X_selected)
# print(round(lr.score(train3_X_selected, train3_y) * 100, 2))
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred3})
# submission.to_csv('submission.csv', index=False)
# 検証4
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# カテゴリデータの扱い　LabelEncoder
# ★SibSpとParchをもとに家族数を作成
# ★もし家族数が1人なら、一人フラグを立てる。
# スコア　0.77990
train4 = train_data.copy()
test4 = test_data.copy()

train4['Age'] = train4['Age'].fillna(train4['Age'].median())
train4['Embarked'] = train4['Embarked'].fillna('S')
train4 = train4.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test4['Age'] = test4['Age'].fillna(test4['Age'].median())
test4['Fare'] = test4['Fare'].fillna(test4['Fare'].median())
test4 = test4.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 検証4
train4['Sex'] = le.fit_transform(train4['Sex'])
train4['Embarked'] = le.fit_transform(train4['Embarked'])
test4['Sex'] = le.fit_transform(test4['Sex'])
test4['Embarked'] = le.fit_transform(test4['Embarked'])
train4['FamilySize'] = train4['SibSp'] + train4['Parch'] + 1
train4['IsAlone'] =  1
train4['IsAlone'].loc[train4['FamilySize'] > 1] = 0
train4 = train4.drop(['SibSp', 'Parch'], axis=1)

test4['FamilySize'] = test4['SibSp'] + test4['Parch'] + 1
test4['IsAlone'] =  1
test4['IsAlone'].loc[test4['FamilySize'] > 1] = 0

test4 = test4.drop(['SibSp', 'Parch'], axis=1)
# 検証4
train4_y = train4["Survived"]
train4_X = train4.drop('Survived', axis=1)
test4_X = test4

lr.fit(train4_X, train4_y)

Y_pred4 = lr.predict(test4_X)
print(round(lr.score(train4_X, train4_y) * 100, 2))
train4_X.head()
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred4})
# submission.to_csv('submission.csv', index=False)
train4_X.head()
# 検証5
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# カテゴリデータの扱い　LabelEncoder
# SibSpとParchをもとに家族数を作成
# もし家族数が1人なら、一人フラグを立てる。
# ★Fareを対数化
# スコア　0.76076
train5 = train_data.copy()
test5 = test_data.copy()

train5['Age'] = train5['Age'].fillna(train5['Age'].median())
train5['Embarked'] = train5['Embarked'].fillna('S')
train5 = train5.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test5['Age'] = test5['Age'].fillna(test5['Age'].median())
test5['Fare'] = test5['Fare'].fillna(test5['Fare'].median())
test5 = test5.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# 検証5
train5['Sex'] = le.fit_transform(train5['Sex'])
train5['Embarked'] = le.fit_transform(train5['Embarked'])
test5['Sex'] = le.fit_transform(test5['Sex'])
test5['Embarked'] = le.fit_transform(test5['Embarked'])
train5['FamilySize'] = train5['SibSp'] + train5['Parch'] + 1
train5['IsAlone'] =  1
train5['IsAlone'].loc[train5['FamilySize'] > 1] = 0
train5 = train5.drop(['SibSp', 'Parch'], axis=1)

test5['FamilySize'] = test5['SibSp'] + test5['Parch'] + 1
test5['IsAlone'] =  1
test5['IsAlone'].loc[test5['FamilySize'] > 1] = 0

test5 = test5.drop(['SibSp', 'Parch'], axis=1)
train5['Fare'] = np.log1p(train5['Fare'])
test5['Fare'] = np.log1p(test5['Fare'])
# 検証5
train5_y = train5["Survived"]
train5_X = train5.drop('Survived', axis=1)
test5_X = test5

lr.fit(train5_X, train5_y)

Y_pred5 = lr.predict(test5_X)
print(round(lr.score(train5_X, train5_y) * 100, 2))
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred5})
# submission.to_csv('submission.csv', index=False)
# 検証6
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# カテゴリデータの扱い　LabelEncoder
# SibSpとParchをもとに家族数を作成
# もし家族数が1人なら、一人フラグを立てる。
# ★Cabinカラムがn/aか否かフラグを追加し学習。
# スコア　0.77033
train6 = train_data.copy()
test6 = test_data.copy()

train6['Age'] = train6['Age'].fillna(train6['Age'].median())
train6['Embarked'] = train6['Embarked'].fillna('S')

train6['HasCabin'] = 1
train6['HasCabin'].loc[train6['Cabin'].isnull()] = 0

train6 = train6.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test6['Age'] = test6['Age'].fillna(test6['Age'].median())
test6['Fare'] = test6['Fare'].fillna(test6['Fare'].median())

test6['HasCabin'] = 1
test6['HasCabin'].loc[test6['Cabin'].isnull()] = 0

test6 = test6.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
sns.countplot(train6['HasCabin'], hue=train6['Survived'])
sns.countplot(train6['HasCabin'], hue=train6['Pclass'])
# 検証6
train6['Sex'] = le.fit_transform(train6['Sex'])
train6['Embarked'] = le.fit_transform(train6['Embarked'])
test6['Sex'] = le.fit_transform(test6['Sex'])
test6['Embarked'] = le.fit_transform(test6['Embarked'])
train6['FamilySize'] = train6['SibSp'] + train6['Parch'] + 1
train6['IsAlone'] =  1
train6['IsAlone'].loc[train6['FamilySize'] > 1] = 0
train6 = train6.drop(['SibSp', 'Parch'], axis=1)

test6['FamilySize'] = test6['SibSp'] + test6['Parch'] + 1
test6['IsAlone'] =  1
test6['IsAlone'].loc[test6['FamilySize'] > 1] = 0

test6 = test6.drop(['SibSp', 'Parch'], axis=1)
# 検証6
train6_y = train6["Survived"]
train6_X = train6.drop('Survived', axis=1)
test6_X = test6

lr.fit(train6_X, train6_y)

Y_pred6 = lr.predict(test6_X)
print(round(lr.score(train6_X, train6_y) * 100, 2))
# submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred6})
# submission.to_csv('submission.csv', index=False)
# 検証7
# 欠損値の扱い　Age：中央値の代入　Embarked：最頻値Sの代入
# 文字列カラム　Name Ticket Cabinのドロップ
# ★カテゴリデータの扱い　OneHotEncoding
# SibSpとParchをもとに家族数を作成
# もし家族数が1人なら、一人フラグを立てる。
# ★敬称追加
# スコア　0.78468
train7 = train_data.copy()
test7 = test_data.copy()

train7_y = train7["Survived"]
train7 = train7.drop(["Survived"], axis=1)

train7["Mr"] = train7["Name"].apply(lambda x: x.count("Mr."))
train7["Mrs"] = train7["Name"].apply(lambda x: x.count("Mrs."))
train7["Miss"] = train7["Name"].apply(lambda x: x.count("Miss."))
train7["Master"] = train7["Name"].apply(lambda x: x.count("Master."))

test7["Mr"] = test7["Name"].apply(lambda x: x.count("Mr."))
test7["Mrs"] = test7["Name"].apply(lambda x: x.count("Mrs."))
test7["Miss"] = test7["Name"].apply(lambda x: x.count("Miss."))
test7["Master"] = test7["Name"].apply(lambda x: x.count("Master."))

le.fit(list(train7['Ticket'].values) + list(test7['Ticket'].values))
train7['Ticket'] = le.transform(list(train7['Ticket'].values))
test7['Ticket'] = le.transform(list(test7['Ticket'].values))

train7['Cabin'] = train7['Cabin'].fillna('U')
test7['Cabin'] = test7['Cabin'].fillna('U')
le.fit(list(train7['Cabin']) + list(test7['Cabin']))
train7['Cabin'] = le.transform(list(train7['Cabin'].values))
test7['Cabin'] = le.transform(list(test7['Cabin'].values))

train7['Age'] = train7['Age'].fillna(train7['Age'].median())
train7['Embarked'] = train7['Embarked'].fillna('S')
train7 = train7.drop(['PassengerId', 'Name'], axis=1)

test7['Age'] = test7['Age'].fillna(test7['Age'].median())
test7['Fare'] = test7['Fare'].fillna(test7['Fare'].median())
test7 = test7.drop(['PassengerId', 'Name'], axis=1)

train7['Fare'] = pd.cut(train7['Fare'], 5, labels=False)
test7['Fare'] = pd.cut(test7['Fare'], 5, labels=False)

train7['Age'] = pd.cut(train7['Age'], 5, labels=False)
test7['Age'] = pd.cut(test7['Age'], 5, labels=False)
train7['Sex'] = le.fit_transform(train7['Sex'])
train7['Embarked'] = le.fit_transform(train7['Embarked'])
test7['Sex'] = le.fit_transform(test7['Sex'])
test7['Embarked'] = le.fit_transform(test7['Embarked'])
train7['FamilySize'] = train7['SibSp'] + train7['Parch'] + 1
train7['IsAlone'] =  1
train7['IsAlone'].loc[train7['FamilySize'] > 1] = 0
train7 = train7.drop(['SibSp', 'Parch'], axis=1)

test7['FamilySize'] = test7['SibSp'] + test7['Parch'] + 1
test7['IsAlone'] =  1
test7['IsAlone'].loc[test7['FamilySize'] > 1] = 0

test7 = test7.drop(['SibSp', 'Parch'], axis=1)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train7 = pd.DataFrame(ss.fit_transform(train7), columns=train7.columns)
test7 = pd.DataFrame(ss.fit_transform(test7), columns=test7.columns)
train7.head()
test7.head()
# 検証7
train7_X = train7
test7_X = test7
# lr.fit(train7_X, train7_y)
# Y_pred7_lr = lr.predict(test7_X)

# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier()
# dtc.fit(train7_X, train7_y)
# Y_pred7_dtc = dtc.predict(test7_X)

# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
# rfc.fit(train7_X, train7_y)
# Y_pred7_rfc = rfc.predict(test7_X)

# from sklearn.grid_search import GridSearchCV
# tuned_parameters = [
#     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#     {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
#     {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
#     {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
#     ]
from sklearn.svm import SVC
# score = 'f1'
# svc = GridSearchCV(
#     SVC(), # 識別器
#     tuned_parameters, # 最適化したいパラメータセット 
#     cv=5, # 交差検定の回数
#     scoring='%s_weighted' % score )
svc = SVC()
svc.fit(train7_X, train7_y)
Y_pred7 = svc.predict(test7_X)

print(round(svc.score(train7_X, train7_y) * 100, 2))
print(Y_pred7)
# Y_pred7 = []
# for i in range(418):
#     if((Y_pred3[i] + Y_pred4[i] + Y_pred5[i] + Y_pred6[i] + Y_pred7_svc[i]) >= 3):
#         Y_pred7.append(1)
#     else:
#         Y_pred7.append(0)
# print(len(Y_pred7))
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred7})
submission.to_csv('sub.csv', index=False)





















