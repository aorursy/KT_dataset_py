import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median()) # 年齢の欠損値を中央値で埋める
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0]) #出港地の欠損値を最頻値で埋める

sex = train["Sex"].unique()
for i in range(len(sex)):# 性別を0から始まる数値に変換
    train.loc[train['Sex'] == sex[i], 'Sex'] = i

train = pd.get_dummies(train, columns=['Embarked'])
# embarked = train["Embarked"].unique()
# for i in range(len(embarked)): # 出港地を0から始まる数値に変換
#     train.loc[train["Embarked"] == embarked[i], "Embarked"] = i

test["Age"] = test["Age"].fillna(test["Age"].median())

nulcol = test.columns[test.isnull().sum()>0]
for i in range(len(nulcol)):# 欠損値がある列をその列の最頻値で埋める
    test[nulcol[i]] = test[nulcol[i]].fillna(test[nulcol[i]].mode()[0])
# test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode()[0]) #最頻値
# test["Fare"] = test["Fare"].fillna(test["Fare"].mode()[0]) #最頻値

sex = test["Sex"].unique()
for i in range(len(sex)):
    test.loc[test['Sex'] == sex[i], 'Sex'] = i

test = pd.get_dummies(test, columns=['Embarked'])
# embarked = test["Embarked"].unique()
# for i in range(len(embarked)):
#     test.loc[test["Embarked"] == embarked[i], "Embarked"] = i

from sklearn import tree
Y_train = train["Survived"].values
X_train = train[["Pclass", "Sex", "Age", "Fare", 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values
mytree = tree.DecisionTreeClassifier()
mytree.fit(X_train, Y_train)
X_test = test[["Pclass", "Sex", "Age", "Fare", 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values
pred = mytree.predict(X_test)

passid = np.array(test['PassengerId'])
sol = pd.DataFrame({"PassengerId": passid, "Survived":pred})
sol.to_csv("tree.csv", index=False) #0.72248, 0.71291

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
pred = lr.predict(X_test)
sol = pd.DataFrame({"PassengerId": passid, "Survived":pred})
sol.to_csv("lr.csv", index=False) # 0.74641, 0.75598,

from sklearn.ensemble import GradientBoostingClassifier
forest = GradientBoostingClassifier(n_estimators=55, random_state=9)
forest.fit(X_train, Y_train)
pred = lr.predict(X_test)
sol = pd.DataFrame({"PassengerId": passid, "Survived":pred})
sol.to_csv("forest.csv", index=False) # 0.74641, 0.75598
import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median()) # 年齢の欠損値を中央値で埋める
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0]) #出港地の欠損値を最頻値で埋める

sex = train["Sex"].unique()
for i in range(len(sex)):# 性別を0から始まる数値に変換
    train.loc[train['Sex'] == sex[i], 'Sex'] = i

embarked = train["Embarked"].unique()
for i in range(len(embarked)): # 出港地を0から始まる数値に変換
    train.loc[train["Embarked"] == embarked[i], "Embarked"] = i


test["Age"] = test["Age"].fillna(test["Age"].median())
nulcol = test.columns[test.isnull().sum()>0]
for i in range(len(nulcol)):# 欠損値がある列をその列の最頻値で埋める
    test[nulcol[i]] = test[nulcol[i]].fillna(test[nulcol[i]].mode()[0])
# test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode()[0]) #最頻値
# test["Fare"] = test["Fare"].fillna(test["Fare"].mode()[0]) #最頻値
sex = test["Sex"].unique()
for i in range(len(sex)):
    test.loc[test['Sex'] == sex[i], 'Sex'] = i
embarked = test["Embarked"].unique()
for i in range(len(embarked)):
    test.loc[test["Embarked"] == embarked[i], "Embarked"] = i

from sklearn import tree
Y_train = train["Survived"].values
X_train = train[["Pclass", "Sex", "Age", "Fare"]].values
mytree = tree.DecisionTreeClassifier()
mytree.fit(X_train, Y_train)
X_test = test[["Pclass", "Sex", "Age", "Fare"]].values
pred = mytree.predict(X_test)

passid = np.array(test['PassengerId'])
sol = pd.DataFrame(np.c_[passid, pred], columns=["PassengerId", "Survived"])
sol.to_csv("tree.csv", index=None)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
pred = lr.predict(X_test)
sol = pd.DataFrame(np.c_[passid, pred], columns=["PassengerId", "Survived"])
sol.to_csv("lr.csv", index=None)

from sklearn.ensemble import GradientBoostingClassifier
forest = GradientBoostingClassifier(n_estimators=55, random_state=9)
forest.fit(X_train, Y_train)
pred = lr.predict(X_test)
sol = pd.DataFrame(np.c_[passid, pred], columns=["PassengerId", "Survived"])
sol.to_csv("forest.csv", index=None)


