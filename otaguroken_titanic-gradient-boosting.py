# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 提出ファイルのフォーマットもこの形式
# You can download an example submission file (gender_submission.csv)
! head /kaggle/input/titanic/gender_submission.csv
# train.csv, survivedが記載されている
! head /kaggle/input/titanic/train.csv
# testはsurvivedはない
! head /kaggle/input/titanic/test.csv
# Save Versionを押してもprivateのまま
# https://www.kaggle.com/c/titanic/notebooks
# 特徴量選択(RandomForest)
# https://qiita.com/rockhopper/items/a68ceb3248f2b3a41c89
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# https://ohke.hateblo.jp/entry/2017/08/04/230000
from sklearn.model_selection import train_test_split
import pandas as pd
traindata = pd.read_csv( '/kaggle/input/titanic/train.csv', delimiter=',' )
testdata = pd.read_csv( '/kaggle/input/titanic/test.csv', delimiter=',' )
sample_submit = pd.read_csv( '/kaggle/input/titanic/gender_submission.csv')
# そのまま使う:PasengerId, Pclass(客室のグレード), Age, SibSp(兄弟の数), Parch(親子の数), Fare(運賃)
# 文字列なので直す必要がある：Sex, Ticket, Cabin
traindata
# 名前の中に敬称の情報があるらしいので取り出しておく
import re
traindata["title"] = traindata["Name"].map(lambda x: re.search(r',(.*)\.', x)[1])
#traindata["ticket"] = traindata["Ticket"].map(lambda x: re.search(r'[^\d]*', x)[0])
#traindata['Survived'].groupby(traindata['ticket']).agg(['mean','count'])
# ticketの文字に意味がありそうだけど、よくわからないので先頭を取り出すだけにする
traindata["ticket"] = traindata["Ticket"].map(lambda x: re.search(r'[^\d]*', x)[0][:1])
traindata['Survived'].groupby(traindata['ticket']).agg(['mean','count'])
import math

# cabinは欠損が多いけれど、利用できる範囲でカテゴリ変数にしておく
traindata["cabin"] = traindata["Cabin"].fillna("0").map(lambda x: x[:2])
traindata["cabin1"] = traindata["Cabin"].fillna("0").map(lambda x: x[:1])
traindata["cabin2"] = traindata["Cabin"].fillna("0").map(lambda x: x[1:2])
traindata['Survived'].groupby(traindata['cabin']).agg(['mean','count'])
# traindata のカテゴリ変数を0,1に変換していく (testdataにしたあとにずれたりしない様に、手動で定義しておく)
traindata["male"] = traindata["Sex"].map(lambda x: 1 if x == "male" else 0)
traindata["female"] = traindata["Sex"].map(lambda x: 1 if x == "female" else 0)
traindata["master"] = traindata["title"].map(lambda x: 1 if x == "Master" else 0)
traindata["miss"] = traindata["title"].map(lambda x: 1 if x == "miss" else 0)
traindata["mr"] = traindata["title"].map(lambda x: 1 if x == "Mr" else 0)
traindata["mrs"] = traindata["title"].map(lambda x: 1 if x == "Mrs" else 0)
traindata["ticketA"] = traindata["ticket"].map(lambda x: 1 if x == "a" else 0)
traindata["ticketC"] = traindata["ticket"].map(lambda x: 1 if x == "c" else 0)
traindata["ticketF"] = traindata["ticket"].map(lambda x: 1 if x == "f" else 0)
traindata["ticketL"] = traindata["ticket"].map(lambda x: 1 if x == "l" else 0)
traindata["ticketP"] = traindata["ticket"].map(lambda x: 1 if x == "p" else 0)
traindata["ticketS"] = traindata["ticket"].map(lambda x: 1 if x == "s" else 0)
traindata["ticketW"] = traindata["ticket"].map(lambda x: 1 if x == "w" else 0)
traindata["cabin10"] = traindata["cabin1"].map(lambda x: 1 if x == "0" else 0)
traindata["cabin1A"] = traindata["cabin1"].map(lambda x: 1 if x == "A" else 0)
traindata["cabin1B"] = traindata["cabin1"].map(lambda x: 1 if x == "B" else 0)
traindata["cabin1C"] = traindata["cabin1"].map(lambda x: 1 if x == "C" else 0)
traindata["cabin1D"] = traindata["cabin1"].map(lambda x: 1 if x == "D" else 0)
traindata["cabin1E"] = traindata["cabin1"].map(lambda x: 1 if x == "E" else 0)
traindata["cabin1F"] = traindata["cabin1"].map(lambda x: 1 if x == "F" else 0)
traindata["cabin21"] = traindata["cabin2"].map(lambda x: 1 if x == "1" else 0)
traindata["cabin22"] = traindata["cabin2"].map(lambda x: 1 if x == "2" else 0)
traindata["cabin23"] = traindata["cabin2"].map(lambda x: 1 if x == "3" else 0)
traindata["cabin24"] = traindata["cabin2"].map(lambda x: 1 if x == "4" else 0)
traindata["cabin25"] = traindata["cabin2"].map(lambda x: 1 if x == "5" else 0)
traindata["cabin26"] = traindata["cabin2"].map(lambda x: 1 if x == "6" else 0)
traindata["cabin27"] = traindata["cabin2"].map(lambda x: 1 if x == "7" else 0)
traindata["cabin28"] = traindata["cabin2"].map(lambda x: 1 if x == "8" else 0)
traindata["cabin29"] = traindata["cabin2"].map(lambda x: 1 if x == "9" else 0)
traindata["embarkedC"] = traindata["Embarked"].map(lambda x: 1 if x == "C" else 0)
traindata["embarkedQ"] = traindata["Embarked"].map(lambda x: 1 if x == "Q" else 0)
traindata["embarkedS"] = traindata["Embarked"].map(lambda x: 1 if x == "S" else 0)
# 値の確認用
key = "Embarked"
traindata[key].groupby(traindata[key]).count()
# 正解データと、扱いづらい名前を特徴量から除く
# なお、データの説明はこちら：https://www.kaggle.com/c/titanic/data?select=train.csv
feature_names = [_ for _ in traindata.columns if  (_!= "Survived"and _!= "Sex" and _!= "Name" and _!= "cabin" and _!= "cabin1" and _!= "cabin2" and _!= "Embarked"  and _!= "Ticket" and _!= "ticket" and _!= "Cabin" and _!= "title")]
feature_names
X = pd.DataFrame(data=traindata, columns=feature_names)
y = pd.DataFrame(data=traindata, columns=['Survived'])
# 全部数字になったことを確認
X
# 欠損があるようなので、とりあえず-1で埋めておく
X = X.fillna(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_test
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(min_samples_leaf=3, random_state=0)
forest.fit(X_train, y_train)
# 評価
from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import confusion_matrix
print('Train score: {}'.format(forest.score(X_train, y_train)))
print('Test score: {}'.format(forest.score(X_test, y_test)))
print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, forest.predict(X_test))))
print('f1 score: {:.3f}'.format(f1_score(y_test, forest.predict(X_test))))
print('train log_loss: {:.3f}'.format(log_loss(y_train, forest.predict_proba(X_train))))
print('test log_loss: {:.3f}'.format(log_loss(y_test, forest.predict_proba(X_test))))
# ほとんど性別、運賃が重要。敬称とか入れてみたけれど、importanceは0
for n, v in zip(feature_names, forest.feature_importances_):
    print(f'importance ：{n} :{v}')
# Gradient boostingも使ってみる こちらの方が良さそう
from sklearn.ensemble import GradientBoostingClassifier
forest = GradientBoostingClassifier(min_samples_leaf=1, random_state=0)
forest.fit(X_train, y_train)
print('Train score: {}'.format(forest.score(X_train, y_train)))
print('Test score: {}'.format(forest.score(X_test, y_test)))
print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, forest.predict(X_test))))
print('f1 score: {:.3f}'.format(f1_score(y_test, forest.predict(X_test))))
print('train log_loss: {:.3f}'.format(log_loss(y_train, forest.predict_proba(X_train))))
print('test log_loss: {:.3f}'.format(log_loss(y_test, forest.predict_proba(X_test))))
# 提出用に全てのデータで学習
forest.fit(X, y)
print('Train score: {}'.format(forest.score(X, y)))
print('Confusion matrix:\n{}'.format(confusion_matrix(y, forest.predict(X))))
print('train log_loss: {:.3f}'.format(log_loss(y, forest.predict_proba(X))))
# 名前の中に敬称の情報があるらしいので取り出しておく
import re
testdata["title"] = testdata["Name"].map(lambda x: re.search(r',(.*)\.', x)[1])

# ticketの文字に意味がありそうだけど、よくわからないので先頭を取り出すだけにする
testdata["ticket"] = testdata["Ticket"].map(lambda x: re.search(r'[^\d]*', x)[0][:1])

# cabinは欠損が多いけれど、利用できる範囲でカテゴリ変数にしておく
testdata["cabin"] = testdata["Cabin"].fillna("0").map(lambda x: x[:2])
testdata["cabin1"] = testdata["Cabin"].fillna("0").map(lambda x: x[:1])
testdata["cabin2"] = testdata["Cabin"].fillna("0").map(lambda x: x[1:2])

# testdata のカテゴリ変数を0,1に変換していく (testdataにしたあとにずれたりしない様に、手動で定義しておく)
testdata["male"] = testdata["Sex"].map(lambda x: 1 if x == "male" else 0)
testdata["female"] = testdata["Sex"].map(lambda x: 1 if x == "female" else 0)
testdata["master"] = testdata["title"].map(lambda x: 1 if x == "Master" else 0)
testdata["miss"] = testdata["title"].map(lambda x: 1 if x == "miss" else 0)
testdata["mr"] = testdata["title"].map(lambda x: 1 if x == "Mr" else 0)
testdata["mrs"] = testdata["title"].map(lambda x: 1 if x == "Mrs" else 0)
testdata["ticketA"] = testdata["ticket"].map(lambda x: 1 if x == "a" else 0)
testdata["ticketC"] = testdata["ticket"].map(lambda x: 1 if x == "c" else 0)
testdata["ticketF"] = testdata["ticket"].map(lambda x: 1 if x == "f" else 0)
testdata["ticketL"] = testdata["ticket"].map(lambda x: 1 if x == "l" else 0)
testdata["ticketP"] = testdata["ticket"].map(lambda x: 1 if x == "p" else 0)
testdata["ticketS"] = testdata["ticket"].map(lambda x: 1 if x == "s" else 0)
testdata["ticketW"] = testdata["ticket"].map(lambda x: 1 if x == "w" else 0)
testdata["cabin10"] = testdata["cabin1"].map(lambda x: 1 if x == "0" else 0)
testdata["cabin1A"] = testdata["cabin1"].map(lambda x: 1 if x == "A" else 0)
testdata["cabin1B"] = testdata["cabin1"].map(lambda x: 1 if x == "B" else 0)
testdata["cabin1C"] = testdata["cabin1"].map(lambda x: 1 if x == "C" else 0)
testdata["cabin1D"] = testdata["cabin1"].map(lambda x: 1 if x == "D" else 0)
testdata["cabin1E"] = testdata["cabin1"].map(lambda x: 1 if x == "E" else 0)
testdata["cabin1F"] = testdata["cabin1"].map(lambda x: 1 if x == "F" else 0)
testdata["cabin21"] = testdata["cabin2"].map(lambda x: 1 if x == "1" else 0)
testdata["cabin22"] = testdata["cabin2"].map(lambda x: 1 if x == "2" else 0)
testdata["cabin23"] = testdata["cabin2"].map(lambda x: 1 if x == "3" else 0)
testdata["cabin24"] = testdata["cabin2"].map(lambda x: 1 if x == "4" else 0)
testdata["cabin25"] = testdata["cabin2"].map(lambda x: 1 if x == "5" else 0)
testdata["cabin26"] = testdata["cabin2"].map(lambda x: 1 if x == "6" else 0)
testdata["cabin27"] = testdata["cabin2"].map(lambda x: 1 if x == "7" else 0)
testdata["cabin28"] = testdata["cabin2"].map(lambda x: 1 if x == "8" else 0)
testdata["cabin29"] = testdata["cabin2"].map(lambda x: 1 if x == "9" else 0)
testdata["embarkedC"] = testdata["Embarked"].map(lambda x: 1 if x == "C" else 0)
testdata["embarkedQ"] = testdata["Embarked"].map(lambda x: 1 if x == "Q" else 0)
testdata["embarkedS"] = testdata["Embarked"].map(lambda x: 1 if x == "S" else 0)

X_target = pd.DataFrame(data=testdata, columns=feature_names)
X_target = X_target.fillna(-1)
pred = forest.predict(X_target)
pred
with open("result.csv", mode='w') as f:
    f.write("PassengerId,Survived\n")
    for pid, pr in zip(X_target["PassengerId"], pred):
        f.write("{},{}\n".format(pid,pr))

# 
! head  result.csv && tail result.csv
#此処より下はGRID SEARCHのテスト
forest_grid_param = {
    'n_estimators': [50,100,200,400],
    'max_features': [3, 5, 10, 50, 100, 'auto']
}

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
# スコア方法をF1に設定
log_loss_min = make_scorer(log_loss)

# グリッドサーチで学習
forest_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=0), forest_grid_param, scoring=log_loss_min, cv=2, verbose=3)
forest_grid_search.fit(X_train, y_train.values.ravel())

print('Train score: {}'.format(forest.score(X_train, y_train)))
print('Test score: {}'.format(forest.score(X_test, y_test)))
print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, forest.predict(X_test))))
print('f1 score: {:.3f}'.format(f1_score(y_test, forest.predict(X_test))))
print('train log_loss: {:.3f}'.format(log_loss(y_train, forest.predict_proba(X_train))))
print('test log_loss: {:.3f}'.format(log_loss(y_test, forest.predict_proba(X_test))))