import pandas as pd #データ分析用

import numpy as np  #データ分析用



train = pd.read_csv('../input/titanic/train.csv') #訓練データ

test = pd.read_csv("../input/titanic/test.csv") #テストデータ

gender = pd.read_csv("../input/titanic/gender_submission.csv") #同梱されていた提出用ファイルのサンプル
train.head() #訓練データの要素を見る：head()で最初の５行を見ることができる
gender.head() #どうやって提出するか見る
data = pd.concat([train, test], sort=False) #concatで複数のdfを結合させる(sortオプションでソートしないようにしている)
data.isnull() #nullが入っているかどうかをチェックする
data.isnull().sum() #sumと組み合わせることで欠損値が入っているかすぐ確認ができる
data['Sex'].replace(['male','female'], [0, 1], inplace=True) #maleを0に、femaleを1に変換する。
data['Embarked'].value_counts()
data['Embarked'].fillna(('S'), inplace=True) #欠損値を置換するメソッド

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) #mapを使った置換
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True) #リストで行を指定することでまとめて削除できる
train = data[:len(train)] #スライスを使って、先頭から訓練データの数だけを切り取る

test = data[len(train):] #訓練データの数から数えてその後を切り取る
y_train = train['Survived'] #訓練データの答えの部分を作る

X_train = train.drop('Survived', axis=1) #訓練データの特徴量だけの部分を作る

X_test = test.drop('Survived', axis=1) #dataに結合した時に付いたSurvivedを削除しておく
X_train.head()
y_train.head()
X_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
clf = RandomForestClassifier(random_state=0)

#clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)

#clf = SVC(kernel = "rbf")
clf.fit(X_train, y_train) #訓練データに基づいた学習
y_pred =  clf.predict(X_test) #y_predに予想が入る
y_pred[:5] #予想が入っているか確認
sub = gender #サンプルファイルを参考に作る

sub['Survived'] = list(map(int, y_pred)) #学習器の予想に書き換える

sub.to_csv("submission.csv", index=False)  #csv形式に変換
sub = pd.read_csv("../input/titanic/test.csv") 

sub = sub.loc[:,["PassengerId"]] #特定の列をdfとして取り出す

sub["Survived"]= list(map(int, y_pred)) #予測を列として追加する
sub.to_csv("submission.csv", index=False)
train_reform = pd.read_csv("../input/titanic/train.csv") 

test_reform = pd.read_csv("../input/titanic/test.csv")

train_reform.head()
import matplotlib.pyplot as plt #グラフの描画

import seaborn as sns #グラフの描画

%matplotlib inline  

#jupyter notebook上でグラフを表示させる
sns.countplot(x="Parch",data=train_reform,hue="Survived")
train_reform['FamilySize'] = train_reform['Parch'] + train_reform['SibSp'] + 1 #本人として１を足す

sns.countplot(x='FamilySize', data = train_reform, hue='Survived')