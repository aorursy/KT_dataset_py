import numpy as np 
import pandas as pd
from sklearn import svm
pd.set_option('line_width', 100)
#データの読み込み
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
#トレーニングデータを上から１０件見てみる
train.head(10)
#テストデータを上から１０件見てみる
test.head(10)
#欠損値があるかどうか調べる
train.isnull().any()
#年齢の欠損値を中央値で補完
train.Age = train.Age.fillna(train.Age.median())
#料金の欠損値を中央値で補完
train.Fare = train.Fare.fillna(train.Fare.median())
#料金の欠損値をSで補完
train.Embarked = train.Embarked.fillna("S")
train.head(10)
#その他の前処理

#Sexをダミー変数に変換
#男→1、女→0に変換
train["Sex"] = train["Sex"].map( {'female': 0, "male": 1} ).astype(int)
#Embarkedをダミー変数に変換
Embark_dum  = pd.get_dummies(train['Embarked'])
train = train.join(Embark_dum)
#不要な列を削除
train.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
train
