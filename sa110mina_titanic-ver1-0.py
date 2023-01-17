#ライブラリインポート

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import tree



#データセット読み込み

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#項目確認

train.head()
#項目確認

test.head()
df_data = [train,test]

#名前からtitle(Mr, Miss,Master)を取り出す

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]

for df in df_data:

    df["Title"] = pd.Series(dataset_title)

    df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df.Title = df.Title.replace('Mlle', 'Miss')

    df.Title = df.Title.replace('Ms', 'Miss')

    df.Title = df.Title.replace('Mme', 'Mrs')

 # 欠損データを埋める

    df.Title = df.Title.fillna('Missing')

# 生存率を確認

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#title毎の年齢の平均値

for df in df_data:

    df.groupby('Title').Age.mean()

#agg:辞書型のオブジェクトを渡すことでカラムに対して特定の集計をするように指示できる

#title毎の中央値

    df.groupby('Title').Age.agg(['count','median'])

#年齢の欠損値を埋める

    df.loc[df.Age.isnull(), 'Age'] = df.groupby(['Title']).Age.transform('median')  
#titleを数値に置き換え

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5, "Missing": 0}

for df in df_data:

    df['Title'] = df['Title'].map(title_mapping)

    df["Title"] = df["Title"].astype(int)
#乗船港欠損値埋める

train["Embarked"] = train["Embarked"].fillna("S")



#性別、乗船港を数値に置き換え

for df in df_data:

    df['Sex']= df[['Sex']].replace(['male','female'],[0,1])

    df['Embarked']= df['Embarked'].replace(['C','Q','S'],[0,1,2])
#性別の生存率関係

print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
#乗船港の生存率関係

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
#目的変数と説明変数を決定して取得

target = train["Survived"].values

explain = train[["Pclass", "Sex", "Age"]].values
#決定木の作成

d_tree = tree.DecisionTreeClassifier()

#fit()で学習させる。第一引数に説明変数、第二引数に目的変数

d_tree = d_tree.fit(explain, target)
#testデータから説明変数を抽出

test_explain = test[["Pclass", "Sex", "Age"]].values

#predict()メソッドで予測する

prediction = d_tree.predict(test_explain)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

# 予測データとPassengerIdをデータフレームにて結合

result = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])

# result.csvとして書き出し

result.to_csv("result.csv", index_label = ["PassengerId"])