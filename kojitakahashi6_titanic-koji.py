#  Python ライブラリの参照



import pandas as pd
# データセットの読み込み



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#  Train データの概要 



train.head(3)
#  不必要な列を drop する





train = train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test = test.drop(['Name','Ticket','Cabin'], axis=1)





# ちゃんと drop されたか確認

train.head(2)
# train データの欠損値の数



train.isnull().sum()
# test データの欠損値の数



test.isnull().sum()
# Train data 内の Embarked の 2つの 欠損値を埋める



# まず最初にEmbarkedの mode (最頻値) を求める。

pd.value_counts(train['Embarked'])
#ModeはSだということが分かったので、Embarkedの２つの欠損値をSで埋める。



_ = train.fillna({'Embarked': 'S'}, inplace = True)
# Test データ内の Fare の欠損値を平均値で埋める



# test の Fare の mean

test['Fare'].mean()
# Test の Fare の　欠損値を fillna をつかって 平均値 (mean) で埋める。



_ = test.fillna({'Fare': 35.627}, inplace = True)
# Embarked と sex が文字列なので、LabelEncoder を使って数列に置き換える。

  

from sklearn.preprocessing import LabelEncoder



LE=LabelEncoder()



labels = ['Embarked','Sex']

for label in labels:

    train[label]=LE.fit_transform(train[label])

    test[label]=LE.fit_transform(test[label])

train.head()
#  Age を機械学習ライブラリを使って埋めていく



from sklearn.ensemble import RandomForestRegressor



#  下記は Poonam Ligade さんの kernel を参照しました



#Random Forest の機械学習ライブラリを使い Age の欠損値を埋めていく



def fill_missing_age(df):

    

    #使う特徴

    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',

                 'Pclass']]

    # Age が欠損値じゃないのと、欠損値であるもので2グループに分ける

    

    train  = age_df.loc[ (df.Age.notnull()) ]# Age あり

    test = age_df.loc[ (df.Age.isnull()) ]# Age 欠損値

    

    # Ageの部分

    y = train.values[:, 0]

    

    # Age以外の部分を特徴量として扱う

    X = train.values[:, 1::]

    

    # 機械学習のモデル作り

    

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # 機械学習モデルを適応する

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # 元のデータフレームに予測された Age の値を返す

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df
# 　上記の機械学習関数を実際のデータセットに適応



train = fill_missing_age(train)

test=fill_missing_age(test)

#  Age と Fare だけ、値の幅が広いので、結果に異常に大きく反映されやすい。

#  よって、 Standard Scaler を使って、他の特徴量と平等の重みになるようにする。



from sklearn.preprocessing import StandardScaler



std_scale = StandardScaler().fit(train[['Age', 'Fare']])

train[['Age', 'Fare']] = std_scale.transform(train[['Age', 'Fare']])





std_scale = StandardScaler().fit(test[['Age', 'Fare']])

test[['Age', 'Fare']] = std_scale.transform(test[['Age', 'Fare']])
# Machine learning で使うデータとセットに分ける。



X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# 機械学習たちの一発適用で、どれが一番精度がいいかを見極める、



from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score





#機械学習モデルをリストに格納

models = []

models.append(("KNC",KNeighborsClassifier()))

models.append(("DTC",DecisionTreeClassifier()))

models.append(("SVM",SVC()))

models.append(("AdaBoost",AdaBoostClassifier()))

models.append(("GradientBoosting",GradientBoostingClassifier()))



#複数のclassifier の適用

results = []

names = []

for name,model in models:

    kfold = KFold(n_splits=10, random_state=42)

    result = cross_val_score(model,X_train,Y_train, cv = kfold, scoring = "accuracy")

    names.append(name)

    results.append(result)





#適用したclassifierのスコア表示

for i in range(len(names)):

     print(names[i], results[i].mean())
#  一番スコアの良い Gradient Boosting を使って予測



clf = SVC()

clf.fit(X_train, Y_train)

result = clf.predict(X_test)
#  提出する

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": result

    })



#  下記の # を外すと、 csv形式のファイルに出力

#  submission.to_csv("submission.csv", index=False)