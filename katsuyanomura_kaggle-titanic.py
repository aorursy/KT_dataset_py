import numpy as np 

import pandas as pd 





import os

#ディレクトリ構造の表示

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#訓練用データをロード

train = pd.read_csv("/kaggle/input/titanic/train.csv")

#テスト用データをロード

test = pd.read_csv("/kaggle/input/titanic/test.csv")

#サンプル提出データをロード

sample_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
#(行数,列数)を表示

train.shape
#上から5行を表示

train.head()
import math

#可視化用モジュール

import seaborn as sns

import matplotlib.pyplot as plt

# 相関係数のヒートマップを表示

sns.heatmap(train.corr(),annot=True,fmt='.2f')
#年齢昇順

train["Age"].value_counts().sort_index()
#年齢昇順

test["Age"].value_counts().sort_index()
# 年齢を10歳区切りでビニング

age_list = [0, 10, 20, 30, 40, 50, 100]

age_span_name_list = ['0-10', '10-20', '20-30', '30-40', '40-50','50-']

train["Age_span"] = pd.cut(train["Age"], bins=age_list, labels=age_span_name_list)

test["Age_span"] = pd.cut(test["Age"], bins=age_list, labels=age_span_name_list)
fig=[]

#大枠設定

figure,ax = plt.subplots(4,2,figsize=(15,15),sharey=True)

#各項目一覧

fig.append(sns.countplot(x="Survived",data=train,ax=ax[0,0]))

fig.append(sns.countplot(x="Sex",data=train,ax=ax[0,1]))

fig.append(sns.countplot(x="Pclass",data=train,ax=ax[1,0]))

fig.append(sns.countplot(x="SibSp",data=train,ax=ax[1,1]))

fig.append(sns.countplot(x="Parch",data=train,ax=ax[2,0]))

fig.append(sns.countplot(x="Embarked",data=train,ax=ax[2,1]))

fig.append(sns.countplot(x="Age_span",data=train,ax=ax[3,0]))

#枠削除

figure.delaxes(ax[3, 1])



#データ数表示用関数

def output_count_to_graph(fig):

    for i, bar in enumerate(fig.patches):

        #サンプル数

        height = bar.get_height()

        if math.isnan(height):

            continue

        #テキスト横軸位置

        x_position = bar.get_width()/2+bar.get_x()

        #テキスト縦軸位置

        y_position = height/2

        #テキスト出力

        fig.text(

            x_position,

            y_position,

            '{}'.format(int(height)),

            ha='center',

            va='bottom', 

            fontweight='bold', 

            fontsize=15)

        

for i in range(7):

    output_count_to_graph(fig[i])
#各カラム別に生死を可視化

fig=[]

#大枠設定

figure,ax = plt.subplots(3,2,figsize=(15,15),sharey=True)

# 余白設定

plt.subplots_adjust(wspace=0.2, hspace=0.4)



#グラフ作成

fig.append(sns.countplot(x="Sex",hue="Survived",data=train,ax=ax[0,0]))

fig.append(sns.countplot(x="Pclass",hue="Survived",data=train,ax=ax[0,1]))

fig.append(sns.countplot(x="SibSp",hue="Survived",data=train,ax=ax[1,0]))

fig.append(sns.countplot(x="Parch",hue="Survived",data=train,ax=ax[1,1]))

fig.append(sns.countplot(x="Embarked",hue="Survived",data=train,ax=ax[2,0]))

fig.append(sns.countplot(x="Age_span",hue="Survived",data=train,ax=ax[2,1]))



for i in range(6):

    output_count_to_graph(fig[i])



# 凡例(anchor)を右上に置く処理

fig[2].legend_._loc=1

fig[3].legend_._loc=1
#ユニーク数を表示

train.nunique()
test.nunique()
#欠損値の確認

train.isnull().sum()
#テスト用データも確認

test.isnull().sum()
Fare_mean = test["Fare"].mean()

Cabin_mode = train["Cabin"].mode()

Embarked_mode = train["Embarked"].mode()[0]

Age_median = train["Age"].median()



print(f"Fare平均値：{Fare_mean}")

print(f"Embarked最頻値：{Embarked_mode}")

print(f"Age中央値：{Age_median}")



train["Embarked"].fillna(Embarked_mode,inplace=True)

train["Age"].fillna(Age_median,inplace=True)



test["Fare"].fillna(Fare_mean,inplace=True)

test["Embarked"].fillna(Embarked_mode,inplace=True)

test["Age"].fillna(Age_median,inplace=True)



train["Age_span"] = pd.cut(train["Age"], bins=age_list, labels=age_span_name_list)

test["Age_span"] = pd.cut(test["Age"], bins=age_list, labels=age_span_name_list)
#説明変数

X_column = ["Pclass","Sex","SibSp","Parch","Fare","Embarked","Age_span"]

#目的変数

y_column = ["Survived"]



#訓練データ

X = train.loc[:,X_column]

y = train.loc[:,y_column]



#テストデータ

test_X = test.loc[:,X_column]
#ダミーコード化(ワンホットエンコーディング)

X = pd.get_dummies(X)

test_X = pd.get_dummies(test_X)
X.head()
#ホールドアウト用モジュール

from sklearn.model_selection import train_test_split

#ホールドアウト

# 7:3に分けて進める

train_X, val_X, train_y, val_y = train_test_split(X,

                                                  y, 

                                                  test_size=0.3, 

                                                  random_state=1)
#決定木を使う

from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(min_samples_leaf = 0.01,

                               max_depth = 5,

                               random_state=1)

model = model.fit(train_X,train_y)

predict = model.predict(val_X)



#評価用データの正解率の表示

accuracy_score(val_y, predict)
import graphviz

from sklearn.tree import export_graphviz

dot_data = export_graphviz(

                        model,

                        feature_names = X.columns,

                        filled = True,

                        rounded = True,

                        out_file=None

                    )

graphviz.Source(dot_data)
#混同行列の表示(注意)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(val_y, predict,labels=[1,0])

cm_labeled = pd.DataFrame(cm, columns=["生存と予測","死亡と予測"], index=["生存","死亡"])

cm_labeled
#提出用データの作成

test_y = model.predict(test_X)
#サンプル提出データの確認

sample_submission.head()
submission = pd.DataFrame(

    {"PassengerId": test["PassengerId"], "Survived": test_y}

)
#提出

submission.to_csv("submission.csv",index=False)