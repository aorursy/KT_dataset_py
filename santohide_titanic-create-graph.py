#まずは、ライブラリインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv") #正直使わない...
test = pd.read_csv("../input/titanic/test.csv")
#trainデータは今回特に使っていきたいもの
train = pd.read_csv("../input/titanic/train.csv")
train.head()
test.head()
train2 = train[["Pclass","Fare","Age"]]
#nan値は今回削除
train2 = train2.dropna()
train2.isnull().sum()
fig,axes = plt.subplots(1,3,figsize=(20,5))
for i in range(1,4):
    data = train2[train2["Pclass"] == i]
    axes[i-1].scatter(data["Age"], data["Fare"])
plt.show()
#Seabornの導入と簡単なdistplotから
import seaborn as sns

x = np.random.normal(size = 100)
sns.distplot(x, kde = 1, rug =0)
#kde:カーネル密度近似関数(曲線のこと)、推定した値
#rug:ラグプロットの有無
#bins:値の刻み方
plt.show()
#次に散布図を作成する
#sns.set()によってグラフの見え方が変わる
sns.set()
sns.jointplot(train["Age"], train["Fare"])
plt.show()
sns.pairplot(train[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived', plot_kws={'alpha':0.5})
#一応hueパラメータを用いるとデータを色分けできる
plt.show()
#簡単な集計
train.groupby("Pclass").mean()
train.groupby("Pclass").sum()
#locを用いると、特定のデータみることができるそうです
for i, Pclass in train.groupby("Pclass"):
    print("Fare mean of Pclass{} is {}".format(i, Pclass["Fare"].mean()))
train.groupby("Pclass").mean()["Fare"]

