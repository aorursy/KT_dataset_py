# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("../input/train.csv")
df.dtypes
df.describe()
df.head()
pd.pivot_table(df,index="Survived",margins=True)
df["Sex"]=pd.get_dummies(df["Sex"])
#0が女性、1が男性
pd.crosstab(df["Survived"],df["Ticket"],margins=True,normalize="all")
df=df.sort_values('Name',ascending=True)
df.head(14)
pd.pivot_table(df,values="Age",index="Sex", columns="Survived", aggfunc="median", dropna=True)
df.loc[(df["Sex"].values == 1) & (df["Survived"].values == 1) & (df["Age"].isnull()), "Age"] = df.query("Sex == 1 and Survived == 1")["Age"].mean()
df.loc[(df["Sex"].values == 1) & (df["Survived"].values == 0) & (df["Age"].isnull()), "Age"] = df.query("Sex == 1 and Survived == 0")["Age"].mean()
df.loc[(df["Sex"].values == 0) & (df["Survived"].values == 1) & (df["Age"].isnull()), "Age"] = df.query("Sex == 0 and Survived == 1")["Age"].mean()
df.loc[(df["Sex"].values == 0) & (df["Survived"].values == 0) & (df["Age"].isnull()), "Age"] = df.query("Sex == 0 and Survived == 0")["Age"].mean()

#df.loc[(df["Survived"].values == 0) & (df["Age"].isnull()), "Age"] = df.query("Survived == 0")["Age"].mean()
#df.loc[(df["Survived"].values == 1) & (df["Age"].isnull()), "Age"] = df.query("Survived == 1")["Age"].mean()

df.describe()
%matplotlib inline
sns.set()
sns.pairplot(df,hue="Sex",vars=["Survived", "Fare", "Pclass","Age"],dropna=True, kind="reg", size=3, markers="x")
sns.set()
sns.jointplot(df["Survived"],df["Pclass"], kind="reg",color="blue")
pd.crosstab(df["Survived"],df["Pclass"])
df.corr()
train_X = df[['Survived', 'Fare', 'Age']].dropna()
train_Y = df['Survived']
# train_X["Age"]
plt.scatter(train_X['Age'],train_X['Fare'],vmin=0, vmax=1,c=train_Y)
# カラーバーを表示
plt.colorbar()
from sklearn.neighbors import KNeighborsClassifier
train_X = train_X.dropna(how='any')

knn = KNeighborsClassifier(n_neighbors=6) # インスタンス生成。n_neighbors:Kの数
knn.fit(train_X, train_Y)                 # モデル作成実行


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cross_validation import train_test_split # trainとtest分割用

# X_patal = pd.DataFrame()
df_train = df[['Survived', 'Age', 'Sex']].dropna()

X = pd.DataFrame()
X["Age"] = df_train["Age"]
X["Sex"]  = df_train["Sex"] 
Y = df_train['Survived']

# normalize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=3) 
X_train
X_test
Y_train
Y_test

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=6) # インスタンス生成。n_neighbors:Kの数

knn.fit(X_train, Y_train)                 # モデル作成実行

Y_pred = knn.predict(X_test)              # 予測実行
metrics.accuracy_score(Y_test, Y_pred)    # 予測精度計測
