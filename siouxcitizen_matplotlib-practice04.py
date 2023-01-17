import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#ビデオゲームのデータを表示・可視化
data = pd.read_csv("../input/videogamesales/vgsales.csv")

#最初の5件を表示
data.head()
plt.figure(figsize=(15,12))
Yearly_Totals = data.Year.value_counts()
sns.barplot(y = Yearly_Totals.values, x = Yearly_Totals.index)
plt.xticks(rotation=90)
#アイリスのデータを表示・可視化
data2 = pd.read_csv("../input/iris/Iris.csv")

#最初の5件を表示
data2.head()
sns.pairplot(data2)
sns.pairplot(data2, hue='Species')
sns.countplot(x="Species", data=data2)
sns.countplot(y="Species", data=data2)
#タイタニックのデータを表示・可視化
data3 = pd.read_csv("../input/titanic-solution-a-beginners-guide/train.csv")

#最初の5件を表示
data3.head()
sns.countplot(x="Pclass", data=data3)
sns.countplot(y="Pclass", data=data3)
sns.countplot(x="Pclass", hue="Sex",data=data3)
sns.countplot(x="Pclass", order=[2, 1, 3], data=data3)
sns.countplot(x="Pclass", hue="Sex", hue_order=['female','male'], data=data3)
sns.countplot(x="Pclass", hue="Sex", palette="Set1", data=data3)
plt.figure(figsize=(15,12))
sns.set_style("whitegrid")
sns.countplot(x="Pclass", hue="Sex", palette="Set1", data=data3)