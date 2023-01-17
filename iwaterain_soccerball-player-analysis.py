# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import warnings

import seaborn as sns



sns.set(style="darkgrid",font="SimHei", font_scale=1.5, rc={"axes.unicode_minus": False})

warnings.filterwarnings("ignore")
# 读取参数指定的文件，返回一个DataFrame类型的对象。

data = pd.read_csv("../input/fifa-20182019/FIFA_Player_2019.csv")

print(data.shape)

data.head(3)

# data.tail()

data.sample()
# 数据集中的列，并非都是我们分析所需要的，我们可以有选择性的进行加载，只加载我们需要的信息列（特征列）。

columns = ["Name", "Age", "Nationality", "Overall", "Potential", "Club", "Value", "Wage", "Preferred Foot",

          "Position", "Jersey Number", "Joined", "Height", "Weight", "Crossing", "Finishing",

          "HeadingAccuracy", "ShortPassing", "Volleys", "Dribbling", "Curve", "FKAccuracy", "LongPassing",

          "BallControl", "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance", "ShotPower", 

          "Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions", "Positioning", "Vision",

          "Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling",

          "GKKicking", "GKPositioning", "GKReflexes", "Release Clause"]

# 参数指定所要读取的列。

data = pd.read_csv("../input/fifa-20182019/FIFA_Player_2019.csv", usecols=columns)

data.head()
# 设置显示的最大列数。

pd.set_option("max_columns", 100)

data.head()
data.info()
data.isnull().sum(axis=0)
# 删除所有含有空值的行。修改原始数据。

data.dropna(axis=0, inplace=True)

data.isnull().sum()
data.describe()

sns.boxplot(data=data[["Age", "Overall"]])
data.drop_duplicates(inplace=True)

data.duplicated().sum()
def tran_height(height):

    v = height.split("'")

    return int(v[0]) * 30.48 + int(v[1]) * 2.54



def tran_weight(weight):

    v = int(weight.replace("lbs", ""))

    return v * 0.45



    data["Height"] = data["Height"].apply(tran_height)

    data["Weight"] = data["Weight"].apply(tran_weight)
data.head()
fig, ax = plt.subplots(1, 2)

fig.set_size_inches((18, 5))

sns.distplot(data[["Height"]], bins=50, ax=ax[0], color="g")

sns.distplot(data["Weight"], bins=50, ax=ax[1])
# 数量上对比

number = data["Preferred Foot"].value_counts()

print(number)

sns.countplot(x="Preferred Foot", data=data)
# 能力上对比

print(data.groupby("Preferred Foot")["Overall"].mean())

sns.barplot(x="Preferred Foot", y="Overall", data=data)
t = data.groupby(["Preferred Foot", "Position"]).size()

t = t.unstack()

t[t < 50] = np.NaN

t.dropna(axis=1, inplace=True)

display(t)
t2 = data[data["Position"].isin(t.columns)]

plt.figure(figsize=(18, 10))

sns.barplot(x="Position", y="Overall", hue="Preferred Foot", hue_order=["Left", "Right"], data=t2)
g = data.groupby("Club")

r = g["Overall"].agg(["mean", "count"])

r = r[r["count"] >= 20]

r = r.sort_values("mean", ascending=False).head(10)

display(r)

r.plot(kind="bar")
g = data.groupby("Nationality")

r = g["Overall"].agg(["mean", "count"])

r = r[r["count"] >= 50]

r = r.sort_values("mean", ascending=False).head(10)

display(r)

r.plot(kind="bar")
t = pd.to_datetime(data["Joined"])

t = t.astype(np.str)

join_year = t.apply(lambda item: int(item.split("-")[0]))

over_five_year = (2018 - join_year) >= 5

t2 = data[over_five_year]

t2 = t2["Club"].value_counts()

# display(t2)

t2.iloc[:15].plot(kind="bar")
data2 = pd.read_csv("../input/fifa-20182019/FIFA_WC2018_Player.csv")

data2.head()
t = data2["Birth Date"].str.split(".", expand=True)

# t

# t[0].value_counts().plot(kind="bar")

# t[1].value_counts().plot(kind="bar")

# t[2].value_counts().plot(kind="bar")

t[2].value_counts().sort_index().plot(kind="bar")
g = data.groupby(["Jersey Number", "Position"])

t = g.size()

# display(t)

t = t[t >= 100]

t.plot(kind="bar")
def to_numeric(item):

    item = item.replace("€", "")

    value = float(item[:-1])

    if item[-1] == "M":

        value *= 1000

    return value



data["Value"] = data["Value"].apply(to_numeric)

data["Wage"] = data["Wage"].apply(to_numeric)

data["Release Clause"] = data["Release Clause"].apply(to_numeric)

data.head()
# sns.scatterplot(x="Value", y="Wage", data=data)

# sns.scatterplot(x="Value", y="Release Clause", data=data)

sns.scatterplot(x="Value", y="Height", data=data)
plt.figure(figsize=(25, 25))

sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap=plt.cm.Greens)

plt.savefig("corr.png", dpi=100, bbox_inches="tight")
g = data.groupby("Position")

g["GKDiving"].mean().sort_values(ascending=False)

plt.figure(figsize=(15, 5))

sns.barplot(x="Position", y="GKDiving", data=data)
sns.scatterplot(x="Age", y="Overall", data=data)
data["Age"].corr(data["Overall"])
# 对一个数组进行切分，可以将连续值变成离散值。

# bins 指定区间数量（桶数）。bins如果为int类型，则进行等分。

# 此处的区间边界与为前开后闭。

# pd.cut(t["Age"], bins=4)

# 如果需要进行区间的不等分，则可以将bins参数指定为数组类型。

# 数组来指定区间的边界。

min_, max_ = data["Age"].min() - 0.5, data["Age"].max()

# pd.cut(t["Age"], bins=[min_, 20, 30, 40, max_])

# pd.cut 默认显示的内容为区间的范围，如果我们希望自定义内容(每个区间显示的内容)，可以通过labels参数

# 进行指定。

t = pd.cut(data["Age"], bins=[min_, 20, 30, 40, max_], labels=["弱冠之年", "而立之年","不惑之年", "知天命"])



t = pd.concat((t, data["Overall"]), axis=1)

# display(t)

g = t.groupby("Age")

display(g["Overall"].mean())

sns.lineplot(y="Overall", marker="*", ms=30, x="Age", data=t)