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

sns.set(style="darkgrid", font="SimHei", rc={"axes.unicode_minus": False})

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/CompletedDataset.csv")

print(data.shape)

data.head()
# data.info()

data.isnull().sum(axis=0)
# data.describe()

sns.boxplot(data=data["Precipitation"])
data.duplicated().sum()
t = data[["City", "AQI"]].sort_values("AQI")

display(t.iloc[:5])

sns.barplot(x="City", y="AQI", data=t.iloc[:5])
display(t.iloc[-5:])

sns.barplot(x="City", y="AQI", data=t.iloc[-5:])
display(data["Coastal"].value_counts())

sns.countplot(x="Coastal", data=data)
sns.swarmplot(x="Coastal", y="AQI", data=data)
display(data.groupby("Coastal")["AQI"].mean())

sns.barplot(x="Coastal", y="AQI", data=data)
sns.boxplot(x="Coastal", y="AQI", data=data)
sns.violinplot(x="Coastal", y="AQI", data=data)
sns.violinplot(x="Coastal", y="AQI", data=data, inner=None)

sns.swarmplot(x="Coastal", y="AQI", color="g", data=data)
# data.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(data.corr(), cmap=plt.cm.RdYlGn, annot=True, fmt=".2f")
sns.scatterplot(x="Longitude", y="Latitude", hue="AQI", palette=plt.cm.RdYlGn_r, data=data)
data["AQI"].mean()
all = np.random.normal(loc=30, scale=50, size=10000)

mean_arr = np.zeros(2000)

for i in range(len(mean_arr)):

    mean_arr[i] = np.random.choice(all, size=50, replace=False).mean()

display(mean_arr.mean())

sns.kdeplot(mean_arr, shade=True)
# 定义标准差

scale = 50

# 定义数据。

x = np.random.normal(0, scale, size=100000)

# 定义标准差的倍数，倍数从1到3。

for times in range(1, 4):

    y = x[(x >= -times * scale) & (x <= times * scale)] 

    print(len(y) / len(x))
mean = data["AQI"].mean()

std = data["AQI"].std()

print(mean, std)

t = (mean - 71) / (std / np.sqrt(len(data)))

print(t)
from scipy import stats



stats.ttest_1samp(data["AQI"], 71)
mean - 1.96 * (std / np.sqrt(len(data))), mean + 1.96 * (std / np.sqrt(len(data)))
print(stats.ttest_1samp(data["AQI"], 70.64536585461275))

print(stats.ttest_1samp(data["AQI"], 80.02336479554205))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X = data.drop(["City","AQI"], axis=1)

y = data["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

lr = LinearRegression()

lr.fit(X_train, y_train)

display(lr.coef_)

display(lr.intercept_)

y_hat = lr.predict(X_test)

display(lr.score(X_train, y_train))

display(lr.score(X_test, y_test))
plt.figure(figsize=(15, 5))

plt.plot(y_test.values, "-r", label="真实值")

plt.plot(y_hat, "-g", label="预测值")

plt.legend()

plt.title("线性回归预测结果")
from sklearn.linear_model import LogisticRegression



X = data.drop(["City","Coastal"], axis=1)

y = data["Coastal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

lr = LogisticRegression(C=0.0001)

lr.fit(X_train, y_train)

display(lr.coef_)

display(lr.intercept_)

y_hat = lr.predict(X_test)

display(lr.score(X_train, y_train))

display(lr.score(X_test, y_test))
plt.figure(figsize=(15, 5))

plt.plot(y_test.values, marker="o", c="r", ms=8, ls="", label="真实值")

plt.plot(y_hat, marker="x", color="g", ms=8, ls="", label="预测值")

plt.legend()

plt.title("逻辑回归预测结果")
probability = lr.predict_proba(X_test)

print(probability[:10])

print(np.argmax(probability, axis=1))

index = np.arange(len(X_test))

pro_0 = probability[:, 0]

pro_1 = probability[:, 1]

tick_label = np.where(y_test == y_hat, "O", "X")

plt.figure(figsize=(15, 5))

# 绘制堆叠图

plt.bar(index, height=pro_0, color="g", label="类别0概率值")

# bottom=x，表示从x的值开始堆叠上去。

# tick_label 设置标签刻度的文本内容。

plt.bar(index, height=pro_1, color='r', bottom=pro_0, label="类别1概率值", tick_label=tick_label)

plt.legend(loc="best", bbox_to_anchor=(1, 1))

plt.xlabel("样本序号")

plt.ylabel("各个类别的概率")

plt.title("逻辑回归分类概率")

plt.show()