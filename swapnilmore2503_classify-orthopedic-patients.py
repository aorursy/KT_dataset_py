# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", '../input']).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/column_2C_weka.csv")

df.head()
df.info()
X = df.drop("class", axis=1)

y = df["class"]
le = LabelEncoder()

le.fit(y)

y = le.transform(y)

df1 = X

df1['class'] = y
colormap = plt.cm.plasma

plt.figure(figsize=(7,7))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df1.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 

            linecolor='white', annot=True)
sns.set_style("whitegrid")

plt.figure(figsize=(12,12))

sns.boxplot(data=df, orient="h")
sns.set(style="ticks")

sns.pairplot(df, hue="class", size = 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, min_samples_split=4)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

pred_train = clf.predict(X_train)

acc = accuracy_score(pred, y_test)

acc_train = accuracy_score(pred_train, y_train)

print(acc)

print(acc_train)
df3C = pd.read_csv("../input/column_3C_weka.csv")

df3C.head()
df3C.info()
X3C = df3C.drop("class", axis=1)

y3C = df3C["class"]
le3C = LabelEncoder()

le3C.fit(y3C)

y3C = le3C.transform(y3C)

df3C1 = X3C

df3C1['class'] = y3C
colormap = plt.cm.plasma

plt.figure(figsize=(7,7))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df3C1.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 

            linecolor='white', annot=True)
sns.set_style("whitegrid")

plt.figure(figsize=(12,12))

sns.boxplot(data=df3C, orient="h")
sns.set(style="ticks")

sns.pairplot(df3C, hue="class", size = 3)
X3C_train, X3C_test, y3C_train, y3C_test = train_test_split(X3C, y3C, test_size=0.2)
clf3C = RandomForestClassifier(n_estimators=100, min_samples_split=4)

clf3C.fit(X3C_train, y3C_train)
pred3C = clf3C.predict(X3C_test)

pred3C_train = clf3C.predict(X3C_train)

acc3C = accuracy_score(pred3C, y3C_test)

acc3C_train = accuracy_score(pred3C_train, y3C_train)

print(acc3C)

print(acc3C_train)