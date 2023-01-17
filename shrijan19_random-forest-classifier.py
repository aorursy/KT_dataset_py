# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,precision_score, auc, recall_score, f1_score,classification_report

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
figsize = (12,8)

fig ,ax = plt.subplots(figsize = figsize)

sns.heatmap(df.corr(), vmin=-1,vmax=1, center=0, annot=True, cmap="coolwarm", ax = ax);
df["quality"] = df["quality"].astype("str")
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "fixed acidity", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "volatile acidity", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "citric acid", data=df);
df.groupby(by=["quality"])["citric acid"].mean()
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "residual sugar", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "chlorides", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "free sulfur dioxide", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "total sulfur dioxide", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "density", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "pH", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "sulphates", data=df);
fig = plt.figure(figsize=(6,6))

sns.barplot(x = "quality", y = "alcohol", data=df);
df["quality"] = df["quality"].astype("int")

bins = (2, 6.5, 8)

group_names = [0,1]

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
df["quality"].value_counts()
sns.countplot(df["quality"]);
y = df["quality"].copy()

X = df.drop(columns=["quality"]).copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_RF = RandomForestClassifier(n_jobs=-1, n_estimators=50, verbose=1, random_state=42, class_weight="balanced")

model_RF.fit(X_train, y_train)

model_RF.score(X_test,y_test)
y_pred = model_RF.predict(X_test)

print(classification_report(y_test,y_pred))
from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10,10))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')

plt.grid()
pca_new = PCA(n_components=3)

x_new = pca_new.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.3, random_state=42)
model_RF = RandomForestClassifier(n_jobs=-1, n_estimators=50, verbose=1, random_state=42, class_weight="balanced")

model_RF.fit(X_train, y_train)

model_RF.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)