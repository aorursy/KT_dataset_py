# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
df.info()
df.isnull().sum()
df.dtypes.tolist()
df.describe().T
df.columns.tolist()
# We need to focus on Target values.

df.target.value_counts()
df.thal.value_counts()
df.cp.value_counts()
df.oldpeak.value_counts().head()
plt.figure(figsize=(8,6))

sns.kdeplot(df.age)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(df.trestbps)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(df.thalach)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(df.oldpeak)

plt.show()
plt.figure(figsize=(16,6))

sns.countplot(df.age)

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(df.sex)

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(df.cp,palette="Blues")

plt.show()
plt.figure(figsize=(14,6))

sns.countplot(df.oldpeak,palette="Blues")

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(df.cp,palette="YlOrRd")

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(df.slope,palette="RdPu")

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(df.target,palette="Greens")

plt.show()
plt.figure(figsize=(10,6))

sns.jointplot(x= "age",y = "chol",data=df,kind="reg")

plt.show()
plt.figure(figsize=(10,6))

sns.jointplot(x= "age",y = "oldpeak",data=df,kind="reg")

plt.show()
plt.figure(figsize=(10,6))

sns.jointplot(x= "age",y = "thalach",data=df,kind="reg")

plt.show()
plt.figure(figsize=(14,6))

sns.barplot(x= "age",y = "target",data=df)

plt.show()
plt.figure(figsize=(11,6))

sns.barplot(x= "target",y = "slope",data=df)

plt.show()
plt.figure(figsize=(14,6))

sns.boxplot(x= "slope",y = "chol",hue="target",data=df)

plt.show()
plt.figure(figsize=(14,6))

sns.boxplot(x= "sex",y = "chol",hue="target",data=df)

plt.show()
sns.pairplot(df);
plt.figure(figsize=(14,6))

sns.violinplot(x= "sex",y = "chol",hue="target",data=df)

plt.show()
plt.figure(figsize=(14,6))

sns.violinplot(x= "slope",y = "chol",hue="target",data=df)

plt.show()
plt.figure(figsize=(14,6))

sns.violinplot(x= "thal",y = "chol",hue="target",data=df)

plt.show()
plt.figure(figsize=(14,6))

sns.stripplot(x='sex',y='chol',data=df, jitter=True, 

              hue='target', dodge=True)

plt.show()
plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True,square=True,linewidths=.5,cmap="Greens")

plt.show()
corr5 = df.corr()["target"].sort_values(ascending=False).head()   # to use later.



#[["target","cp","thalach","slope","restecg"]]
df.info()
y = df.target

X = df.drop("target",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25,random_state=0)
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

pred = logreg.predict(X_test)

print(accuracy_score(y_test,pred)*100)