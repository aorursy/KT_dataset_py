import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
df.shape
df.head()
df=df.drop('Id',axis=1)
df.describe()
df['Species'].unique()
df.groupby("Species").describe()
df.columns
df.info()
import seaborn as sns
sns.set_style("whitegrid")
sns.pairplot(df,hue='Species',size=3)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = df.drop("Species",axis=1)
Y = df["Species"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33)
reg = LogisticRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
print(metrics.accuracy_score(Y_test, Y_pred))
X = df.drop(["Species","SepalLengthCm","SepalWidthCm"],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33)
reg = LogisticRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
print(metrics.accuracy_score(Y_test, Y_pred))
print(X_train.shape)
print(X.head())
