import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

x = np.random.randn(20, 1)+10
y = x*2 + np.random.randn(20, 1)

data = np.array([x,y])
data.shape
data = data.reshape(2, 20)
data.shape
df =pd.DataFrame(data.T, columns=["a", "b"])
df.head()
plt.scatter(df["a"], df["b"])
plt.title("Scatter plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
x = np.arange(7, 13)
y = x*2 

plt.scatter(df["a"], df["b"])
plt.title("Scatter plot")
plt.plot(x,y, "r--", lw=4, alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["$y = 2 \cdot x$", "Points"])
plt.show()
df["c"] = np.abs(np.random.randn(20))*20
df.head()
fig, ax = plt.subplots()

ax.scatter(df["a"], df["b"], color="r", edgecolors="k", label="b", s=50)
ax.scatter(df["a"], df["c"], color="g", edgecolors="k", label="c", s=50)
ax.title.set_text("Scatter plot")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
df.corr()
import seaborn as sns
plt.figure(figsize=(7,4)) 
sns.heatmap(df.corr(),annot=True)
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.array(df["a"]), np.array(df["b"]), test_size=0.3)

print(X_train.shape)
print(y_train.shape)
from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
X_train = X_train.reshape(-1, 1)
X_train.shape
X_test = X_test.reshape(-1, 1)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
lr.coef_
lr.intercept_
X_pred = np.arange(6, 13)
X_pred = X_pred.reshape(-1,1)

lr.predict(X_pred)
fig, ax = plt.subplots()

ax.scatter(df["a"], df["b"], color="r", edgecolors="k", label="data", s=50)
ax.plot(X_pred, lr.predict(X_pred), color="k", label="prediction")
ax.title.set_text("Scatter plot")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
df["d"] = df["a"]**2+np.random.randn(20)
df.head()
X_train, X_test, y_train, y_test = train_test_split(np.array(df["a"]), np.array(df["d"]), test_size=0.2)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

lr2 = linear_model.LinearRegression()
lr2.fit(X_train, y_train)

lr2.score(X_test, y_test)
print(lr2.coef_, lr2.intercept_)
X_pred = np.arange(6, 13)
X_pred = X_pred.reshape(-1,1)



fig, ax = plt.subplots()

ax.scatter(df["a"], df["d"], color="r", edgecolors="k", label="data", s=50)
ax.plot(X_pred, lr2.predict(X_pred), color="k", label="prediction")
ax.title.set_text("Scatter plot")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
X_0 = X_train[:]
y_0 = y_train[:]

X_1 = X_test[:]
y_1 = y_test[:]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pr = make_pipeline(PolynomialFeatures(2), Ridge())

pr.fit(X_0, y_0)

X_pred = np.arange(6, 13)
X_pred = X_pred.reshape(-1,1)



fig, ax = plt.subplots()

ax.scatter(df["a"], df["d"], color="r", edgecolors="k", label="data", s=50)
ax.plot(X_pred, pr.predict(X_pred), color="k", label="prediction")
ax.title.set_text("Scatter plot")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
pr.score(X_test, y_test)
X_0 = X_train[:]
y_0 = y_train[:]

X_1 = X_test[:]
y_1 = y_test[:]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pr = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())

pr.fit(X_0, y_0)

X_pred = np.arange(6, 13)
X_pred = X_pred.reshape(-1,1)



fig, ax = plt.subplots()

ax.scatter(df["a"], df["d"], color="r", edgecolors="k", label="data", s=50)
ax.plot(X_pred, pr.predict(X_pred), color="k", label="prediction")
ax.title.set_text("Scatter plot")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
pr.score(X_1, y_1)
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target


predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
