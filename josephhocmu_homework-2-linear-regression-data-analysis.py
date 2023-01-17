import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model as lm

import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns



print(os.listdir("../input"))
d1=pd.read_csv("../input/2017-18_teamBoxScore.csv")

df = d1[["teamAbbr","teamFIC40","teamRslt"]]

df = df.replace({'Win': 1, 'Loss': 0})

df.head(10)
df.isnull().values.any()
df.hist()
print(df)
gs = df.loc[df['teamAbbr'] == 'GS']

gs.head()
gs.corr()
inputDF = gs[["teamFIC40"]]

outcomeDF = gs['teamRslt']

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



print(model.intercept_, model.coef_)
d2=pd.read_csv("../input/2017-18_playerBoxScore.csv")

df = d2[["playHeight","playFG%"]]

df.columns = ["height","fgPercent"]

print(df)
df.isnull().values.any()
df.hist()
df.corr()
h = df[["height"]]

f = df[["fgPercent"]]
model = lm.LinearRegression()

results = model.fit(h,f)

print(model.intercept_, model.coef_)
plt.scatter(h,f, alpha=0.50)

plt.show()
y = model.predict(h)

print(y)
plt.scatter(h,f)

plt.plot(h,y, color="blue")

plt.show()
print(df)
mean = (df.sum(axis = 0, skipna = True))/26109

print(mean)
m = pd.DataFrame(data=df["height"])

a = pd.DataFrame(data=df["height"])

for col in m.columns:

    m[col].values[:] = 78.984488

s = np.sum((m[:]-a[:])**2)

print(s)
d1=pd.read_csv("../input/2017-18_playerBoxScore.csv")

d1.head(10)
plt.boxplot(d1["playPTS"])

plt.show()
plt.boxplot(d1["playAST"])

plt.show()
plt.boxplot(d1["playSTL"])

plt.show()
plt.boxplot(d1["playBLK"])

plt.show()
plt.boxplot(d1["playTO"])

plt.show()
a1=pd.read_csv("../input/2017-18_playerBoxScore.csv")

a1.head(10)
plt.scatter(a1["playHeight"], a1["playWeight"], alpha=0.50)

plt.show()
plt.scatter(a1["playHeight"], a1["playPTS"], alpha=0.50)

plt.show()
plt.scatter(a1["playHeight"], a1["playAST"], alpha=0.50)

plt.show()
plt.scatter(a1["playHeight"], a1["playBLK"], alpha=0.50)

plt.show()