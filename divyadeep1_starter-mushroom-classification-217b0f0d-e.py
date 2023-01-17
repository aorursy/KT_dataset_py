import os

import pandas as pd
os.listdir("../input")
df = pd.read_csv("../input/mushrooms.csv")
df.head()
df.shape
from sklearn import preprocessing
enc = preprocessing.OrdinalEncoder()

enc.fit(df)
tf = enc.transform(df)
type(tf)
x = tf[:,1:]
y = tf[:,0]
y
from sklearn.linear_model import LinearRegression
c = LinearRegression()
c.fit(x,y)
c.score(x,y)
from sklearn import svm

c2 = svm.SVC()

c2.fit(x,y)

c2.score(x,y)