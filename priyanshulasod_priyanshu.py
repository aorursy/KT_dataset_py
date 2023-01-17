

import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/complete.csv")

df
df = df.drop("ConfirmedcasesIndia",axis=1)

df = df.drop("ConfirmedcasesForeign",axis=1)

df
df.keys()
new_data = df[df["StateaAndUT"] == "Maharashtra"]

new_data
x=new_data["TotalConfirmedcases"]

y=new_data["Cured"]
x=new_data.iloc[:,5:6].values
LR=LinearRegression()

LR.fit(x,y)
y_pred=LR.predict(x)

y_pred

print(y)

print(x)

print(y_pred)

plt.figure(figsize=(20,10)) 

plt.scatter(x,y)

plt.plot(x, y_pred, color="red")

plt.xlabel("TotalConfirmedcases")

plt.ylabel("Cured")

plt.show()