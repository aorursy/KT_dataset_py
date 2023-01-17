import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv("../input/random-linear-regression/train.csv")

train.head()
print(train.dtypes)
import matplotlib.pyplot as plt

train = pd.read_csv("../input/random-linear-regression/train.csv")

print(train.isnull().sum())

train.fillna(" ")

x=train["x"].tolist()

y=train["y"].tolist()

plt.scatter(x,y)