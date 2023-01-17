# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/googleplaystore.csv")
data.head(10)
data.describe()
data.info()

data["Type"] = data["Type"].map({"Free":0, "Paid":1})
data.head()
data["Size"] = data["Size"].map(lambda x:x.rstrip("M"))
data.head()
data["Installs"] = data["Installs"].map(lambda x: x.rstrip("+"))
data["Price"] = data["Price"].map(lambda x:x.lstrip("$"))
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
data.head()
X= data[["Reviews","Size", "Type", "Installs", "Price"]]
y = data["Rating"]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
