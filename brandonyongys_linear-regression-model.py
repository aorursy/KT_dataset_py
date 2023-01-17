# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation

import seaborn as sns # data visualisation

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cereal.csv")

manufacturer = {"A":"American Home Food Products",

               "G":"General Mills",

               "K":"Kelloggs",

               "N":"Nabisco",

               "P":"Post",

               "Q":"Quaker Oats",

               "R":"Ralston Purina"}



df["Manufacturer"] = df["mfr"].map(manufacturer)

df.head()
df.describe()
criteria_1 = df["carbo"] < 0

criteria_2 = df["sugars"] < 0

criteria_3 = df["potass"] < 0



df[criteria_1 | criteria_2 | criteria_3]
df[df["weight"] < 1].head()
cols = ['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']



for i in cols:

    df[i] = (df[i]/df["weight"]).round(0).astype(int)
df[df["weight"] < 1].head()
df["Manufacturer"].value_counts()
fig, ax = plt.subplots(figsize = (7,7))

sns.boxplot(y = "Manufacturer", x = "rating", data = df.sort_values("Manufacturer"))
criteria_1 = df["Manufacturer"] == "Kelloggs"

criteria_2 = df["rating"] > 90



df[criteria_1 & criteria_2]
def scatter_this(macro,y):

    fig, ax = plt.subplots(figsize = (4,4))

    sns.regplot(x = macro, y = y, data = df)

    plt.title("%s vs %s" %(y, macro))
df.columns
cols = df.columns[3:15]



for i in cols:

    scatter_this(i,"rating")
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
X = df[["calories","protein","fat","sugars"]]

y = df["rating"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)
r2_4 = r2_score(y_test, y_pred)

r2_4
X = df[["calories","protein","fat","sugars","carbo"]]

y = df["rating"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)
r2_5 = r2_score(y_test, y_pred)

r2_5
X = df[["calories","protein","fat","sugars","potass"]]

y = df["rating"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)
r2_6 = r2_score(y_test, y_pred)

r2_6
X = df[['calories', 'protein', 'fat', 'sodium', 'fiber','carbo', 'sugars', 'potass', 'vitamins']]

y = df["rating"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)
r2_all = r2_score(y_test, y_pred)

r2_all