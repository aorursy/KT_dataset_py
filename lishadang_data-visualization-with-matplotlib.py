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
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn import datasets

import seaborn as sns
x = [2,5,4,6,1,2,3,5,4]

plt.plot(x)

plt.show()
x = np.arange(0,10)

y = np.random.randint(22,42,10)

color = np.random.rand(10)

plt.plot(x, y, c = "black")

plt.scatter(x,y, c = ["red", "green", "pink", "yellow", "blue", "red", "green", "pink", "yellow", "blue"])

plt.xlabel("Day")

plt.ylabel("Temp")

plt.title("Day Vs Temp")

plt.show()
df = pd.read_csv("../input/nba-players-data/all_seasons.csv")

df.head(4)
plt.hist(df.age,bins = 10)
plt.figure(figsize=(15,6))

sns.countplot(data=df, x = "age")

plt.show()
cm = [

    [132, 12, 21],

    [23, 432, 32]

]

sns.heatmap(cm, annot=True)


x = np.random.randint(10,40, 6)

y = np.random.randint(2010, 2020, 6)

explode = np.random.randint(0,1,6)

explode = explode.astype(float)

explode[x.argmin()] = 0.2

plt.pie(x, labels=y, explode=explode, autopct="%1.1f%%")

plt.show()
iris = datasets.load_iris()

iris
irisData = pd.DataFrame(iris.data, columns=iris.feature_names)

irisData["Species"] = iris.target
sns.FacetGrid(irisData, hue="Species", height=5).map(plt.scatter, 

            "petal width (cm)" , "sepal width (cm)")
df = pd.read_csv("../input/california-housing-prices/housing.csv")

plt.figure(figsize=(20,12))

df.plot(kind = "scatter", x = "longitude",

        y = "latitude",c = "median_house_value", cmap = plt.get_cmap("jet"),

       s = df["population"]/100)
sns.FacetGrid(df, hue="ocean_proximity", height=5).map(plt.scatter, 

            "longitude" , "latitude")