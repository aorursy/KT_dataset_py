# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
df=data.drop(["Id"], axis=1)  # dropping column "Id"
df.plot()
plt.show()
df.plot(x="SepalLengthCm",
         y="PetalLengthCm")
plt.show()
df.iloc[:20,1].plot(kind="bar")  # getting first 20 rows of column "SepalWidthCm"
plt.show()

df.iloc[:20].plot.bar()  # getting first 20 rows of all columns
plt.show()
df.iloc[:20].plot.bar(stacked=True)  # getting first 20 rows of all columns
plt.show()
df.iloc[:20].plot.barh(stacked=True)  # getting first 20 rows of all columns
plt.show()
df.plot.hist(figsize=(10,4),
              alpha=0.5)
plt.show()
df.plot.hist(figsize=(10,4),
              alpha=0.5,
              stacked=True,
              bins=25)
plt.show()
df.iloc[:,0].plot.hist(cumulative=True) # getting all rows and first column
plt.show()
df.hist()
plt.show()
df.hist(color='r', alpha=0.5, bins=50, figsize=(7,5))
plt.show()
df.hist(by="Species",
         bins=5,
         alpha=0.8,
         figsize=(7,5))
plt.show()
df.plot.box()
plt.show()
color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange',
         'medians': 'DarkBlue', 'caps': 'Gray'}
df.plot.box(color=color, sym="b.")
plt.show()
df.plot.box(vert=False,
             positions=[1, 4, 5, 6])
plt.show()
df.boxplot()
plt.show()
df.boxplot(by="Species",
            figsize=(10,5))
plt.show()
df.groupby("Species").boxplot(figsize=(15,8))
plt.show()
df.boxplot(column=["SepalLengthCm", "PetalLengthCm"],
            by="Species",
            figsize=(10,5))
plt.show()
df.plot.area()
plt.show()
df.plot.area(stacked=False) 
plt.show()
df.plot.scatter(x="SepalLengthCm",
                 y="SepalWidthCm")
plt.show()
ax=df.plot.scatter(x="SepalLengthCm",
                     y="SepalWidthCm",
                     color="b",
                    label="Sepals")
df.plot.scatter(x="PetalLengthCm",
                 y="PetalWidthCm",
                 color="r",
                 label="Petals",
                 ax=ax)
plt.xlabel("LenghtCm")  # changing label of x axis
plt.ylabel("WidthCm")   # changing label of y axis
plt.show()
ax=data.plot.scatter(x="Id",
                     y="SepalLengthCm",
                     color="b",
                     label="SepalLengthCm",
                     alpha=0.5)

data.plot.scatter(x="Id",
                  y="SepalWidthCm",
                  color="r",
                  label="SepalWidthCm",
                  alpha=0.5,
                  ax=ax)

data.plot.scatter(x="Id",
                  y="PetalLengthCm",
                  color="g",
                  label="PetalLengthCm",
                  alpha=0.5,
                  ax=ax)

data.plot.scatter(x="Id",
                  y="PetalWidthCm",
                  color="y",
                  label="PetalWidthCm",
                  alpha=0.5,
                  ax=ax)
plt.ylabel("Length & Width")
plt.show()
data.plot.scatter(x="Id",
                  y="SepalWidthCm", 
                  c="PetalLengthCm",
                  s=100)     # s --> scatter size
plt.show()
data.plot.scatter(x="Id",
                  y="SepalWidthCm", 
                  s=data["PetalWidthCm"]*100)     
plt.show()
data.plot.hexbin(x="PetalLengthCm",
                y="PetalWidthCm",
                gridsize=20)
plt.show()
data.plot.hexbin(x="PetalLengthCm",
                y="PetalWidthCm",
                C="SepalLengthCm",
                gridsize=20)
plt.show()

# In plot : color density shows the Sepal Lengths of at each Petal Lenght and Petal Width cross.
# We can say that: sepal lenght ranges from 4 to 8 cm.
data.plot.hexbin(x="PetalLengthCm",
                y="PetalWidthCm",
                C="SepalLengthCm",
                reduce_C_function=np.sum,
                gridsize=20)
plt.show()

# In this plot: color density shows the sum of Sepal Length values at each x and y cros.
# For example: samples that having petal length around 1.5 cm and petal width of around 0.25 cm; sum of Sepal Length ıf those samples is around 85. 
# getting species by groupby and sum method
df.groupby("Species").sum().plot.pie(subplots=True, 
                                     figsize=(20,20),
                                     legend=False)

plt.show()
df.Species.value_counts().plot.pie()   # getting species count by value_counts() method
plt.show()
df.Species.value_counts().plot.pie(legend=True,
                                  labels=["Virginica","Versicolor","Setosa"],
                                  colors=["red","blue","orange"],
                                  autopct="%.2f",
                                  fontsize=10,
                                  figsize=(5,5),
                                  rotatelabels=True,
                                  labeldistance=0.1,
                                  pctdistance=0.8)   
plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='hist')
plt.show()
from pandas.plotting import radviz
radviz(df,"Species")
plt.show()