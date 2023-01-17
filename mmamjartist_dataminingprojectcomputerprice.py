

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

mydata=pd.read_csv("../input/basic-computer-data-set/Computers.csv")



# Any results you write to the current directory are saved as output.
#Print the first 5 rows of the dataframe.

mydata.head()









mydata.head()

mydata['cd'] = mydata['cd'].eq('yes').astype(int)
mydata.head()
mydata['multi'] = mydata['multi'].eq('yes').astype(int)
mydata.head()


mydata['premium'] = mydata['premium'].eq('yes').astype(int)
mydata.head()
## gives information about the data types,columns, null value counts, memory usage etc

## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html

mydata.info(verbose=True)
## basic statistic details about the data (note only numerical columns would be displayed here unless parameter include="all")

## for reference: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe

mydata.describe()
mydata.describe().T
p = mydata.hist(figsize = (20,20))
mydata.shape
from mlxtend.plotting import plot_decision_regions

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

import missingno as msno

p=msno.bar(mydata)


print(mydata.cd.value_counts())

p=mydata.cd.value_counts().plot(kind="bar")


print(mydata.multi.value_counts())

p=mydata.multi.value_counts().plot(kind="bar")
p=sns.pairplot(mydata, hue = 'cd')
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(mydata.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap