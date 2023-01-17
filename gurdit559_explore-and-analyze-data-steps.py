# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/iris/Iris.csv")

df.head(10)
df.tail(10)
df.shape  #there are 150 rows and 6 columns
df.columns
df.isnull().sum()  #check if there are any null values
df["Species"].value_counts()
df.info()
df.describe()
sns.relplot(x="SepalLengthCm",y="SepalWidthCm",hue="Species",kind="scatter",data=df)
sns.boxplot(x="Species",y="SepalLengthCm",data=df)
setosa=(df[df["Species"]=="Iris-setosa"])

setosa
versicolor=df[df["Species"]=="Iris-versicolor"]

versicolor
virginica=df[df["Species"]=="Iris-virginica"]

virginica
print("SepalLengthCm range min to max for iris-setosa : ",min(setosa["SepalLengthCm"]),"to",max(setosa["SepalLengthCm"]))

print("SepalWidthCm range min to max for iris-setosa : ",min(setosa["SepalWidthCm"]),"to",max(setosa["SepalWidthCm"]))

print("PetalLengthCm range min to max for iris-setosa : ",min(setosa["PetalLengthCm"]),"to",max(setosa["PetalLengthCm"]))

print("PetalWidthCm range min to max for iris-setosa : ",min(setosa["PetalWidthCm"]),"to",max(setosa["PetalWidthCm"]))

print("SepalLengthCm range min to max for iris-versicolor : ",min(versicolor["SepalLengthCm"]),"to",max(versicolor["SepalLengthCm"]))

print("SepalWidthCm range min to max for iris-versicolor : ",min(versicolor["SepalWidthCm"]),"to",max(versicolor["SepalWidthCm"]))

print("PetalLengthCm range min to max for iris-versicolor : ",min(versicolor["PetalLengthCm"]),"to",max(versicolor["PetalLengthCm"]))

print("PetalWidthCm range min to max for iris-versicolor : ",min(versicolor["PetalWidthCm"]),"to",max(versicolor["PetalWidthCm"]))

print("SepalLengthCm range min to max for iris-virginica : ",min(virginica["SepalLengthCm"]),"to",max(virginica["SepalLengthCm"]))

print("SepalWidthCm range min to max for iris-virginica : ",min(virginica["SepalWidthCm"]),"to",max(virginica["SepalWidthCm"]))

print("PetalLengthCm range min to max for iris-virginica : ",min(virginica["PetalLengthCm"]),"to",max(virginica["PetalLengthCm"]))

print("PetalWidthCm range min to max for iris-virginica : ",min(virginica["PetalWidthCm"]),"to",max(virginica["PetalWidthCm"]))

data={"Species":["Iris-setosa","iris-versicolor","iris-virginica"],"sepalLengthRange":["4.3-5.8","4.9-7.0","4.9-7.9"],"SepalWidthRange":["2.3-4.4","2.0-3.4","2.2-3.8"],"PetalLengthRange":["1.0-1.9","3.0-5.1","4.5-6.9"],"PetalWidthRange":["0.1-0.6","1.0-1.8","1.4-2.5"]}

nd=pd.DataFrame(data)

nd
sns.violinplot(x="Species",y="PetalLengthCm",data=df)
a=sns.FacetGrid(df,col="Species")

a.map(sns.distplot,"SepalLengthCm")
a=sns.FacetGrid(df,hue="Species",size=3)

a.map(sns.distplot,"SepalLengthCm")

b=sns.FacetGrid(df,hue="Species",size=3)

b.map(sns.distplot,"SepalWidthCm")
c=sns.FacetGrid(df,hue="Species",size=3)

c.map(sns.distplot,"PetalLengthCm")

d=sns.FacetGrid(df,hue="Species",size=3)

d.map(sns.distplot,"PetalWidthCm")
counts,bin_edges=np.histogram(setosa["PetalLengthCm"],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.title("iris-setosa")

plt.xlabel("Petal Length")

plt.ylabel("frequency")
counts,bins_edges=np.histogram(versicolor["PetalLengthCm"],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bins_edges[1:],pdf)

plt.plot(bins_edges[1:],cdf)

plt.title("iris-versicolor")

plt.xlabel("Petal Length")

plt.ylabel("Frequency")

plt.show()
counts,bins_edges=np.histogram(virginica["PetalLengthCm"],bins=10,density=True)

pdf=counts/sum(counts)

cdf=np.cumsum(pdf)

plt.plot(bins_edges[1:],pdf)

plt.plot(bins_edges[1:],cdf)

plt.title("iris-verginica")

plt.xlabel("Petal Length")

plt.ylabel("Frequency")

plt.show()
df.corr()
heat=sns.heatmap(df.corr(),annot=True)

heat
sns.pairplot(df,hue="Species",size=3) 