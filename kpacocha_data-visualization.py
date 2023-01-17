#Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

import pylab

from statsmodels.graphics.gofplots import qqplot

%matplotlib inline

print("done")
#Read data:

data = pd.read_csv('../input/irisdataset/iris.csv')
data.plot(figsize=(10,5))

plt.axis([-5, 155, 0, 9])

plt.show()
data.plot(subplots=True,figsize=(10,5),color="skyblue")

plt.show()
data['sepal.length'].plot(kind='box', color='green',figsize=(7,7))

plt.title('Length of Sepal')

plt.show()
data.boxplot(by='variety',figsize=(10,10))

plt.show()
plt.subplots(figsize=(7,5))

sns.boxplot(x="variety", y="sepal.length", data=data)

plt.xlabel("Variety")

plt.ylabel("Length of Sepal")

plt.title("Box plots for Variety")
data.boxplot()

plt.grid(b=None)

plt.show()
plt.scatter(x=data['sepal.length'],y=data['sepal.width'])

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')

plt.title('Length and width of sepal')

plt.show()
from pandas.plotting import scatter_matrix

scatter_matrix(data,alpha=0.2,figsize=(8,8),diagonal='kde')

plt.show()
sns.pairplot(data,size=2)
sns.pairplot(data,hue="variety",size=2)
sns.swarmplot(x='variety',y='sepal.length',data=data)
sns.violinplot(x='variety', y='sepal.length', data=data,palette='cool')
#QQplot - very useful to check normality of variable

qqplot(data['sepal.length'], loc = 0, scale = 1, line='s')

pylab.title('QQ plot')

pylab.show()
f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="black", fmt= '.2f',ax=ax)

plt.show()
data['sepal.length'].value_counts().sort_index().plot.line()

plt.show()
data['sepal.length'].value_counts().sort_index().plot.area()

plt.show()
data['sepal.length'].plot.hist(title='Histogram for Sepal Length', bins=10, color='purple',fontsize=15, figsize=(6,4))

plt.xlabel('Sepal Length')

plt.ylabel('Frequency')

plt.show()
sns.distplot(data['sepal.length'],kde=False,fit=norm)

sns.despine(bottom=True,left=True)
data.hist(figsize=(10,5), color='purple')

plt.grid(b=None)

plt.show()
data['variety'].value_counts().sort_index().plot.bar(color='pink')
sns.jointplot(x='sepal.length', y='sepal.width',data=data)
sns.jointplot(x='sepal.length', y='sepal.width',data=data, kind='reg')
sns.jointplot(x='sepal.length', y='sepal.width',data=data, kind='hex')
data2 = pd.read_csv('../input/suicidedataset/master-2.csv')
data2['suicides_no'].plot()
sns.pairplot(data2,hue="sex",size=2)
plot = data2['age'].value_counts().plot(kind = "pie", figsize = (5,5))

plot.set_title("Suicides number by age")
plt = data2[['sex', 'suicides_no']].groupby('sex').mean().suicides_no.plot('bar')

plt.set_title('Suicides number by sex')

plt.set_ylabel('Suicides number')

plt.set_xlabel('Sex')