import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data=sns.load_dataset('titanic')

data
data.info()
#distribution plot

sns.distplot(data['age'])
#joint plot

sns.jointplot(x='age',y='fare',data=data)
plt.figure(figsize=(20,5))

sns.jointplot(x='age',y='survived',data=data)
iris=sns.load_dataset('iris')
sns.pairplot(iris,hue='species')

#understanding from visulization : Sentosa can be linearly sepearable
sns.lmplot(x='petal_width',y='petal_length',data=iris)
sns.scatterplot(x='petal_width',y='petal_length',data=iris,hue="species")
# sns.rugplot(data['fare'])

sns.distplot(data['fare'], rug=True, hist=False)
sns.barplot(x='sex',y='survived',data=data)

#Survival of female is more than male:(
sns.boxplot(x='sex',y='fare',data=data)
sns.set_style('ticks')

sns.countplot(x='pclass',data=data)
sns.violinplot(x='sex',y='age',data=data)
mat=data.corr()
mat
sns.heatmap(mat)  #annot=true
sns.clustermap(mat,annot=True)