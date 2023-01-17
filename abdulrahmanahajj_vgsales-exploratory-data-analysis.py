import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline 

data=pd.read_csv("../input/videogamesales/vgsales.csv")

data
data.shape
data.info()
data.describe()
data.isnull().sum()
best_Plat=data.Platform.value_counts().nlargest(5)

best_Plat
data[data.NA_Sales > 10.00]
data[data.Global_Sales==data.Global_Sales.max()]
gsh=data[data.Global_Sales==data.Global_Sales.max()]

gsh.Platform
data[data.Global_Sales==data.Global_Sales.min()]
gsl=data[data.Global_Sales==data.Global_Sales.min()]

gsl.Platform.value_counts()
jhs=data[(data.Genre==data.Genre)&(data.JP_Sales==data.JP_Sales.max())]

jhs[["Name","Platform","Genre","Global_Sales"]]
import seaborn as sns

#get correlations of each features in dataset

corr_matrix = data.corr()

top_corr_features = corr_matrix.index

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.pairplot(data)
best_Platform=data.Platform.value_counts().nlargest(10)

best_Platform


gradable_counts=best_Platform

# Plot a pie chart

gradable_counts.plot(kind='pie', title='Platform best 10', figsize=(8,6),autopct='%.2f %%') 



plt.legend()

 



plt.show()
best_Genre=data.Genre.value_counts().nlargest(10)

best_Genre
gradable_counts1 = best_Genre

# Plot a pie chart

gradable_counts1.plot(kind='pie', title='Genre best 10', figsize=(8,6),autopct='%.2f %%') 



plt.legend()

 



plt.show()