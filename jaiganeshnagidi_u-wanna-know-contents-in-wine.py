import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv("/kaggle/input/wine-quality/winequality.csv")

df.head()
#checking for null values

df.isna().sum()
plt.figure(figsize=(10,6))



sns.set_style('whitegrid')



sns.countplot(x='color',data=df)
corr=df.corr()

plt.figure( figsize=(16,8))



hm = sns.heatmap(corr, 

                 cmap='YlGnBu',

                 annot=True, 

                 fmt='.2f')       # String formatting code to use when adding annotation
sns.set_style('whitegrid')

cols = ['density',  'residual sugar',  'total sulfur dioxide', 'free sulfur dioxide', 'fixed acidity','color']

sns.pairplot(df[cols], kind="scatter", hue="color", markers=["o", "D"], palette="Set2")

plt.show()

sns.set_style('ticks')

fig = plt.figure(figsize=(10,8))

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.93, wspace=0.3)



ax = fig.add_subplot(1,1,1)

ax.set_xlabel("Sulphates")

ax.set_ylabel("Frequency") 



g = sns.FacetGrid(data=df, 

                  hue='color', 

                  palette={"red": "r", "white": "blue"})



g.map(sns.distplot, 'sulphates', 

      kde=True, bins=15, ax=ax)



ax.legend(title='Wine Type')

plt.close(2)
fig = plt.figure(figsize=(10,8))

title = fig.suptitle("citric acid  Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.93, wspace=0.3)



ax = fig.add_subplot(1,1,1)

ax.set_xlabel("citric acid")

ax.set_ylabel("Frequency") 



g = sns.FacetGrid(data=df, 

                  hue='color', 

                  palette={"red": "r", "white": "blue"})



g.map(sns.distplot, 'citric acid', 

      kde=True, bins=15, ax=ax)



ax.legend(title='Wine Type')

plt.close(2)
sns.set_style('whitegrid')



f, ax = plt.subplots(1, 1, figsize=(14, 6))

f.suptitle('Wine Quality - Alcohol Content', fontsize=14)



sns.violinplot(data=df,  

            x="quality", 

            y="alcohol", 

            ax=ax)



ax.set_xlabel("Wine Quality",size=12,alpha=0.8)

ax.set_ylabel("Wine Alcohol %",size=12,alpha=0.8)
f, (ax) = plt.subplots(1, 1, figsize=(14, 6))

f.suptitle('Wine Quality - Sulphates Content', fontsize=14)



sns.boxplot(data=df,

               x="quality", 

               y="sulphates",   

               ax=ax)



ax.set_xlabel("Wine Quality",size=12,alpha=0.8)

ax.set_ylabel("Wine Sulphates",size=12,alpha=0.8)
plt.figure(figsize=(16,8))

sns.set_style('darkgrid')

ax = sns.kdeplot(df['sulphates'],

                 df['alcohol'],   

                 cmap="YlOrBr", 

                 shade=True, shade_lowest=False)

plt.show()