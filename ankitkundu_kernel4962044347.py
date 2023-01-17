import numpy as np

import pandas as pd

import  matplotlib.pyplot as plt

import seaborn as sns

from  sklearn import datasets

from sklearn.cluster import KMeans
df = pd.read_csv('Iris.csv') 
df.describe()
df.info()
df['Species'].value_counts()
sns.countplot(df['Species'])
import pandas as pd

df = pd.read_csv('Iris.csv')

df.plot()
from sklearn import datasets
sns.pairplot(df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])
sns.heatmap(df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].corr(),annot=True)
sns.countplot(df['SepalLengthCm'])
sns.countplot(df['SepalWidthCm'])
sns.countplot(df['PetalLengthCm'])
sns.countplot(df['PetalWidthCm'])
df.head()
df.tail()
sns.lmplot('Id','SepalLengthCm',data=df, hue='Species',palette='Set1',height=6,aspect=1,fit_reg=False)
sns.lmplot('SepalLengthCm','SepalWidthCm',data=df, hue='Species',palette='Set1',height=6,aspect=1,fit_reg=False)
sns.lmplot('PetalLengthCm','PetalWidthCm',data=df, hue='Species',palette='Set1',height=6,aspect=1,fit_reg=False)
kmeans = KMeans(n_clusters=3)
kmeans.fit(df.drop('Species',axis=1))
df['klabels'] = kmeans.labels_

df.head()
centers = kmeans.cluster_centers_

centers
f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2,

                             sharey = True, figsize = (10,6))



ax1.set_title('K Means (K = 3)')

ax1.scatter(x = df['Id'], y = df['SepalWidthCm'], 

            c = df['klabels'], cmap='rainbow')

ax1.scatter(x=centers[:, 0], y=centers[:,2],

            c='black',s=300, alpha=0.5);

 

ax2.set_title("Original")

ax2.scatter(x = df['Id'], y = df['SepalWidthCm'], 

            c = df['PetalLengthCm'], cmap='rainbow')
for k in range(1, 10):

    kmeans = KMeans(n_clusters=k).fit(df.drop('Species',axis=1))
sum_square = {}

sum_square[k] = kmeans.inertia_ 
plt.plot(list(sum_square.keys()), list(sum_square.values()),

         linestyle ='-', marker = 'H', color = 'g',

         markersize = 8,markerfacecolor = 'b')
kmeans = KMeans(n_clusters=3)

sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,'SepalLengthCm').add_legend()
sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,'SepalWidthCm').add_legend()
sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,'PetalLengthCm').add_legend()
sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,'PetalWidthCm').add_legend()
sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,'Id').add_legend()
sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,'klabels').add_legend()
print("MEAN , MEDIAN , STANDARD DEVIATION")
df_setosa = df.loc[df['Species']=='Iris-setosa']
print(np.mean(df_setosa['PetalLengthCm']))
print(np.mean(df_setosa['PetalWidthCm']))
print(np.mean(df_setosa['SepalLengthCm']))
print(np.mean(df_setosa['SepalWidthCm']))
print(np.mean(df_versicolor['PetalWidthCm']))
df_versicolor = df.loc[df['Species']=='Iris-versicolor']
print(np.mean(df_versicolor['PetalLengthCm']))
print(np.mean(df_versicolor['SepalWidthCm']))
print(np.mean(df_versicolor['SepalLengthCm']))
print(np.mean(df_versicolor['Id']))
df_virginica = df.loc[df['Species']== 'Iris-virginica']
print(np.mean(df_virginica['SepalWidthCm']))
print(np.mean(df_virginica['SepalLengthCm']))
print(np.mean(df_virginica['PetalWidthCm']))
print(np.mean(df_virginica['PetalLengthCm']))
print(np.mean(df_virginica['Id']))
print(np.std(df_setosa['PetalLengthCm']))
print(np.std(df_setosa['PetalWidthCm']))
print(np.std(df_setosa['SepalLengthCm']))
print(np.std(df_setosa['SepalWidthCm']))
print(np.std(df_setosa['Id']))
print(np.std(df_versicolor['Id']))
print(np.std(df_versicolor['SepalLengthCm']))
print(np.std(df_versicolor['SepalWidthCm']))
print(np.std(df_versicolor['PetalLengthCm']))
print(np.std(df_versicolor['PetalWidthCm']))
print(np.std(df_virginica['Id']))
print(np.std(df_virginica['SepalWidthCm']))
print(np.std(df_virginica['SepalLengthCm']))
print(np.std(df_virginica['PetalLengthCm']))
print(np.std(df_virginica['PetalWidthCm']))
print(np.median(df_setosa['Id']))
print(np.median(df_setosa['PetalLengthCm']))
print(np.median(df_setosa['PetalWidthCm']))
print(np.median(df_setosa['SepalLengthCm']))
print(np.median(df_setosa['SepalWidthCm']))
print(np.median(df_versicolor['Id']))
print(np.median(df_versicolor['PetalLengthCm']))
print(np.median(df_versicolor['PetalWidthCm']))
print(np.median(df_versicolor['SepalLengthCm']))
print(np.median(df_versicolor['SepalWidthCm']))
print(np.median(df_virginica['Id']))
print(np.median(df_virginica['SepalLengthCm']))
print(np.median(df_virginica['SepalWidthCm']))
print(np.median(df_virginica['PetalLengthCm']))
print(np.median(df_virginica['PetalWidthCm']))
sns.boxplot(x='Species',y='SepalLengthCm',data=df)
sns.boxplot(x='Species',y='Id',data=df)
sns.boxplot(x='Species',y='SepalWidthCm',data=df)
sns.boxplot(x='Species',y='PetalLengthCm',data=df)
sns.boxplot(x='Species',y='PetalWidthCm',data=df)
sns.boxplot(x='SepalWidthCm',y='SepalLengthCm',data=df)
sns.boxplot(x='PetalWidthCm',y='PetalLengthCm',data=df)
sns.pairplot(df,hue='Species')
print("---------------------------------------> THE END <-------------------------------------------------------")