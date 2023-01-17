import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler





%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df = df.drop('CustomerID', axis=1)

df.columns = ['Gender', 'Age', 'Income', 'Score']

df.head(5)
# basic information about columns

df.info()
# count null values in each column

df.isnull().sum()
plt.figure(figsize=(20,4))

plt.subplot(1,3,1)

sns.distplot(df.Age[df['Gender']=='Female'], color='orange', hist=False, kde=True, label='Female')

sns.distplot(df.Age[df['Gender']=='Male'], color='blue', hist=False, kde=True, label='Male')

plt.title('Age')



plt.subplot(1,3,2)

sns.distplot(df.Income[df['Gender']=='Female'], color='orange', hist=False, kde=True, label='Female')

sns.distplot(df.Income[df['Gender']=='Male'], color='blue', hist=False, kde=True, label='Male')

plt.title('Income')



plt.subplot(1,3,3)

sns.distplot(df.Score[df['Gender']=='Female'], color='orange', hist=False, kde=True, label='Female')

sns.distplot(df.Score[df['Gender']=='Male'], color='blue', hist=False, kde=True, label='Male')

plt.title('Score')



plt.show()

plt.figure(figsize=(20,5))

plt.subplot(1,3,1)

sns.boxplot(x=df.Gender, y=df.Age)

plt.title('Age')



plt.subplot(1,3,2)

sns.boxplot(x=df.Gender, y=df.Income)

plt.title('Income')



plt.subplot(1,3,3)

sns.boxplot(x=df.Gender, y=df.Score)

plt.title('Score')



plt.show()
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)

sns.scatterplot(x=df.Age, y=df.Income, hue=df.Gender)

plt.title('Age vs Income')



plt.subplot(1,3,2)

sns.scatterplot(x=df.Age, y=df.Score, hue=df.Gender)

plt.title('Age vs Score')



plt.subplot(1,3,3)

sns.scatterplot(x=df.Income, y=df.Score, hue=df.Gender)

plt.title('Income vs Score')



plt.show()
plt.figure(figsize=(20,8))

plt.subplot(2,1,1)

sns.barplot(x=df.Age, y=df.Income, hue=df.Gender, ci=0)

plt.title('Income by Age')

plt.xlabel('')



plt.subplot(2,1,2)

sns.barplot(x=df.Age, y=df.Score, hue=df.Gender, ci=0)

plt.title('Score by Age')



plt.show()
# one hot encoding, keeping just male column, so 1 = male, 0 = female

df = pd.get_dummies(df, columns=['Gender'], drop_first=True)       #Thanks to Evan for suggestion

df = df.rename(columns={'Gender_Male':'Gender'})
# create new dataframe with transformed values

df_t = df.copy()



ss = StandardScaler()

df_t['Age'] = ss.fit_transform(df['Age'].values.reshape(-1,1))

df_t['Income'] = ss.fit_transform(df['Income'].values.reshape(-1,1))

df_t['Score'] = ss.fit_transform(df['Score'].values.reshape(-1,1))
plt.figure(figsize=(20,10))



plt.subplot(2,3,1)

sns.scatterplot(x=df.Age, y=df.Income, hue=df.Gender)

plt.title('Age vs Income')



plt.subplot(2,3,2)

sns.scatterplot(x=df.Age, y=df.Score, hue=df.Gender)

plt.title('Age vs Score')



plt.subplot(2,3,3)

sns.scatterplot(x=df.Income, y=df.Score, hue=df.Gender)

plt.title('Income vs Score')



plt.subplot(2,3,4)

sns.scatterplot(x=df_t.Age, y=df_t.Income, hue=df_t.Gender)

plt.title('Age vs Income - Tranformed')



plt.subplot(2,3,5)

sns.scatterplot(x=df_t.Age, y=df_t.Score, hue=df_t.Gender)

plt.title('Age vs Score - Tranformed')



plt.subplot(2,3,6)

sns.scatterplot(x=df_t.Income, y=df_t.Score, hue=df_t.Gender)

plt.title('Income vs Score - Tranformed')



plt.show()
# untransformed data

inertia = []

for i in range(1, 12):

    km = KMeans(n_clusters=i).fit(df)

    inertia.append(km.inertia_)



# transformed data

inertia_t = []

for i in range(1, 12):

    km = KMeans(n_clusters=i).fit(df_t)

    inertia_t.append(km.inertia_)



# plot results

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

sns.lineplot(x=range(1,12), y=inertia)

plt.title('KMeans inertia on original data')



plt.subplot(1,2,2)

sns.lineplot(x=range(1,12), y=inertia_t)

plt.title('KMeans inertia on transformed data')



plt.show()
# collect cluster labels as well as cluster centers

clusters = [2,3,4,5]

cluster_centers = {}



for c in clusters:

    km = KMeans(n_clusters=c).fit(df[['Age', 'Income', 'Score', 'Gender']])

    df['cluster' + str(c)] = km.labels_

    cluster_centers[str(c)] = km.cluster_centers_
plt.figure(figsize=(20,15))

for i, c in enumerate(clusters):

    plt.subplot(2,2,i+1)

    sns.scatterplot(df.Income, df.Score, df['cluster' + str(c)], s=120, palette=sns.color_palette("hls", c))

    sns.scatterplot(cluster_centers[str(c)][:,1], cluster_centers[str(c)][:,2], color='black', s=300)

    plt.title('Number of clusters: ' + str(c))

    

plt.show()
plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

sns.scatterplot(df.Income, df.Score, df['cluster5'], s=120, palette=sns.color_palette("hls", 5))

plt.title('Income vs Score')

   

plt.subplot(1,3,2)

sns.scatterplot(df.Age, df.Score, df['cluster5'], s=120, palette=sns.color_palette("hls", 5))

plt.title('Age vs Score')



plt.subplot(1,3,3)

sns.scatterplot(df.Age, df.Income, df['cluster5'], s=120, palette=sns.color_palette("hls", 5))

plt.title('Age vs Income')



plt.show()