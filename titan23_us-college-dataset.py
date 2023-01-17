import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.read_csv("../input/College.csv")



df.head()
sns.set_style('whitegrid')

sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',

           palette='coolwarm',height=6,aspect=1,fit_reg=False)





sns.set_style('whitegrid')

sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',

           palette='coolwarm',height=6,aspect=1,fit_reg=False)

sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)

g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)



df[df['Grad.Rate'] > 100]

df['Grad.Rate']['Cazenovia College'] = 100

df[df['Grad.Rate'] > 100]

sns.set_style('darkgrid')

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)

g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

X = df.drop(['Private', 'Unnamed: 0'],axis=1)


# Using K means clustering



from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss,"bx-")

plt.grid()

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
