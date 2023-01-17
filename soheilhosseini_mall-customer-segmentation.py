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
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import plotly.express as px



from sklearn.cluster import KMeans

customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers.head()
customers.info()
customers.describe()
# Distribution Plot

# plot a univariate distribution of observations



sns.distplot(customers['Age'], color='red')

plt.show()
# Distribution Plot

# plot a univariate distribution of observations



sns.distplot(customers['Annual Income (k$)'], color='red')

plt.show()
# Distribution Plot

# plot a univariate distribution of observations



sns.distplot(customers['Spending Score (1-100)'], color='red')

plt.show()



sns.pairplot(customers, hue='Gender')

plt.show()
#Finding relation between Age, Age,Annual income and Spening Score

fig, axs = plt.subplots(3, 3, figsize=(15,15))



L = ['Age', 'Annual Income (k$)','Spending Score (1-100)']

for i in L:

    for j in L:

            sns.regplot(x=i, y=j, data=customers, color='red', ax=axs[L.index(i),L.index(j)])
plt.figure(figsize = (16,6))

sns.countplot(x="Age", data=customers, hue='Gender', palette='plasma')

plt.show()
plt.figure(figsize = (18,6))

plt.xticks(rotation=90)



sns.countplot(x="Annual Income (k$)", data=customers, hue='Gender', palette='plasma')

plt.show()
plt.figure(figsize = (18,6))

plt.xticks(rotation=90)



sns.countplot(x="Spending Score (1-100)", data=customers, hue='Gender')

plt.show()
sns.heatmap(data=customers.corr(), annot=True, vmin=-1, vmax=1)

plt.show()
fig = px.pie(customers, names='Gender', title='distribution of Gender in the Mall')

fig.show()
sns.boxplot(x="Gender", y="Age", data=customers)

plt.show()
plt.figure(figsize=(18,5))

plt.title("Relation between Age and Spending score")

sns.barplot(x='Age', y='Spending Score (1-100)', palette='plasma', data=customers, ci=None)

plt.xticks(rotation=90)



plt.show()
np.sum(pd.isnull(customers))
from matplotlib.ticker import MaxNLocator



ax = plt.figure().gca()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))



wss =[]

X = customers[['Annual Income (k$)', 'Spending Score (1-100)']]



for k in range(1,20):

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    wss.append(kmeans.score(X))



plt.title('Elbow method: Annual Income (k$)vs Spending Score (1-100)')

plt.plot(range(1,20), np.abs(wss),  linewidth=3, color='red')

plt.xticks(np.arange(min(range(1,20)), max(range(1,20))+1, 1))

plt.xlabel('Number of Clusters')

plt.show()
### ?????

# How can plot this??



# K = 5

X = customers[['Annual Income (k$)', 'Spending Score (1-100)']]



kmeans = KMeans(n_clusters=5, random_state=0, n_jobs=-1).fit(X)

X_clustered = kmeans.transform(X)

labels = kmeans.labels_

centers = kmeans.cluster_centers_
clusters = pd.DataFrame (data = customers[['Annual Income (k$)', 'Spending Score (1-100)']], columns=['Annual Income (k$)', 'Spending Score (1-100)'])

clusters['labels']=labels
g = sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=clusters, palette='plasma',

                    size='labels', hue='labels', sizes=(30, 200), legend='full')



plt.scatter(x=centers[:,0], y=centers[:,1],marker='8',s=400, color='red')



g.legend(loc='center left', bbox_to_anchor=(1, .8))

plt.title('Annual Income(k$) vs Spending Score')





plt.show()
ax.plot([1, 2, 3])

ax.legend(['A simple line'])

from matplotlib.ticker import MaxNLocator



ax = plt.figure().gca()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))



wss =[]

X = customers[['Age', 'Spending Score (1-100)']]



for k in range(1,20):

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    wss.append(kmeans.score(X))



plt.title('Elbow method: Age vs Spending Score')

plt.plot(range(1,20), np.abs(wss),  linewidth=3, color='red')

plt.xticks(np.arange(min(range(1,20)), max(range(1,20))+1, 1))

plt.xlabel('Number of clusters')

plt.show()
# K = 4

X = customers[['Age', 'Spending Score (1-100)']]



kmeans = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(X)

labels = kmeans.labels_

centers = kmeans.cluster_centers_



clusters2 = pd.DataFrame (data = X, columns=['Age', 'Spending Score (1-100)'])

clusters2['labels']=labels
import matplotlib.patches as mpatches

#plt.legend(handles=[red_patch])



plt.figure(figsize=(8,8))

g = sns.scatterplot(x='Age',y='Spending Score (1-100)',data=clusters2, palette='plasma',

                    size='labels', hue='labels',  sizes=(40, 200), legend='full')



plt.scatter(x=centers[:,0], y=centers[:,1],marker='8',s=400, color='red')



g.legend(loc='center left', bbox_to_anchor=(1, .8))

plt.title('Age vs Spending Score')





plt.show()