#import python libraries

import pandas as pd 

import os

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/habit-loop-tracker/habitss.csv"  ,usecols=range(1,8))

df = df.replace(2, 1)
df.head(10)
df.describe()
df2 = df[df == 1].count()

df2.plot(kind='bar',figsize=(10,7))

plt.show()
df.corr("pearson").sort_values('Out on time', ascending=False)
f, ax = plt.subplots(figsize=(10, 10))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)



plt.show()
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#using elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

from kmodes.kmodes import KModes

X = df.iloc[:,[2, 6]].values
kmodes = KModes(n_clusters = 3, init="Huang", n_init = 10, verbose=1)

clusters = kmodes.fit_predict(X)

df['clusters'] = clusters.astype(int)

df['clusters'] = df['clusters'].map({1: 'cluster_1', 2: 'cluster_2',0:'cluster_3'})
cl1 = df[df['clusters'] == 'cluster_1']

cl1f = df[df['clusters'] == 'cluster_1']

cl1 = cl1[cl1 == 1].count()

cl1.plot(kind='bar',figsize=(10,7))

plt.show()
cl2 = df[df['clusters'] == 'cluster_2']

cl2f = df[df['clusters'] == 'cluster_2']

cl2 = cl2[cl2 == 1].count()

cl2.plot(kind='bar',figsize=(10,7))

plt.show()
cl3 = df[df['clusters'] == 'cluster_3']

cl3f = df[(df['clusters'] == 'cluster_3')]

cl3 = cl3[cl3 == 1].count()

cl3.plot(kind='bar',figsize=(10,7))

plt.show()
cluster_all = pd.concat([cl2f,cl1f,cl3f])

cluster_all.groupby('clusters').count().plot.bar(stacked=True, figsize=(10,7))

plt.show()
cluster_all[cluster_all==1].count().plot.bar(stacked=True, figsize=(10,7))

plt.show()
# cluster_all.set_index('clusters').groupby('clusters').plot.bar(stacked=True, figsize=(10,7))

# plt.show()
# cluster_all.pivot(index='clusters')

cc = cluster_all[cluster_all == 1].count()

cc.plot.bar(stacked=True, figsize=(10,7))

plt.show()
cluser1 = cl1f[cl1f==1].count().tolist()

cluser2 = cl2f[cl2f==1].count().tolist()

cluser3 = cl3f[cl3f==1].count().tolist()
cluser1
# Heights of bars1 + bars2

bars = np.add(cluser1, cluser2).tolist()

 

# The position of the bars on the x-axis

r = [0,1,2,3,4,5,6,7]

 

# Names of group and bar width

names = ['Morning Meditation', 'Morning readings', 'Midday Sleep', 'Arthour',

       'Before sleep readings', 'Before sleep meditation', 'Out on time']

barWidth = 1

plt.figure(figsize=(15,6))

# Create brown bars

plt.bar(r, cluser1,  edgecolor='white', width=barWidth)

# Create green bars (middle), on top of the firs ones

plt.bar(r, cluser2, bottom=cluser1,  edgecolor='white', width=barWidth)

# Create green bars (top)

plt.bar(r, cluser3, bottom=bars,  edgecolor='white', width=barWidth)

# Custom X axis

plt.title('Clustering Habits')

plt.legend()

plt.xticks(r, names, fontweight='bold')

plt.tick_params(axis='x', labelrotation=45)

plt.xlabel("Habits",labelpad=14)

plt.legend(["The Supervisor", "The Idealist","The Visionary"]);

# Show graphic

plt.show()
