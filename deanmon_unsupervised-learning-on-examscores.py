import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings 

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



df = pd.read_csv("../input/StudentsPerformance.csv")

df.rename(index = int, columns = {"race/ethnicity":"ethnicity"},inplace=True)
df.head()
df.tail()
scores = df.loc[:,["math score","reading score","writing score"]]

scores.rename(index = int, columns = {"math score":"mthscore","reading score":"readscr","writing score":"writingscr"},inplace=True)
import seaborn as sns



sns.pairplot(data = df,hue="ethnicity",palette="Set1")

plt.show()
from sklearn.cluster import KMeans
wcss = []

for i in range(1,15):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(scores)

    wcss.append(kmeans.inertia_)





plt.plot(range(1,15),wcss,"-o")

plt.xlabel("Number of K value")

plt.ylabel("WCSS Value")

plt.show()
kmeans2 = KMeans(n_clusters = 5)

clusters = kmeans2.fit_predict(scores)

scores["examscores"] = clusters





plt.scatter(scores.mthscore[scores.examscores == 0],scores.readscr[scores.examscores == 0],color="red")

plt.scatter(scores.mthscore[scores.examscores == 1],scores.readscr[scores.examscores == 1],color="green")

plt.scatter(scores.mthscore[scores.examscores == 2],scores.readscr[scores.examscores == 2],color="blue")

plt.scatter(scores.mthscore[scores.examscores == 3],scores.readscr[scores.examscores == 3],color="black")

plt.scatter(scores.mthscore[scores.examscores == 4],scores.readscr[scores.examscores == 4],color="brown")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(scores, method = "ward")

dendrogram(merg,leaf_rotation=90)

plt.xlabel("Scores")

plt.ylabel("Distances")

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")

cluster = hc.fit_predict(scores)



scores["examscores"] = cluster

plt.scatter(scores.mthscore[scores.examscores == 0],scores.readscr[scores.examscores == 0],color="red")

plt.scatter(scores.mthscore[scores.examscores == 1],scores.readscr[scores.examscores == 1],color="green")

plt.scatter(scores.mthscore[scores.examscores == 2],scores.readscr[scores.examscores == 2],color="blue")

plt.scatter(scores.mthscore[scores.examscores == 3],scores.readscr[scores.examscores == 3],color="black")

plt.scatter(scores.mthscore[scores.examscores == 4],scores.readscr[scores.examscores == 4],color="brown")

plt.show()