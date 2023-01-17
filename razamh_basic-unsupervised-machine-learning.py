# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nba = pd.read_csv("/kaggle/input/unsupervised-ml/nba_2013.csv")

nba.head()
# make simple data then I can use large data for clustering

nba["pos"].unique()
point_gaurd = nba[nba.pos == "PG"].copy()

point_gaurd.head()
# Create a two new column name point per garud and atr

point_gaurd["ppg"] = point_gaurd["pts"]/point_gaurd["g"]
point_gaurd = point_gaurd[point_gaurd["tov"] != 0].copy()
point_gaurd["atr"] = point_gaurd["ast"]/point_gaurd["tov"]
point_gaurd.head()
point_gaurd[["ppg","atr"]].head(5)
point_gaurd.index

point_gaurd.shape[0]
num_cluster = 5

num_cluster1 = 2

np.random.seed(1)

random_initial_value = np.random.choice(point_gaurd.index, size = num_cluster)

random_initial_value
centroids = point_gaurd.loc[random_initial_value]

centroids
# {0:[12.535211,1.670455], 1 :[16.671429,1.785425]}



def centroids_to_dic(centroids):

    dictionary = {}

    counter = 0

    for index, row in centroids.iterrows():

        coordinates = [row["ppg"], row["atr"]]

        dictionary[counter] = coordinates

        counter += 1

    return dictionary
centroids_dic = centroids_to_dic(centroids)

centroids_dic
point_gaurd.iloc[0][["ppg", "atr"]]
euclidean_distance = np.sqrt(np.array([[12.53-13.0986]])**2 +np.array([[1.67- 2.504]])**2 )

euclidean_distance
def calculate_distance(q,p):

    distance = 0

    for i in range(len(q)):

        distance += (q[i] - p[i])**2

    return np.sqrt(distance)
q = [12.53, 1.67]

p = [13.0986,2.504]

print(calculate_distance(q,p))
row1_distances = []

for q1 in centroids_dic.values():

#     print(q1)

    distance = calculate_distance(q1,p)

    row1_distances.append(distance)
row1_distances
minimum = min(row1_distances)



row1_distances.index(minimum)
def assign_cluster(row):

    row_distances = []

    p = [row["ppg"], row["atr"]]

    for q in centroids_dic.values():

        distance = calculate_distance(q,p)

        row_distances.append(distance)

    minimum = min(row_distances)

    cluster = row_distances.index(minimum)

    return cluster
assign_cluster(point_gaurd.iloc[0])
assign_cluster(point_gaurd.iloc[81])
point_gaurd["cluster"] = point_gaurd.apply(lambda row :assign_cluster(row),axis = 1)
point_gaurd["cluster"].value_counts()
def visualize_cluster(df, num_cluster):

    colors = ["b","g", "r", "y", "k"]

    for i in range(num_cluster):

        cluster = df[df["cluster"] == i]

        plt.scatter(cluster["ppg"], cluster["atr"], c = colors[i])

    plt.show()
visualize_cluster(point_gaurd, num_cluster)
cluster_0 = point_gaurd[point_gaurd["cluster"] == 0]

ppg = cluster_0["ppg"].mean()

atr = cluster_0["atr"].mean()



cen = {0: [ppg, atr]}

cen
def recalculate_cent(df):

    dictionary = {}

    for i in range(num_cluster):

        cluster = point_gaurd[point_gaurd["cluster"] == i]

        ppg = cluster["ppg"].mean()

        atr = cluster["atr"].mean()

        dictionary[i] = [ppg,atr]

    return dictionary
centroids_dic = recalculate_cent(point_gaurd)

centroids_dic
point_gaurd["cluster"] = point_gaurd.apply(lambda row :assign_cluster(row),axis = 1)
visualize_cluster(point_gaurd, num_cluster)
kmeans =  KMeans(n_clusters=num_cluster, random_state = 1)

kmeans.fit(point_gaurd[["ppg", "atr"]])

kmeans.labels_
point_gaurd["cluster"] = kmeans.labels_
visualize_cluster(point_gaurd, num_cluster)
point_gaurd.head(10)
votes = pd.read_csv("/kaggle/input/unsupervised-ml/114_congress.csv")
votes.head()
votes["party"].value_counts()
kmean = KMeans(n_clusters= 2, random_state = 1)

kmean.fit(votes.iloc[:,3:])
labels = kmean.labels_

labels
votes["group"] = labels
votes[["party", "group"]]
pd.crosstab(votes["group"], votes["party"])
boolean = (votes["party"] == "D") & (votes["group"] == 0)

votes[boolean]