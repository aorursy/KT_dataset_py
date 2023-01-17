# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path



from sklearn.cluster import KMeans



from tqdm.notebook import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
KAGGLE_PATH = Path('/kaggle/input/trends-assessment-prediction')



# subject-levels

#SCN - Sub-cortical Network

#ADN - Auditory Network

#SMN - Sensorimotor Network

#VSN - Visual Network

#CON - Cognitive-control Network

#DMN - Default-mode Network

#CBN - Cerebellar Network

SL = ['SCN','ADN','SMN','VSN','CON','DMN','CBN']
sfnc = pd.read_csv(KAGGLE_PATH/'fnc.csv') #.drop('Id',axis=1)



sfnc_group_clusters = pd.DataFrame(sfnc.pop('Id'))



cols = sfnc.columns



sfnc.shape
group_columns={}



for c in cols:

    groupkey = c.split('(')[0] + '_' + c.split('(')[1].split('_',-1)[2]

    

    group_col_list = group_columns.get(groupkey)

    

    if group_col_list == None:

        group_col_list = [c]

    else:

        group_col_list += [c] 

    

    group_columns[groupkey] = group_col_list



# test

group_columns['SCN_SCN']
from sklearn.metrics import silhouette_score



#https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb





for gc in tqdm(group_columns):



    # TODO find optimal cluster number

    #n_clusters = 3

    

    X = sfnc[group_columns[gc]].values

    

    

    sil = []

    kmax = 5



    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2

    for k in range(2, kmax+1):

        kmeans = KMeans(n_clusters = k).fit(X)

        labels = kmeans.labels_

        sil.append(silhouette_score(X, labels, metric = 'euclidean'))

        

    

    n_clusters = np.argmax(sil) + 2 #index starts from zero and cluster start from 2 thus index 0 means 2 

    

    print(n_clusters)

    

    #break



    kmeans = KMeans(n_clusters=n_clusters, random_state=2020).fit(X)

    sfnc_group_clusters[gc] = kmeans.labels_



    #preds = kmeans.predict(sfnc[group_columns[gc]].head().values)  # ==> same as kmeans.labels

    #kmeans.cluster_centers_,



sfnc_group_clusters
sfnc_group_clusters.to_csv('sfnc_group_clusters.csv',index=False)
sil
np.argmax(sil) +2
X
kmeans = KMeans(n_clusters = 1).fit(X)
def calculate_WSS(points = X, kmax = 10):

    sse = []

    for k in range(1, kmax+1):

        kmeans = KMeans(n_clusters = k).fit(points)

        centroids = kmeans.cluster_centers_

        pred_clusters = kmeans.predict(points)

        curr_sse = 0

    

    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS

        for i in range(len(points)):

            curr_center = centroids[pred_clusters[i]]

            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

    

        sse.append(curr_sse)

    

    return sse
SSE = calculate_WSS(points = X, kmax = 10)
import matplotlib.pyplot as plt
plt.plot(SSE)
SSE
sfnc_group_clusters.columns