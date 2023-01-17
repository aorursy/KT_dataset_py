# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/realAWSCloudwatch/realAWSCloudwatch/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np



from sklearn.cluster import DBSCAN

from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs

from sklearn.preprocessing import StandardScaler

import warnings

import itertools

import pandas as pd

import numpy as np

import datetime as dt

import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



fpath = "../input/"

fname = "iap.csv"

fpath = "../input/realAWSCloudwatch/realAWSCloudwatch/"

fname = "ec2_cpu_utilization_825cc2.csv"

# fname = "grok_asg_anomaly.csv"



fullPath = fpath + fname



def parser(x):

	return dt.datetime.strptime(x, "%Y-%m")

# return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")





data = pd.read_csv(fullPath)

data.plot()



a = []

x = []

y=[]

for i in range(0, len(data)-1):

    a.append([i,data["value"][i]])    



X = a



X = StandardScaler().fit_transform(X)

for i in range(0,len(X)):

    x.append(X[i][0])

    y.append(X[i][1])



plt.scatter(x,y)

plt.show()

db = DBSCAN(eps=0.5, min_samples=200).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)



print('Estimated number of clusters: %d' % n_clusters_)



unique_labels = set(labels)

plt.figure(num=None, figsize=(10, 10), facecolor='w', edgecolor='k')

for k in unique_labels:

    col=[0,0.5,1,1]

    if k == -1:

        col = [1, 0, 0, 1]

    class_member_mask = (labels == k)



    xy = X[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', color=tuple(col),markersize=5, alpha=0.5)



    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', color=tuple(col), markersize=5, alpha=0.5)



plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()