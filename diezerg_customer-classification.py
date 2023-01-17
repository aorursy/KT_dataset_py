import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
#os.getcwd()
flt = pd.read_csv('../input/flight_ENG_2.csv')
print(flt.shape)
flt.head()
missing = flt.shape[0] - flt.count()
print(missing[missing != 0])
flt['SUM_YR'] = flt['SUM_YR_1'] + flt['SUM_YR_2']
def hist_list(lst):
    plt.figure(figsize=[len(lst)*3,5])
    i = 1
    for col in lst:
        ax = plt.subplot(1,len(lst),i)
        ax.hist(flt[col].dropna(),50)
        plt.title(col)
        i = i+1
key_columns = ['SUM_YR','FLIGHT_COUNT','LAST_TO_END','SEG_KM_SUM']
hist_list(key_columns)
print (flt[key_columns].describe())
flt['R'] = -scale(np.log(flt['LAST_TO_END']))
flt['F'] = scale(np.log(flt['FLIGHT_COUNT']))
flt['M'] = scale(np.log(flt['SEG_KM_SUM']))
RFM = ['R','F','M']
hist_list(RFM)

km = KMeans(n_clusters=6, max_iter=500)
km.fit(flt[RFM])
flt['cluster']=km.labels_
ax = plt.subplot(111,projection='3d')
ax.scatter(flt['R'],flt['F'],flt['M'], 
           c=flt['cluster'])
ax.set_xlabel('R')
ax.set_ylabel('F')
ax.set_zlabel('M')
plt.show()
def cluster_bar(names,grp='cluster'):
    a = len(names)
    i = 1
    cl = ['bgrcmykw'[c] for c in range(len('bgrcmykw'))]
    plt.figure(figsize=(8,4))
    plt.subplots_adjust(wspace=3)

    for col in names:
        ax = plt.subplot(1,a,i)
        g = flt.groupby(grp)
        x = g[col].mean().index
        y = g[col].mean().values
        ax.barh(x,y,color=cl[i-1])
        plt.title(col)
        i = i+1
        
cluster_bar(['LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM'])
cluster_name = {0:'lost_casual',1:'lapsing hero',2:'new tester',
                3:'hero',4:'lapsing customer',5:'growing hero'}
flt['classification'] = flt['cluster'].map(cluster_name)
criteria = ['AGE','SUM_YR','Points_Sum','avg_discount']
cluster_bar(criteria,'classification')