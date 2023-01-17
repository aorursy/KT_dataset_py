# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/train.csv')
df.head()
temp = df.select_dtypes(include='object')

dumb = pd.get_dummies(temp)

df2 = df.select_dtypes(exclude='object')

df3 = pd.concat([df2, dumb], axis=1, sort=False)

# df3.LotFrontage = df3.LotFrontage.astype(int)
df4 = df3.fillna(0).astype(int)

df_tr = df4

clmns = list(df4.columns.values)
#print(clmns)
ks = range(1, 10)
inertias = []
sse = {}

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df4)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
    #for plot... please work
    sse[k] = model.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()
model1 = KMeans(n_clusters=4, random_state=0)
df_tr_std = stats.zscore(df_tr[clmns])
model1.fit(df_tr_std)
labels = model1.labels_
df_tr['clusters'] = labels
clmns.extend(['clusters'])
x1 = df_tr[["OverallQual","GrLivArea","GarageCars","clusters"]]
print(x1.groupby(['clusters']).mean())
print(x1.groupby(['clusters']).std())
# this has all been super cool.  lets do it with the train dataset now
adf = pd.read_csv('../input/test.csv')
temp2 = adf.select_dtypes(include='object')

dumb2 = pd.get_dummies(temp2)

adf2 = adf.select_dtypes(exclude='object')

adf3 = pd.concat([adf2, dumb2], axis=1, sort=False)

# df3.LotFrontage = df3.LotFrontage.astype(int)
adf4 = adf3.fillna(0).astype(int)

adf_tr = adf4

clmns2 = list(adf4.columns.values)

#print(clmns2)
ks2 = range(1, 10)
inertias2 = []
sse2 = {}

for k in ks2:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(adf4)
    
    # Append the inertia to the list of inertias
    inertias2.append(model.inertia_)
    
    #for plot... please work
    sse2[k] = model.inertia_
plt.figure()
plt.plot(list(sse2.keys()), list(sse2.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()
model2 = KMeans(n_clusters=4, random_state=0)
adf_tr_std = stats.zscore(adf_tr[clmns2])
model2.fit(adf_tr_std)
labels2 = model2.labels_
adf_tr['clusters'] = labels2
clmns2.extend(['clusters'])
x1 = df_tr[["OverallQual","GrLivArea","GarageCars","clusters"]]
print(x1.groupby(['clusters']).mean())
x2 = adf_tr[["OverallQual","GrLivArea","GarageCars","clusters"]]
print(x2.groupby(['clusters']).mean())
# 3a=3b, 0a=1b, 1a=2b, 2a=0b
print(x1.groupby(['clusters']).std())
print(x2.groupby(['clusters']).std())
adf_tr.head()
adf_tr_0=adf_tr.loc[adf_tr['clusters']==0]
adf_tr_1=adf_tr.loc[adf_tr['clusters']==1]
adf_tr_2=adf_tr.loc[adf_tr['clusters']==2]
adf_tr_3=adf_tr.loc[adf_tr['clusters']==3]

df_tr_0=df_tr.loc[df_tr['clusters']==2]
df_tr_1=df_tr.loc[df_tr['clusters']==0]
df_tr_2=df_tr.loc[df_tr['clusters']==1]
df_tr_3=df_tr.loc[df_tr['clusters']==3]
from sklearn import linear_model
lin_model = linear_model.LinearRegression()

a0 = df_tr_0[["OverallQual","GrLivArea","GarageCars"]]
a1 = df_tr_1[["OverallQual","GrLivArea","GarageCars"]]
a2 = df_tr_2[["OverallQual","GrLivArea","GarageCars"]]
a3 = df_tr_3[["OverallQual","GrLivArea","GarageCars"]]

b0 = df_tr_0["SalePrice"]
b1 = df_tr_1["SalePrice"]
b2 = df_tr_2["SalePrice"]
b3 = df_tr_3["SalePrice"]

#will have to re-run this for each of the 4 clusters
lin_model.fit(a0,b0)
adf_tr_0_test = adf_tr_0[["OverallQual","GrLivArea","GarageCars"]]
adf_tr_0_id = adf_tr_0[["Id"]].reset_index()
cluster_0_predictions = lin_model.predict(adf_tr_0_test)
cluster_0_predictions = pd.DataFrame(cluster_0_predictions)
cluster_0_predictions = cluster_0_predictions.merge(adf_tr_0_id,left_index=True, right_index=True)

lin_model.fit(a1,b1)
adf_tr_1_test = adf_tr_1[["OverallQual","GrLivArea","GarageCars"]]
adf_tr_1_id = adf_tr_1[["Id"]].reset_index()
cluster_1_predictions = lin_model.predict(adf_tr_1_test)
cluster_1_predictions = pd.DataFrame(cluster_1_predictions)
cluster_1_predictions = cluster_1_predictions.merge(adf_tr_1_id,left_index=True, right_index=True)

lin_model.fit(a2,b2)
adf_tr_2_test = adf_tr_2[["OverallQual","GrLivArea","GarageCars"]]
adf_tr_2_id = adf_tr_2[["Id"]].reset_index()
cluster_2_predictions = lin_model.predict(adf_tr_2_test)
cluster_2_predictions = pd.DataFrame(cluster_2_predictions)
cluster_2_predictions = cluster_2_predictions.merge(adf_tr_2_id,left_index=True, right_index=True)

lin_model.fit(a3,b3)
adf_tr_3_test = adf_tr_3[["OverallQual","GrLivArea","GarageCars"]]
adf_tr_3_id = adf_tr_3[["Id"]].reset_index()
cluster_3_predictions = lin_model.predict(adf_tr_3_test)
cluster_3_predictions = pd.DataFrame(cluster_3_predictions)
cluster_3_predictions = cluster_3_predictions.merge(adf_tr_3_id,left_index=True, right_index=True)
frames = [cluster_0_predictions,cluster_1_predictions,cluster_2_predictions,cluster_3_predictions]
predictions = pd.concat(frames)
predictions.to_csv("predictions.csv")