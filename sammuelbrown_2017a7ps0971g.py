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
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV File", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    

    html = html.format(payload=payload, title=title, filename=filename)

    return HTML(html)
from collections import Counter

import math
data = pd.read_csv("../input/dmassign1/data.csv")

train_cluster = data.drop(['ID','Class'], axis = 1)

dtype_dict = train_cluster.columns.to_series().groupby(train_cluster.dtypes).groups

dict_dtype2 = {k.name: v for k,v in dtype_dict.items()}

object_type = [i for i in dict_dtype2['object']]
a = train_cluster.nunique()

a = a.tolist()

col = train_cluster.columns
b = [col[i] for i,j in enumerate(a) if col[i] in object_type and j > 1000]
len(b)
len(object_type)
train_cluster.replace('?', np.nan,inplace=True)
train_cluster.info(verbose=True, null_counts=True)

for col in b:

    train_cluster[col] = train_cluster[col].astype('float64')
means = train_cluster.mean()

train_cluster2 = train_cluster.fillna(means)
only_string_list = []



for col in object_type:

    if col not in b:

        only_string_list.append(col)
for i in only_string_list:

    print(Counter(train_cluster2[i]))
def change_string(st):

    if(pd.isnull(st)):

        return st

    st = st.lower()

    if(st == "m.e."):

        return "me"

    return st
for i in only_string_list:

    train_cluster2[i] = train_cluster2[i].apply(lambda x: change_string(x))

    mode = train_cluster2[i].mode()

    train_cluster2[i] = train_cluster2[i].fillna(mode[0])

    print(Counter(train_cluster2[i]))
train_cluster2.info(verbose=True, null_counts=True)
from sklearn.preprocessing import StandardScaler



train_cluster3 = train_cluster2.copy(deep=True)



for col in train_cluster2.columns:

    if col not in only_string_list:

        np_scaled = StandardScaler().fit_transform(train_cluster3[col].values.reshape(-1,1))

        

        train_cluster3[col] = np_scaled

        

train_cluster3.head()
train_cluster_final = pd.get_dummies(train_cluster3)
train_cluster_final
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



pca = PCA().fit(train_cluster_final)





plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') 



plt.show()
pca_act = PCA(n_components = 0.95)

principalComponents = pca_act.fit_transform(train_cluster_final)

st = "principal component "

columns = []

for i in range(1,48):

    columns.append(st + str(i))

    

principalDf = pd.DataFrame(data = principalComponents, columns = columns)
principalDf
from sklearn.cluster import KMeans

from sklearn.metrics import davies_bouldin_score



list_elbow = []

list_bouldin = []

for i in range(2,80):

    kmeans = KMeans(n_clusters = i, random_state = 42)

    kmeans.fit(principalDf)

    list_elbow.append(kmeans.inertia_)

    list_bouldin.append(davies_bouldin_score(principalDf, kmeans.labels_))
plt.plot(range(2,80), list_elbow)

plt.show()
plt.plot(range(2,80), list_bouldin)

plt.show()
list_bouldin.index(min(list_bouldin))
list_bouldin[74]


n_clusters = 74



pred = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(principalDf)

davies_bouldin_score(principalDf, kmeans.labels_)


map_class_clust = np.zeros((74,5))



principalDf["Cluster"] = pred

principalDf["Class"] = data["Class"]



for i in principalDf.head(1300).iterrows():

    map_class_clust[int(i[1]['Cluster'])][int(i[1]['Class']) - 1] +=1

    
pred_clust = np.argmax(map_class_clust, axis=1)
m_list = []



for i in principalDf.iterrows():

    m_list.append(pred_clust[int(i[1]['Cluster'])] + 1)
final_csv = principalDf.copy(deep=True)

final_csv['Class'] = m_list

final_csv['ID'] = data['ID']
final_csv2 = final_csv[['ID','Class']]

final_csv2 = final_csv2.drop(final_csv2.index[:1300])

final_csv2.head()
create_download_link(final_csv2)
#principalDf = principalDf.drop(['Class','Cluster'],axis=1)
#dict_where_class = {}



#principalDf['Class'] = data['Class']



#dict_where_class[1] = principalDf[principalDf["Class"] == 1]

#dict_where_class[2] = principalDf[principalDf["Class"] == 2]

#dict_where_class[3] = principalDf[principalDf["Class"] == 3]

#dict_where_class[4] = principalDf[principalDf["Class"] == 4]

#dict_where_class[5] = principalDf[principalDf["Class"] == 5]



#for i in dict_where_class:

    #dict_where_class[i] = dict_where_class[i].drop(['Class'], axis=1)
#principalDf2 = principalDf.copy(deep=True)

#principalDf2 = principalDf2.drop(['Class'],axis=1)
#def recalculate(d1, temp):

#    v = np.vstack((d1,temp))

#    return np.mean(v)
#centroids = np.zeros((5,60))



#for i in dict_where_class:

#    temp_arr = dict_where_class[i].values

#    cent = np.mean(temp_arr, axis=0)

#    centroids[i-1] = cent

#from operator import itemgetter



#def find_min_centroid(centroids, temp_arr):

#    min_list = []

#    for i in centroids:

#        dist = np.linalg.norm(temp_arr - i)

#        min_list.append(dist)

        

#    return min(enumerate(min_list), key=itemgetter(1))[0] 
#import math



#for i in principalDf.iterrows():

#    if(math.isnan(i[1]['Class'])):

#        temp_arr = np.array(principalDf2.iloc[i[0]])

#        j = find_min_centroid(centroids, temp_arr)

#        i[1]['Class'] =j

#        centroids[j] = recalculate(dict_where_class[j+1].values,temp_arr)
#principalDf
#final_csv_cent = pd.concat([data['ID'], principalDf['Class']], axis=1, keys=['ID', 'Class'])

#final_csv_cent2 = final_csv_cent.drop(final_csv.index[:1300])
#final_csv_cent2['Class'] = final_csv_cent2['Class']+1.0
#final_csv_cent2.tail()
#create_download_link(final_csv_cent2)