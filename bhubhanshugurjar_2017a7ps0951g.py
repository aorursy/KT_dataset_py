import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
original=pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=',')

data = original

temp=original

temp['Class'].dtype
data=data.drop(['ID','Class'],1)
data.head()
original.head()
col=data.columns

col
# for i in col:

#     indexNames = data[ data[i] == '?' ].index

#     data.drop(indexNames , inplace=True)



# for i in col:

#     indexNames = temp[ temp[i] == '?' ].index

#     temp.drop(indexNames , inplace=True)

data=data.replace('?',np.NaN)

data=data.fillna(data.mean())

data.fillna(method='ffill', inplace=True)
original.head()
data.info()
col=data.columns[data.eq('?').any()]

col
# data['Col189'].replace({

#     'yes':1,

#     'no':0

#     },inplace=True)
col = data.columns[data.dtypes == np.object]

col
ob=[]

for i in col:

    try:

        data[i] = data[i].apply(pd.to_numeric)

    except:

        ob.append(i)

ob
# data['Col190'].replace({

#     'sacc1':0,

#     'sacc2':1,

#     'sacc4':2,

#     'sacc5':3

#     },inplace=True)

# data['Col189'].replace({

#     'yes':1,

#     'no':0

#     },inplace=True)



# data['Col191'].replace({

#     'time1':0,

#     'time2':1,

#     'time3':2,

#     },inplace=True)

# data['Col192'].replace({

#     'p1':0,

#     'p2':1,

#     'p3':2,

#     'p4':3,

#     'p5':4,

#     'p6':5,

#     'p7':6,

#     'p8':7,

#     'p9':8,

#     'p10':9

#     },inplace=True)

# data['Col193'].replace({

#     'F1':0,

#     'M1':1,

#     'M0':2,

#     'F0':3

#     },inplace=True)

# data['Col194'].replace({

#     'ad':0,

#     'ab':1,

#     'ac':2,

#     },inplace=True)

# data['Col195'].replace({

#     'Jb1':0,

#     'Jb2':1,

#     'Jb3':2,

#     'Jb4':3

#     },inplace=True)

# data['Col196'].replace({

#     'H1':0,

#     'H2':1,

#     'H3':2,

#     },inplace=True)

# data['Col197'].replace({

#     'me':'ME',

#     'sm':'SM',

#     'la':'LA',

#     'M.E.':'ME',

#     },inplace=True)
data = pd.get_dummies(data, columns=[i for i in ob])

data.head()
# from sklearn import preprocessing

# #Performing Min_Max Normalization

# min_max_scaler = preprocessing.MinMaxScaler()

# np_scaled = min_max_scaler.fit_transform(data)

# data1 = pd.DataFrame(np_scaled)

# data1.head()



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaled_data=scaler.fit(data).transform(data)

data1=pd.DataFrame(scaled_data,columns=data.columns)

data1.head()


from sklearn.manifold import TSNE

ts = TSNE(n_components=2).fit_transform(data1)





# from sklearn.decomposition import PCA

# pca1 = PCA(n_components=50)

# pca1.fit(data1)

# T1 = pca1.transform(data1)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 12,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(ts)

plt.scatter(ts[:, 0], ts[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(ts, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_ac=cut_tree(linkage_matrix1, n_clusters = 12)

y_ac

pred=[i[0] for i in y_ac]

pred
rows, cols = (12, 6)

arr = [[0 for i in range(cols)] for j in range(rows)]

for i,j in zip(pred,temp['Class']):

    if(np.isnan(j)):

        break

    else:

        arr[i][int(j)]=arr[i][int(j)]+1



print(arr)
original=pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=',')
res = []

for i in range(len(original)-1300):

    i=i+1300

    if pred[i] == 8 or pred[i] == 2 or pred[i] == 10 or pred[i] == 11:

        res.append(1)

    elif pred[i] == 4 or pred[i] == 9 :

        res.append(2)

    elif pred[i] == 0 or pred[i] == 5:

        res.append(3)

    elif pred[i] == 1 or pred[i] == 3:

        res.append(4)

    elif pred[i] == 6 or pred[i] == 7:

        res.append(5)



len(res)

        

res1 = pd.DataFrame(res)

original=original[1300:]

res1.reset_index(drop=True, inplace=True)

original.reset_index(drop=True, inplace=True)



final = pd.concat([original["ID"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final.head()
final.to_csv("final3.csv",index=False)