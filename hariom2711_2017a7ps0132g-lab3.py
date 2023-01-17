import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.decomposition import PCA

from sklearn import cluster, datasets, mixture

from sklearn.neighbors import kneighbors_graph, NearestNeighbors

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import cycle, islice

from sklearn.metrics import accuracy_score
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

sss = []

for element in df['TotalCharges']:

    try:

        float(element)

    except ValueError:

        sss.append(element)



for element in sss:

    df = df[df['TotalCharges'] != element]

df['TotalCharges'] = df['TotalCharges'].astype(float)



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for col in df.columns:

    if df[col].dtype == 'O':

        le.fit(df[col].unique())

        df[col] = le.transform(df[col])

#dataTrimmed

X = df.loc[:, 'custId':'TotalCharges']

X = MinMaxScaler().fit_transform(X)

y = df['Satisfied']
pca = PCA(n_components='mle',

          random_state=22         

         )
pca.fit(X)
# max_acc = 0

# max_val = 0

# i=20

# while i < 51:

#     pca = PCA(n_components='mle',

#              random_state=i

#              )

#     pca.fit(X)

#     scl = cluster.SpectralClustering(n_clusters=2, 

#                                  affinity='poly',

#                                  coef0 = 0.5,

#                                  degree=3,

#                                  random_state=25,

#                                  n_init=5,

#                                  gamma=0.8

#                                 )

#     scl.fit(X)

#     cur = accuracy_score(scl.labels_, y)

#     if cur > max_acc:

#         max_acc = cur

#         max_val = i

#     i = i+5



# print(max_acc)

# print(max_val)

# max_acc = 0

# max_val = 0

# i=20

# while i < 31:

#     scl = cluster.SpectralClustering(n_clusters=2, 

#                                  affinity='poly',

#                                  coef0 = 0.5,

#                                  degree=22,

#                                  random_state=i,

#                                  n_init=5,

#                                  gamma=0.8

#                                 )

#     scl.fit(X)

#     cur = accuracy_score(scl.labels_, y)

#     if cur > max_acc:

#         max_acc = cur

#         max_val = i

#     i = i+1



# print(max_acc)

# print(max_val)

# # scl = cluster.SpectralClustering(n_clusters=2, 

# #                                  affinity='poly',

# #                                  coef0 = 0.7,

# #                                  degree=3,

# #                                  random_state=25,

# #                                  n_init=5,

# #                                  gamma=0.8

# #                                 )
scl = cluster.SpectralClustering(n_clusters=2, 

                                 affinity='poly',

                                 coef0 = 0.5,

                                 degree=3,

                                 random_state=25,

                                 n_init=5,

                                 gamma=0.8

                                )

scl.fit(X)

accuracy_score(scl.labels_, y)
scl.fit(X)
accuracy_score(scl.labels_, y)
scl = cluster.SpectralClustering(n_clusters=2, 

                                 affinity='rbf',

#                                  coef0 = 0.5,

#                                  degree=22,

                                 random_state=22,

#                                  n_init=5,

                                 gamma=0.8

                                )
scl.fit(X)
accuracy_score(scl.labels_, y)
scl = cluster.SpectralClustering(n_clusters=2, 

                                 affinity='poly',

                                 coef0 = 0.5,

                                 degree=3,

                                 random_state=25,

                                 n_init=5,

                                 gamma=0.8

                                )

scl.fit(X)

accuracy_score(scl.labels_, y)
df_test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df_test['TotalCharges'] = df_test['TotalCharges'].replace({" ": "0"})

df_test['TotalCharges'] = df_test['TotalCharges'].astype(float)



le = preprocessing.LabelEncoder()

for col in df_test.columns:

    if df_test[col].dtype == 'O':

        le.fit(df_test[col].unique())

        df_test[col] = le.transform(df_test[col])

X = MinMaxScaler().fit_transform(df_test)

# pca = PCA()

pca.fit(X)
scl.fit(X)
y_pd=pd.Series(scl.labels_)

# df_test['custId']

test_id = df_test['custId']

ans = pd.DataFrame()

ans['custId'] = test_id

ans['Satisfied'] = y_pd



ans.to_csv('solnF.csv', index=False)