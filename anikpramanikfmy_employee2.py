import numpy as np

import pandas as pd

import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df = pd.read_csv("../input/employee_dataset2.csv")

df.dropna(inplace=True)

df=df.groupby(['Employee_name']).mean()

df
from sklearn import preprocessing

df2=df

minmax_processed = preprocessing.MinMaxScaler().fit_transform(df2)

df_numeric_scaled = pd.DataFrame(minmax_processed, index=df2.index, columns=df2.columns)

kmeans = KMeans(n_clusters=3)

kmeans.fit(df_numeric_scaled)

df_numeric_scaled.head()

df['cluster'] = kmeans.labels_



df['total'] =df['Q.1']+ df['Q.2']+df['Q.3']+ df['Q.4']+df['Q.5']

res=df.groupby(['cluster']).mean()

res=res.sort_values(by=['total'],ascending = False)

res['cluster']=res.index.values

#fl

res 



dff=df

for x in range(0,dff.shape[0]):

    if dff.iloc[x,5] == res.iloc[0,6]:

        dff.iloc[x,5]=0

    elif dff.iloc[x,5] == res.iloc[1,6]:

        dff.iloc[x,5]=1

    elif dff.iloc[x,5] == res.iloc[2,6]:

        dff.iloc[x,5]=2

    

df=dff

df


X = df.iloc[:,0:5].values

X

Y=df['cluster']

Y



from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(X_std)

Y_sklearn
labelname = {0: 'Good',

              1: 'Mediocore',

              2: 'Corrupt'}


with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip((0, 1, 2),

                        ('blue', 'green', 'red')):

        plt.scatter(Y_sklearn[Y==lab, 0],

                    Y_sklearn[Y==lab, 1],

                    label=labelname[lab],

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()







