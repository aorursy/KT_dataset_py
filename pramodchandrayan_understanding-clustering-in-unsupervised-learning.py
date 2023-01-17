import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import warnings



import itertools

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
#loading the data set



ws_df = pd.read_csv('../input/Wholesale%20customers%20data.csv')

ws_df.head(100)
ws_df.drop(labels=(['Channel','Region']),axis=1,inplace=True)
ws_df.info()

ws_df.shape
ws_df.describe().T
import itertools



attr_col = [i for i in ws_df.columns if i not in 'strength']

length = len(attr_col)

cs = ["b","r","g","c","m","k"]

fig = plt.figure(figsize=(13,25))



for i,j,k in itertools.zip_longest(attr_col,range(length),cs):

    plt.subplot(4,2,j+1)

    ax = sns.distplot(ws_df[i],color=k,rug=True)

    ax.set_facecolor("w")

    plt.axvline(ws_df[i].mean(),linestyle="dashed",label="mean",color="k")

    plt.legend(loc="best")

    plt.title(i,color="navy")

    plt.xlabel("")
#Summary View of all attribute , The we will look into all the boxplot individually to trace out outliers



ax = sns.boxplot(data=ws_df, orient="h")
from sklearn.preprocessing import normalize



X_std = normalize(ws_df)

X_std = pd.DataFrame(X_std, columns=ws_df.columns)

X_std.head()



import scipy.cluster.hierarchy as shc

plt.figure(figsize=(15, 10))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(X_std, method='ward'))

plt.figure(figsize=(15, 10))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(X_std, method='ward'))

plt.axhline(y=6, color='y', linestyle='-')
from sklearn.cluster import AgglomerativeClustering



agg_clu = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  

agg_clu.fit_predict(X_std)
plt.figure(figsize=(15, 10))  

plt.scatter(X_std['Milk'], X_std['Grocery'], c=agg_clu.labels_) 