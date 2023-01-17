import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

%matplotlib inline
import os

os.listdir('../input/dmassign1/')
df = pd.read_csv('../input/dmassign1/data.csv')

df = pd.read_csv('../input/dmassign1/data.csv')

labels = np.asarray(df.dropna()['Class'], dtype=int)

ids = np.asarray(df['ID'], dtype=object)

df.drop(labels=['ID', 'Class'], axis=1, inplace=True)



# find all columns and datatypes

col = list(df.columns)

num_cols = col[:188]

obj_cols = col[188:]



# replace ? with nan

df.replace('?', np.nan, inplace=True)



df[num_cols] = df[num_cols].astype(float)

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())



# remove all categorical nans usign mode

for i in obj_cols:

    df[i] = df[i].fillna(df[i].mode()[0])

drop_cols = ['Col'+(str)(x) for x in range(1, 189)]

# df.drop(drop_cols, axis=1, inplace=True)
df1 = df[drop_cols]
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2)

df_dropped = tsne.fit_transform(df1)
import numpy as np

import pandas as pd

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import accuracy_score

from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix



class clust():

    def _load_data(self, df, labels):

        self.X = df

        self.labels = labels

        self.labels = [x-1 for x in self.labels]

        

    def __init__(self, df, labels):

        self._load_data(df, labels)

    

    def _find_mapping(self, predicted, n_clusters):      

        self.mapping = np.zeros((n_clusters, len(np.unique(self.labels))), int)

        for i in range(len(predicted)):

            self.mapping[predicted[i]][self.labels[i]]+=1

        self.mapping = np.argmax(self.mapping, axis=1)



    def map_labels(self, predicted):

        for i in range(len(predicted)):

            predicted[i] = self.mapping[predicted[i]]

        return predicted



    def Kmeans(self, n_clusters, verbose=0):

        clf = KMeans(n_clusters = n_clusters, n_jobs=2, random_state=42)

        predicted = clf.fit_predict(self.X)

        y_labels_train = predicted[:len(labels)]

        self._find_mapping(y_labels_train, n_clusters)

        y_labels_train = self.map_labels(y_labels_train)

        self.inertia = clf.inertia_

        self.train_randscore = adjusted_rand_score(self.labels, y_labels_train)

        self.train_accuracy = accuracy_score(self.labels, y_labels_train)

        if(verbose):

            print("n_cluster : ", n_clusters, "rand_score :", self.train_randscore, "accuracy :", self.train_accuracy)

        

        y_labels_test = predicted[len(labels):]

        for i in range(len(y_labels_test)):

            y_labels_test[i] = self.mapping[y_labels_test[i]]+1

        return y_labels_test

    

    def Agglomerative(self, n_clusters, affinity="cosine", linkage="average", verbose=0):

        clf = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)

        predicted = clf.fit_predict(self.X)

        y_labels_train = predicted[:len(self.labels)]

        self._find_mapping(y_labels_train, n_clusters)

        y_labels_train = self.map_labels(y_labels_train)

        

        self.train_randscore = adjusted_rand_score(self.labels, y_labels_train)

        self.train_accuracy = accuracy_score(self.labels, y_labels_train)

        if(verbose):

            print("n_cluster : ", n_clusters, "rand_score :", self.train_randscore, "accuracy :", self.train_accuracy)



        y_labels_test = predicted[len(labels):]

        for i in range(len(y_labels_test)):

            y_labels_test[i] = self.mapping[y_labels_test[i]]+1

        return y_labels_test
cl = clust(df_dropped, labels)

y_ans1 = cl.Kmeans(n_clusters=49, verbose=1)

sub_df1 = pd.DataFrame(data={'ID':ids[1300:], 'Class':y_ans1})

pd.DataFrame.to_csv(sub_df1, 'sub_tsne_49.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html     =     '<a     download="{filename}"     href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub_df1)