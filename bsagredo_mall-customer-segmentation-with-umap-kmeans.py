import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import itertools as it



import sklearn

from sklearn import preprocessing

from umap import UMAP

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

import plotly.express as px
mall = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

mall.sample(10)
mall['Gender Numeric'] = mall['Gender'].map({'Male': -1, 'Female': 1})

mall
numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender Numeric']



numeric_transformer = Pipeline(steps=[('scaler', preprocessing.MinMaxScaler())])



preprocessor = ColumnTransformer(

    transformers=[('num', numeric_transformer, numeric_features)])
umap = UMAP(n_components=2, random_state=2020)

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('uamp', umap)])

umap_out = pipe.fit_transform(mall)
sse = []

nclusters = list(range(1,11))

for k in nclusters:

    kmeans = KMeans(n_clusters=k)

    clusters = kmeans.fit_predict(umap_out)

    sse.append(kmeans.inertia_)

    

sb.pointplot(nclusters, sse).set_title('Inertia');
kmeans = KMeans(n_clusters = 6, random_state = 2020)

clusters = kmeans.fit_predict(umap_out)
df_umap = pd.DataFrame(data = umap_out, columns = ['Embedding 1', 'Embedding 2'])

df_clusters = pd.DataFrame(data = clusters, columns = ['Clusters']).apply(lambda x: 'C'+x.astype(str))



results = pd.concat([mall, df_umap, df_clusters], axis = 1)
results
fig = px.scatter(results, x = 'Embedding 1', y='Embedding 2',

                    color= 'Clusters',

                    hover_data = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)'],

                    width=600, height=600)

fig.show()