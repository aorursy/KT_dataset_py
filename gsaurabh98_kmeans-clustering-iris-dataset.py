import pandas as pd

import numpy as np

import matplotlib.pyplot as plt  

import plotly.graph_objs as go

from plotly.graph_objs import Figure, Data

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans 
dataset = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
dataset.head()
dataset.dropna(inplace=True)

dataset.drop('species', axis = 1 , inplace=True)
dataset.head()
fig = go.Figure()

fig.add_trace(go.Histogram(x=dataset['sepal_length'], name='sepal_length'))

fig.add_trace(go.Histogram(x=dataset['sepal_width'], name='sepal_width'))

fig.add_trace(go.Histogram(x=dataset['petal_length'], name='petal_length'))

fig.add_trace(go.Histogram(x=dataset['petal_width'], name='petal_width'))



# Overlay histograms

fig.layout.update(barmode='overlay')

# Reduce opacity to see histograms

fig.update_traces(opacity=0.75)

fig.show()
sd = StandardScaler()
scaled_data = sd.fit_transform(dataset)
dataset = pd.DataFrame(scaled_data, columns=dataset.columns)
dataset.head()
df = dataset.copy(deep=True)
pca = PCA()
decomposed_data = pca.fit_transform(dataset)
#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Iris Dataset Explained Variance')

plt.show()
pca = PCA(n_components=2)
decomposed_data = pca.fit_transform(dataset)
columns = ['component#%i' % i for i in range(2)]
decomposed_df = pd.DataFrame(decomposed_data, columns=columns)
decomposed_df.head()
# fig = go.Figure(data=go.Scatter(x=decomposed_df['component#0'], y=decomposed_df['component#1'], mode='markers'))



# Create a trace

trace = go.Scatter(

    x = decomposed_df['component#0'],

    y = decomposed_df['component#1'],

    mode = 'markers',

    marker=dict(

        size=16,

        color=np.random.randn(500), #set color equal to a variable

        colorscale='Viridis', # one of plotly colorscales

        showscale=True

    )

)

layout = {

    "title": "Scatter plot",

    "xaxis": {

        "showgrid": True,

        "zeroline": False,

        "showticklabels": False

    },

    "yaxis": {

        "showgrid": True,

        "zeroline": False,

        "showticklabels": False

    },

    "legend": {"font": {"size": 16}},

    "titlefont": {"size": 24}

}

data = [trace]

fig = Figure(data=data, layout=layout)

fig.show()
pca.explained_variance_ratio_
wcss = []

clusters = []

d = []

for k in range(1, 40):

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10,

                    max_iter=300)

    kmeans.fit_predict(dataset)

    wcss.append(kmeans.inertia_)

    clusters.append(k)
data = {

                "type": "scatter",

                "x": clusters,

                "y": wcss

            }



data = Data([data])

layout = go.Layout(

    title="Computing WCSS for KMeans++",

    yaxis=dict(title='Sum of squared errors'),

    xaxis=dict(title='Number of clusters'),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig = Figure(data=data, layout=layout)

fig.show()
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10,

                    max_iter=300)
kmeans
kmeans.fit(decomposed_df)
kmeans.labels_
dataset.insert(loc=0, column='clusters', value=pd.Series(kmeans.labels_).astype(int))
dataset.head(10)
decomposed_df.insert(loc=0, column='clusters', value=pd.Series(kmeans.labels_).astype(int))
dataset['clusters'].value_counts()
kmeans.cluster_centers_
centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns = decomposed_df.columns[1:])
centroid_df.head()
dataset.head()
decomposed_df.columns
clusters = decomposed_df['clusters'].unique()



columns = decomposed_df.columns[1:]



symbol = ('circle', 'square', 'triangle-up', 'diamond', 'cross', 'x',

                  'triangle-down', 'asterisk', 'octagon', 'diamond-tall-down')

color = ('yellow','blue',  'magenta', 'green', 'teal', 'navy','peru',

                 'lightslategrey', 'red', 'olive')



plot_data = []



for c in clusters:

    trace1 = {

      "mode": "markers", 

      "name": "cluster"+str(c),

      "type": "scatter", 

      "x": decomposed_df.loc[decomposed_df['clusters']==c][columns[0]], 

      "y": decomposed_df.loc[decomposed_df['clusters']==c][columns[1]],

      "marker": {

        "line": {

          "color": "navy", 

          "width": 0.5

        }, 

        "size": 12, 

        "color": color[c],

        "symbol": symbol[c]

      }

    }

    trace2 = {

      "name": "centroid"+str(c),

      "type": "scatter", 

      "x": [centroid_df[columns[0]][c]], 

      "y": [centroid_df[columns[1]][c]],

      "marker": {

        "color": "rgb(200,10,10)", 

        "symbol": symbol[c]

      }

    }

    plot_data.append(trace1)

    plot_data.append(trace2)

    

data = Data(plot_data)

layout = {

    "title": "K-Means Clustering (k=%s)" % len(clusters),

    "xaxis": {

        "showgrid": True,

        "zeroline": False,

        "showticklabels": False

    },

    "yaxis": {

        "showgrid": True,

        "zeroline": False,

        "showticklabels": False

    },

    "legend": {"font": {"size": 16}},

    "titlefont": {"size": 24}

}

fig = Figure(data=data, layout=layout)

fig.show()