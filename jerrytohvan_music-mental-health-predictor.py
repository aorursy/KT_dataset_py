import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn import datasets, linear_model

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



%matplotlib inline







data_frame = pd.read_csv("../input/music-and-mental-health-data/master_song_data.csv")



data_frame.info()



data_frame = data_frame.iloc[: , [0,1,2,4,5,6,7,9,11,12,13,14]]



data_frame.head()
from sklearn import preprocessing



exclude = ["Participant_ID", "Song_name", "Artist"]

for column in data_frame.columns:

    if data_frame[column].dtype == type(object) and column not in exclude:

        le = preprocessing.LabelEncoder()

        data_frame[column] = le.fit_transform(data_frame[column])

        

data_frame.head()

data_frame
scaler = StandardScaler(with_mean=False)

transformed = scaler.fit_transform(data_frame[['Loudness', 'Valence', 'Danceability', 'Acousticness']])

for idx, col in enumerate(data_frame.columns[4:-6]):

    data_frame[col] = transformed[:,idx]

    

data_frame.describe()
x = data_frame["Danceability"].values

y = data_frame["Valence"].values

l = data_frame["Loudness"].values

a = data_frame["Acousticness"].values

z = data_frame["Total_mental_health"].values





z = z.reshape(z.shape[0], 1)

x = x.reshape(x.shape[0], 1)

y = y.reshape(y.shape[0], 1)

l = l.reshape(l.shape[0], 1)

a = a.reshape(a.shape[0], 1)



regr = linear_model.LinearRegression()

regr.fit(z, x)





fig = plt.figure(figsize=(6, 6))

fig.suptitle("Correlation between Danceability and Mental Health Metric")



zx = plt.subplot(1, 1, 1)

zx.scatter(z, x, alpha=0.5)

zx.plot(z, regr.predict(z), color="red", linewidth=3)

plt.xticks(())

plt.yticks(())





plt.xlabel("Mental Health Metric")

plt.ylabel("Danceability")



plt.show()
regr = linear_model.LinearRegression()

regr.fit(z, y)



fig = plt.figure(figsize=(6, 6))

fig.suptitle("Correlation between mental health assessment and valence")



az = plt.subplot(1, 1, 1)

az.scatter(z, y, alpha=0.5)

az.plot(z, regr.predict(z), color="red", linewidth=3)

plt.xticks(())

plt.yticks(())





plt.xlabel("Mental Health Metric")

plt.ylabel("Valence")



plt.show()
regr = linear_model.LinearRegression()

regr.fit(z, a)



fig = plt.figure(figsize=(6, 6))

fig.suptitle("Correlation between mental health assessment and acousticness")



za = plt.subplot(1, 1, 1)

za.scatter(z, a, alpha=0.5)

za.plot(z, regr.predict(z), color="red", linewidth=3)

plt.xticks(())

plt.yticks(())



plt.xlabel("Mental Health Metric")

plt.ylabel("Acousticness")



plt.show()
regr = linear_model.LinearRegression()

regr.fit(z, l)



fig = plt.figure(figsize=(6, 6))

fig.suptitle("Correlation between mental health assessment and loudness")



zl = plt.subplot(1, 1, 1)

zl.scatter(z, l, alpha=0.5)

zl.plot(z, regr.predict(z), color="red", linewidth=3)

plt.xticks(())

plt.yticks(())





plt.xlabel("Mental Health Metric")

plt.ylabel("Loudness")



plt.show()
x = "Valence"

y = "Total_mental_health"



#Structure graphs layout

fig, (ax2) = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(10, 5))

fig.suptitle("Valence X Total Mental Health")



#add  "heatmap" that illustrates the number of songs found at all values of valence and danceability.

h = ax2.hist2d(data_frame[x], data_frame[y], bins=20)

ax2.set_xlabel(x)

ax2.set_ylabel(y)



plt.colorbar(h[3], ax=ax2)



plt.show()

x = "Loudness"

y = "Total_mental_health"





#Structure graphs layout

fig, (ax2) = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(10, 5))

fig.suptitle("Loudness X Total Mental Health")



#add  "heatmap" that illustrates the number of songs found at all values of valence and danceability.

h = ax2.hist2d(data_frame[x], data_frame[y], bins=20)

ax2.set_xlabel(x)

ax2.set_ylabel(y)



plt.colorbar(h[3], ax=ax2)



plt.show()

x = "Danceability"

y = "Total_mental_health"





#Structure graphs layout

fig, (ax2) = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(10, 5))

fig.suptitle("Danceability X Total Mental Health")



#add  "heatmap" that illustrates the number of songs found at all values of valence and danceability.

h = ax2.hist2d(data_frame[x], data_frame[y], bins=20)

ax2.set_xlabel(x)

ax2.set_ylabel(y)





plt.colorbar(h[3], ax=ax2)



plt.show()

x = "Acousticness"

y = "Total_mental_health"





#Structure graphs layout

fig, (ax2) = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(10, 5))

fig.suptitle("Acousticness X Total Mental Health")



#add  "heatmap" that illustrates the number of songs found at all values of valence and danceability.

h = ax2.hist2d(data_frame[x], data_frame[y], bins=20)

ax2.set_xlabel(x)

ax2.set_ylabel(y)





plt.colorbar(h[3], ax=ax2)



plt.show()

x = "Audio + Lyrics analysis"

y = "Total_mental_health"





#Structure graphs layout

fig, (ax2) = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(10, 5))

fig.suptitle("Audio & Lyrics Sentiment X Total Mental Health")



#add  "heatmap" that illustrates the number of songs found at all values of valence and danceability.

h = ax2.hist2d(data_frame[x], data_frame[y], bins=4)

ax2.set_xlabel(x)

ax2.set_ylabel(y)





plt.colorbar(h[3], ax=ax2)



plt.show()

chosen = ["Loudness", "Valence", "Danceability", "Acousticness", "Audio + Lyrics analysis" ,"Total_mental_health"]



text1 = data_frame["Artist"] + " - " + data_frame["Song_name"]

text2 = text1.values 



X = data_frame[chosen].values

y = data_frame["Total_mental_health"].values



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



pca = PCA(n_components=3)

pca.fit(X)



X = pca.transform(X)



import plotly.offline as py

#import plotly.express as px

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#x,y,z = PCA Index 

#can we add markers value?

trace = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=X[:,2],

    text=text2,

    mode="markers",

    marker=dict(

        size=8,

        color=y

    )

)



fig = go.Figure(data=[trace])



py.iplot(fig, filename="test-graph")
chosen = ["Loudness", "Valence", "Danceability","Acousticness","Audio + Lyrics analysis","Total_mental_health"]

text1 = data_frame["Artist"] + " - " + data_frame["Song_name"]

text2 = text1.values



# X = data_frame.drop(droppable, axis=1).values

X = data_frame[chosen].values

y = data_frame["Total_mental_health"].values



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



pca = PCA(n_components=2)

pca.fit(X)



X = pca.transform(X)



fig = {

    "data": [

        {

            "x": X[:, 0],

            "y": X[:, 1],

            "text": text2,

            "mode": "markers",

            "marker": {"size": "8", "color": y}

        }

    ],

    "layout": {

         "xaxis": {"title": "PCA X"},

        "yaxis": {"title": "PCA Y"},

    }

}



py.iplot(fig, filename="test-graph2")
kmeans_X_Y = pd.DataFrame({'pca1': X[:, 0], 'pca2': X[:, 1]})
import time



chosen = ["Loudness", "Valence", "Danceability", "Acousticness", "Audio + Lyrics analysis","Total_mental_health"]





X = data_frame[chosen].values

y = data_frame["Total_mental_health"].values



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)

tsne_results = tsne.fit_transform(X)



print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))



fig = {

    "data": [

        {

            "x": tsne_results[:, 0],

            "y": tsne_results[:, 1],

            "text": text2,

            "mode": "markers",

            "marker": {"size": "8", "color": y}

        }

    ],

    "layout": {

        "xaxis": {"title": "x-tsne"},

        "yaxis": {"title": "y-tsne"}

    }

}



py.iplot(fig, filename="test-graph2")
kmeans_X_Y_sne = pd.DataFrame({'pca1': tsne_results[:, 0], 'pca2': tsne_results[:, 1]})
def add_cluster_number_to_dataframe(model, df,pca_df):

    df = df.copy() # Prevent adding column in-place

    df['cluster'] = model.labels_ + 1

    df['cluster'] = 'cluster ' + df['cluster'].astype(str)

    

    df['pca1'] = pca_df['pca1']

    df['pca2'] = pca_df['pca2']

    

    df = df.sort_values(['cluster'])

    

    

    return df




kmeans = KMeans(n_clusters=3, random_state = 424, n_jobs=-1)

kmeans.fit(kmeans_X_Y)



data_frame_pca = add_cluster_number_to_dataframe(kmeans, data_frame, kmeans_X_Y)



data_frame_pca.head()

df_filtered_1 = data_frame_pca[data_frame_pca['cluster'] == "cluster 1"]

df_filtered_1_X = df_filtered_1['pca1'].values

df_filtered_1_Y = df_filtered_1['pca2'].values

df_filtered_1_text1 = df_filtered_1["Artist"] + " - " + df_filtered_1["Song_name"]

df_filtered_1_text2 = df_filtered_1_text1.values 



trace1 = {

  "mode": "markers", 

  "name": "Cluster 1", 

  "type": "scatter", 

    "x": df_filtered_1_X,

    "y": df_filtered_1_Y,

    "text":df_filtered_1_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "blue"

  }

}



df_filtered_2 = data_frame_pca[data_frame_pca['cluster'] == "cluster 2"]

df_filtered_2_X = df_filtered_2['pca1'].values

df_filtered_2_Y = df_filtered_2['pca2'].values

df_filtered_2_text1 = df_filtered_2["Artist"] + " - " + df_filtered_2["Song_name"]

df_filtered_2_text2 = df_filtered_2_text1.values 



trace2 = {

  "mode": "markers", 

  "name": "Cluster 2", 

  "type": "scatter", 

"x": df_filtered_2_X,

"y": df_filtered_2_Y,

      "text":df_filtered_2_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "green"

  }

}





df_filtered_3 = data_frame_pca[data_frame_pca['cluster'] == "cluster 3"]

df_filtered_3_X = df_filtered_3['pca1'].values

df_filtered_3_Y = df_filtered_3['pca2'].values

df_filtered_3_text1 = df_filtered_3["Artist"] + " - " + df_filtered_3["Song_name"]

df_filtered_3_text2 = df_filtered_3_text1.values 





trace3 = {

  "mode": "markers", 

  "name": "Cluster 3", 

  "type": "scatter", 

    "x": df_filtered_3_X,

    "y": df_filtered_3_Y,

      "text":df_filtered_3_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "red"

  }

}





data = [trace1, trace2,trace3]

layout = {

  "title": "PCA K-Means Clustering (k=3)", 

  "xaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "yaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "legend": {"font": {"size": 16}}, 

  "titlefont": {"size": 24}

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="test-graph4")
SSE = []

for i in range(2,11):

    model = KMeans(n_clusters= i, random_state=99)

    model.fit(kmeans_X_Y_sne)

    SSE.append(model.inertia_)

k = (range(2,11))

plt.title('Elbow curve')

plt.xlabel('k')

plt.ylabel('SSE')

plt.grid(True)

plt.plot(k, SSE)

plt.show()
kmeans = KMeans(n_clusters=3, random_state = 424, n_jobs=-1)

kmeans.fit(kmeans_X_Y_sne)



data_frame_sne = add_cluster_number_to_dataframe(kmeans, data_frame, kmeans_X_Y_sne)



data_frame_sne.head()

df_filtered_1 = data_frame_sne[data_frame_sne['cluster'] == "cluster 1"]

df_filtered_1_X = df_filtered_1['pca1'].values

df_filtered_1_Y = df_filtered_1['pca2'].values

df_filtered_1_text1 = df_filtered_1["Artist"] + " - " + df_filtered_1["Song_name"]

df_filtered_1_text2 = df_filtered_1_text1.values 



trace1 = {

  "mode": "markers", 

  "name": "Cluster 1", 

  "type": "scatter", 

    "x": df_filtered_1_X,

    "y": df_filtered_1_Y,

    "text":df_filtered_1_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "blue"

  }

}



df_filtered_2 = data_frame_sne[data_frame_sne['cluster'] == "cluster 2"]

df_filtered_2_X = df_filtered_2['pca1'].values

df_filtered_2_Y = df_filtered_2['pca2'].values

df_filtered_2_text1 = df_filtered_2["Artist"] + " - " + df_filtered_2["Song_name"]

df_filtered_2_text2 = df_filtered_2_text1.values 



trace2 = {

  "mode": "markers", 

  "name": "Cluster 2", 

  "type": "scatter", 

"x": df_filtered_2_X,

"y": df_filtered_2_Y,

      "text":df_filtered_2_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "green"

  }

}





df_filtered_3 = data_frame_sne[data_frame_sne['cluster'] == "cluster 3"]

df_filtered_3_X = df_filtered_3['pca1'].values

df_filtered_3_Y = df_filtered_3['pca2'].values

df_filtered_3_text1 = df_filtered_3["Artist"] + " - " + df_filtered_3["Song_name"]

df_filtered_3_text2 = df_filtered_3_text1.values 





trace3 = {

  "mode": "markers", 

  "name": "Cluster 3", 

  "type": "scatter", 

    "x": df_filtered_3_X,

    "y": df_filtered_3_Y,

      "text":df_filtered_3_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "red"

  }

}





data = [trace1, trace2,trace3]

layout = {

  "title": "t-SNE K-Means Clustering (k=3)", 

  "xaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "yaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "legend": {"font": {"size": 16}}, 

  "titlefont": {"size": 24}

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="test-graph5")
kmeans = KMeans(n_clusters=4, random_state = 200, n_jobs=-1)

kmeans.fit(kmeans_X_Y_sne)



data_frame_sne = add_cluster_number_to_dataframe(kmeans, data_frame, kmeans_X_Y_sne)



data_frame_sne.head()

df_filtered_1 = data_frame_sne[data_frame_sne['cluster'] == "cluster 1"]

df_filtered_1_X = df_filtered_1['pca1'].values

df_filtered_1_Y = df_filtered_1['pca2'].values

df_filtered_1_text1 = df_filtered_1["Artist"] + " - " + df_filtered_1["Song_name"]

df_filtered_1_text2 = df_filtered_1_text1.values 



trace1 = {

  "mode": "markers", 

  "name": "Cluster 1", 

  "type": "scatter", 

    "x": df_filtered_1_X,

    "y": df_filtered_1_Y,

    "text":df_filtered_1_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "blue"

  }

}



df_filtered_2 = data_frame_sne[data_frame_sne['cluster'] == "cluster 2"]

df_filtered_2_X = df_filtered_2['pca1'].values

df_filtered_2_Y = df_filtered_2['pca2'].values

df_filtered_2_text1 = df_filtered_2["Artist"] + " - " + df_filtered_2["Song_name"]

df_filtered_2_text2 = df_filtered_2_text1.values 



trace2 = {

  "mode": "markers", 

  "name": "Cluster 2", 

  "type": "scatter", 

"x": df_filtered_2_X,

"y": df_filtered_2_Y,

      "text":df_filtered_2_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "green"

  }

}





df_filtered_3 = data_frame_sne[data_frame_sne['cluster'] == "cluster 3"]

df_filtered_3_X = df_filtered_3['pca1'].values

df_filtered_3_Y = df_filtered_3['pca2'].values

df_filtered_3_text1 = df_filtered_3["Artist"] + " - " + df_filtered_3["Song_name"]

df_filtered_3_text2 = df_filtered_3_text1.values 





trace3 = {

  "mode": "markers", 

  "name": "Cluster 3", 

  "type": "scatter", 

    "x": df_filtered_3_X,

    "y": df_filtered_3_Y,

      "text":df_filtered_3_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "red"

  }

}







df_filtered_4 = data_frame_sne[data_frame_sne['cluster'] == "cluster 4"]

df_filtered_4_X = df_filtered_4['pca1'].values

df_filtered_4_Y = df_filtered_4['pca2'].values

df_filtered_4_text1 = df_filtered_4["Artist"] + " - " + df_filtered_4["Song_name"]

df_filtered_4_text2 = df_filtered_4_text1.values 





trace4 = {

  "mode": "markers", 

  "name": "Cluster 4", 

  "type": "scatter", 

    "x": df_filtered_4_X,

    "y": df_filtered_4_Y,

      "text":df_filtered_4_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "Yellow"

  }

}







data = [trace1, trace2,trace3,trace4]

layout = {

  "title": "t-SNE K-Means Clustering (k=4)", 

  "xaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "yaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "legend": {"font": {"size": 16}}, 

  "titlefont": {"size": 24}

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="test-graph6")
kmeans = KMeans(n_clusters=5, random_state = 424, n_jobs=-1)

kmeans.fit(kmeans_X_Y_sne)



data_frame_sne = add_cluster_number_to_dataframe(kmeans, data_frame, kmeans_X_Y_sne)



data_frame_sne.head()

df_filtered_1 = data_frame_sne[data_frame_sne['cluster'] == "cluster 1"]

df_filtered_1_X = df_filtered_1['pca1'].values

df_filtered_1_Y = df_filtered_1['pca2'].values

df_filtered_1_text1 = df_filtered_1["Artist"] + " - " + df_filtered_1["Song_name"]

df_filtered_1_text2 = df_filtered_1_text1.values 



trace1 = {

  "mode": "markers", 

  "name": "Cluster 1", 

  "type": "scatter", 

    "x": df_filtered_1_X,

    "y": df_filtered_1_Y,

    "text":df_filtered_1_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "blue"

  }

}



df_filtered_2 = data_frame_sne[data_frame_sne['cluster'] == "cluster 2"]

df_filtered_2_X = df_filtered_2['pca1'].values

df_filtered_2_Y = df_filtered_2['pca2'].values

df_filtered_2_text1 = df_filtered_2["Artist"] + " - " + df_filtered_2["Song_name"]

df_filtered_2_text2 = df_filtered_2_text1.values 



trace2 = {

  "mode": "markers", 

  "name": "Cluster 2", 

  "type": "scatter", 

"x": df_filtered_2_X,

"y": df_filtered_2_Y,

      "text":df_filtered_2_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "green"

  }

}





df_filtered_3 = data_frame_sne[data_frame_sne['cluster'] == "cluster 3"]

df_filtered_3_X = df_filtered_3['pca1'].values

df_filtered_3_Y = df_filtered_3['pca2'].values

df_filtered_3_text1 = df_filtered_3["Artist"] + " - " + df_filtered_3["Song_name"]

df_filtered_3_text2 = df_filtered_3_text1.values 





trace3 = {

  "mode": "markers", 

  "name": "Cluster 3", 

  "type": "scatter", 

    "x": df_filtered_3_X,

    "y": df_filtered_3_Y,

      "text":df_filtered_3_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "red"

  }

}







df_filtered_4 = data_frame_sne[data_frame_sne['cluster'] == "cluster 4"]

df_filtered_4_X = df_filtered_4['pca1'].values

df_filtered_4_Y = df_filtered_4['pca2'].values

df_filtered_4_text1 = df_filtered_4["Artist"] + " - " + df_filtered_4["Song_name"]

df_filtered_4_text2 = df_filtered_4_text1.values 





trace4 = {

  "mode": "markers", 

  "name": "Cluster 4", 

  "type": "scatter", 

    "x": df_filtered_4_X,

    "y": df_filtered_4_Y,

      "text":df_filtered_4_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "Yellow"

  }

}



df_filtered_5 = data_frame_sne[data_frame_sne['cluster'] == "cluster 5"]

df_filtered_5_X = df_filtered_5['pca1'].values

df_filtered_5_Y = df_filtered_5['pca2'].values

df_filtered_5_text1 = df_filtered_5["Artist"] + " - " + df_filtered_5["Song_name"]

df_filtered_5_text2 = df_filtered_5_text1.values 





trace5 = {

  "mode": "markers", 

  "name": "Cluster 5", 

  "type": "scatter", 

    "x": df_filtered_5_X,

    "y": df_filtered_5_Y,

      "text":df_filtered_5_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "Orange"

  }

}







data = [trace1, trace2,trace3,trace4,trace5]

layout = {

  "title": "t-SNE K-Means Clustering (k=5)", 

  "xaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "yaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "legend": {"font": {"size": 16}}, 

  "titlefont": {"size": 24}

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="test-graph7")
kmeans = KMeans(n_clusters=6, random_state = 424, n_jobs=-1,max_iter=500)

kmeans.fit(kmeans_X_Y_sne)



data_frame_sne = add_cluster_number_to_dataframe(kmeans, data_frame, kmeans_X_Y_sne)



data_frame_sne.head()

df_filtered_1 = data_frame_sne[data_frame_sne['cluster'] == "cluster 1"]

df_filtered_1_X = df_filtered_1['pca1'].values

df_filtered_1_Y = df_filtered_1['pca2'].values

df_filtered_1_text1 = df_filtered_1["Artist"] + " - " + df_filtered_1["Song_name"]

df_filtered_1_text2 = df_filtered_1_text1.values 



trace1 = {

  "mode": "markers", 

  "name": "Cluster 1", 

  "type": "scatter", 

    "x": df_filtered_1_X,

    "y": df_filtered_1_Y,

    "text":df_filtered_1_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "blue"

  }

}



df_filtered_2 = data_frame_sne[data_frame_sne['cluster'] == "cluster 2"]

df_filtered_2_X = df_filtered_2['pca1'].values

df_filtered_2_Y = df_filtered_2['pca2'].values

df_filtered_2_text1 = df_filtered_2["Artist"] + " - " + df_filtered_2["Song_name"]

df_filtered_2_text2 = df_filtered_2_text1.values 



trace2 = {

  "mode": "markers", 

  "name": "Cluster 2", 

  "type": "scatter", 

"x": df_filtered_2_X,

"y": df_filtered_2_Y,

      "text":df_filtered_2_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "green"

  }

}





df_filtered_3 = data_frame_sne[data_frame_sne['cluster'] == "cluster 3"]

df_filtered_3_X = df_filtered_3['pca1'].values

df_filtered_3_Y = df_filtered_3['pca2'].values

df_filtered_3_text1 = df_filtered_3["Artist"] + " - " + df_filtered_3["Song_name"]

df_filtered_3_text2 = df_filtered_3_text1.values 





trace3 = {

  "mode": "markers", 

  "name": "Cluster 3", 

  "type": "scatter", 

    "x": df_filtered_3_X,

    "y": df_filtered_3_Y,

      "text":df_filtered_3_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "red"

  }

}







df_filtered_4 = data_frame_sne[data_frame_sne['cluster'] == "cluster 4"]

df_filtered_4_X = df_filtered_4['pca1'].values

df_filtered_4_Y = df_filtered_4['pca2'].values

df_filtered_4_text1 = df_filtered_4["Artist"] + " - " + df_filtered_4["Song_name"]

df_filtered_4_text2 = df_filtered_4_text1.values 





trace4 = {

  "mode": "markers", 

  "name": "Cluster 4", 

  "type": "scatter", 

    "x": df_filtered_4_X,

    "y": df_filtered_4_Y,

      "text":df_filtered_4_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "Yellow"

  }

}



df_filtered_5 = data_frame_sne[data_frame_sne['cluster'] == "cluster 5"]

df_filtered_5_X = df_filtered_5['pca1'].values

df_filtered_5_Y = df_filtered_5['pca2'].values

df_filtered_5_text1 = df_filtered_5["Artist"] + " - " + df_filtered_5["Song_name"]

df_filtered_5_text2 = df_filtered_5_text1.values 





trace5 = {

  "mode": "markers", 

  "name": "Cluster 5", 

  "type": "scatter", 

    "x": df_filtered_5_X,

    "y": df_filtered_5_Y,

      "text":df_filtered_5_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "Orange"

  }

}



df_filtered_6 = data_frame_sne[data_frame_sne['cluster'] == "cluster 6"]

df_filtered_6_X = df_filtered_6['pca1'].values

df_filtered_6_Y = df_filtered_6['pca2'].values

df_filtered_6_text1 = df_filtered_6["Artist"] + " - " + df_filtered_6["Song_name"]

df_filtered_6_text2 = df_filtered_6_text1.values 





trace6 = {

  "mode": "markers", 

  "name": "Cluster 6", 

  "type": "scatter", 

    "x": df_filtered_6_X,

    "y": df_filtered_6_Y,

      "text":df_filtered_6_text2,

  "marker": {

    "line": {

      "color": "white", 

      "width": 0.5

    }, 

    "size": 12, 

    "color": "Brown"

  }

}



data = [trace1, trace2,trace3,trace4,trace5,trace6]

layout = {

  "title": "t-SNE K-Means Clustering (k=6)", 

  "xaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "yaxis": {

    "ticks": "", 

    "showgrid": True, 

    "zeroline": True, 

    "showticklabels": True

  }, 

  "legend": {"font": {"size": 16}}, 

  "titlefont": {"size": 24}

}



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="test-graph8")