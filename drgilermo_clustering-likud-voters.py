import numpy as np 

import pandas as pd 

import folium



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.cluster import SpectralClustering, DBSCAN, KMeans

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

from sklearn.metrics.pairwise import euclidean_distances



import networkx as nx

from community import best_partition



import plotly.offline as py

import plotly.graph_objs as go

import networkx as nx

import matplotlib.pyplot as plt



import os



py.init_notebook_mode(connected=True)
df = pd.read_csv(r'../input/likud_new.csv' ,  encoding='latin-1')

df.drop('Hebrew Location', axis=1, inplace=True)
for col in df.columns[3:]:

    if df[col].sum() < 1000:

        df = df.drop(col, axis=1)
locations = df['Location']

lat = df.Lat

lng = df.Lng

sizes = df.sum(axis=1)
df.drop(['Location', 'Lat', 'Lng'], axis=1).sum().sort_values(ascending=False)
normalized_df = df.drop(['Location', 'Lat', 'Lng'], axis=1)

normalized_df = normalized_df[normalized_df.sum().sort_values(ascending=False).index]

normalized_df = normalized_df.truediv(normalized_df.sum(axis=1), axis=0)
normalized_df = normalized_df.fillna(0)

plt.figure(figsize=(7, 7))

pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(normalized_df))

pca_df['locations'] = locations

pca_df['sizes'] = sizes



plt.scatter(pca_df[0], pca_df[1], s=50, alpha=0.6)

plt.xlabel('PC 1')

plt.ylabel('PC 2')

plt.show()
for i in np.arange(1,12):

    kmeans = KMeans(n_clusters=5).fit(normalized_df)

    print(i, silhouette_score(normalized_df, kmeans.predict(normalized_df)))
kmeans = KMeans(n_clusters=5, random_state=42).fit(normalized_df)



pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(normalized_df))

pca_df['label'] = kmeans.predict(normalized_df)

pca_df['locations'] = locations

pca_df['sizes'] = sizes
trace1 = go.Bar(

        x=normalized_df.columns,

        y=kmeans.cluster_centers_[0],

        name='Cluster 1'

        )



trace2 = go.Bar(

        x=normalized_df.columns,

        y=kmeans.cluster_centers_[1],

        name='Cluster 2'

        )



trace3 = go.Bar(

        x=normalized_df.columns,

        y=kmeans.cluster_centers_[2],

        name='Cluster 3'

        )



trace4 = go.Bar(

        x=normalized_df.columns,

        y=kmeans.cluster_centers_[3],

        name='Cluster 4'

        )



trace5 = go.Bar(

        x=normalized_df.columns,

        y=kmeans.cluster_centers_[4],

        name='Cluster 5'

        )



layout = go.Layout(

     title='<b>Cluster Centers</b>',

     titlefont=dict(size = 30, color='#7f7f7f'),

     hovermode='closest'

)



fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5], layout=layout)

py.iplot(fig)
traces = []



for label in sorted(pca_df.label.unique()):

    traces.append(go.Scatter(

            x=pca_df[pca_df.label == label][0],

            y=pca_df[pca_df.label == label][1],

            text=pca_df[pca_df.label == label]['locations'],

            mode='markers',

            hoverinfo='text',

            marker=dict(

                size=[np.sqrt(a)/5 for a in (pca_df[pca_df.label == label]['sizes'])],

                opacity=0.3,

          )

           )

                     )

    

layout = go.Layout(

        title= 'Likud voting clusters',

        hovermode='closest',

        xaxis=dict(

            autorange=True,

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False

        ),

        yaxis=dict(

            autorange=True,

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False

        ))

fig = go.Figure(data=traces, layout=layout)

py.iplot(fig)
ans = []

for i in range(len(df)):    

    ans.append(list(df.loc[i, df.columns[3:]].transpose().sort_values().index))

    

pca_df['winners'] = ans

pca_df['winners'] = pca_df.winners.apply(lambda x: x[::-1])



pca_df['first'] = pca_df.winners.apply(lambda x: x[0])

pca_df['second'] = pca_df.winners.apply(lambda x: x[1])

pca_df['third'] = pca_df.winners.apply(lambda x: x[2])

pca_df['forth'] = pca_df.winners.apply(lambda x: x[3])

pca_df['lat'] = lat

pca_df['lng'] = lng

pca_df['sizes'] = sizes

pca_df['locations'] = locations

pca_df = pca_df.fillna(0)

pca_df['locations'] = pca_df.locations.apply(lambda x: '' if x == 0 else x)




m = folium.Map(location=[32.13,34.8],zoom_start=9, tiles="CartoDB dark_matter" )



colors = ['blue', 'orange', 'green', 'crimson', 'purple']

for row in pca_df.iterrows():

    folium.Circle(

              location= (row[1].lat, row[1].lng),    

              radius=0.3*row[1].sizes,

              popup= '<b>' + row[1].locations + '</b><br>' + row[1]['first'] + '<br>' + row[1]['second'],

              color=colors[row[1].label],

              fill=True,

              fill_color=colors[row[1].label]

        ).add_to(m)

    

m
m.save('plot_data.html')

candidates = df.drop(['Location', 'Lat', 'Lng'], axis=1).transpose()

candidates = candidates[candidates.transpose().sum(axis=1).sort_values(ascending=False)[:32].index]

candidates = candidates.truediv(candidates.sum(axis=0), axis=1)

candidates = candidates.fillna(0)
sizes = candidates.sum(axis=1).values
C=np.corrcoef(candidates.transpose(),rowvar=0)

A=1*(C>0.4)

G=nx.Graph(A)

G=nx.relabel_nodes(G,dict(zip(G.nodes(),candidates.transpose().columns.values)))

communities=best_partition(G)
community_colors=dict(zip(np.unique(sorted(communities.values())), np.linspace(0, 1, len(np.unique(sorted(communities.values()))))))

node_coloring=[community_colors[communities[node]] for node in G.nodes()]



nx.pos=nx.fruchterman_reingold_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1, center=None)



nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=10)

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.show()
nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=10)

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.xlim([-0.5, 0.5])

plt.ylim([-0.5, 0.6])
candidates
for edge in list(G.edges):

    print('{ target: "' +edge[0] + '" ,source: "' + edge[1] + '", strength:  ' + str(candidates.transpose().corr()[edge[0]][edge[1]]) + '},')
communities
C=np.corrcoef(candidates.transpose(),rowvar=0)

A=1*(C>0.4)

G=nx.Graph(A)

pos=nx.spring_layout(G)#, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1, center=None)

labels = candidates.transpose().columns.values

N = len(G.nodes)

E = G.edges

Xv=[pos[k][0] for k in range(N)]

Yv=[pos[k][1] for k in range(N)]

Xed=[]

Yed=[]

for edge in E:

    Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]

    Yed+=[pos[edge[0]][1],pos[edge[1]][1], None] 

    

trace3=go.Scatter(x=Xed,

               y=Yed,

               mode='lines',

               line=dict(color='rgb(210,210,210)', width=1),

               hoverinfo='none'

               )

trace4=go.Scatter(x=Xv,

               y=Yv,

               mode='markers',

               name='net',

               marker=dict(symbol='circle-dot',

                             size=[40*np.sqrt(x) for x in sizes], 

                             color= node_coloring,

                            colorscale='Viridis',

                             line=dict(color='rgb(50,50,50)', width=0.5)

                             ),

               text=labels,

               hoverinfo='text'

               )





layout = go.Layout(

    title= 'Zoom in for a better view!',

        hovermode='closest',

        xaxis=dict(

            autorange=True,

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False

        ),

        yaxis=dict(

            autorange=True,

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False

    )

)

data1=[trace3, trace4]

fig1=go.Figure(data=data1, layout=layout)

# fig1['layout']['annotations'][0]['text']=annot

py.iplot(fig1,)