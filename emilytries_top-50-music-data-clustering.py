import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import statsmodels.formula.api as smf



from warnings import filterwarnings

filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Importing Dataset

data=pd.read_csv('../input/top50.csv', encoding='ISO-8859-1')

data.head()
print(data.shape)
data=data.dropna(how='all')
data.info()
print(data.groupby('Genre').size())
data.isnull().sum().sum()
data=data.sort_values(['Unnamed: 0'])

data=data.reindex(data['Unnamed: 0'])

data=data.drop("Unnamed: 0",axis=1)

data.head()
data.describe().T
data=data.loc[:49,:]
#Rename Column

data=data.rename(columns={"Loudness..dB..": "Loudness", 

                          "Acousticness..": "Acousticness",

                          "Speechiness.":"Speechiness",

                          "Valence.":"Valence",

                          "Length.":"Length"})
final=data.copy()
#Correlation Matrix Between Numeric Features

plt.figure(figsize = (16,7))



corrMatrix = final.corr()

sns.heatmap(corrMatrix, annot=True, linewidths=.5);
#Grouping Some Features According To Genre

a=final[['Genre', 'Popularity','Energy', 'Length','Liveness','Acousticness']].groupby(

    ['Genre'], as_index=False).mean().sort_values(by='Energy', ascending=True)

a
sorted_energy=final.sort_values(by=['Energy'])
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(

    x=sorted_energy['Energy'],

    y=sorted_energy['Loudness'],

    name="Energy and Loudness"       # this sets its legend entry

))





fig.add_trace(go.Scatter(

    x=sorted_energy['Energy'],

    y=sorted_energy['Acousticness'],

    name="Energy and Acousticness"

))



fig.update_layout(

    title="Acousticness-Loudness values according to Energy",

    xaxis_title="Energy",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)



fig.show()
import plotly.express as px



fig = px.bar(a, x='Genre', y='Popularity',

             hover_data=['Energy', 'Length'],

             color='Energy',height=400)



fig.update_layout(

    title="Popularity and energy comparison by Genre",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)



fig.show()
fig = px.bar(a, y='Acousticness', x='Genre').update_xaxes(categoryorder='total ascending')



fig.update_layout(

    title="Acousticness comparison by Genre",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)

fig.show()
#Add to new column according to Genre



final['GeneralGenre']=['hip hop' if each =='atl hip hop'

                      else 'hip hop' if each =='canadian hip hop'

                      else 'hip hop' if each == 'trap music'

                      else 'pop' if each == 'australian pop'

                      else 'pop' if each == 'boy band'

                      else 'pop' if each == 'canadian pop'

                      else 'pop' if each == 'dance pop'

                      else 'pop' if each == 'panamanian pop'

                      else 'pop' if each == 'pop'

                      else 'pop' if each == 'pop house'

                      else 'electronic' if each == 'big room'

                      else 'electronic' if each == 'brostep'

                      else 'electronic' if each == 'edm'

                      else 'electronic' if each == 'electropop'

                      else 'rap' if each == 'country rap'

                      else 'rap' if each == 'dfw rap'

                      else 'hip hop' if each == 'hip hop'

                      else 'latin' if each == 'latin'

                      else 'r&b' if each == 'r&n en espanol'

                      else 'raggae' for each in final['Genre']]
# histogram

final.hist()

plt.gcf().set_size_inches(15, 15)    #Thanks to this graphic, we can see the feature is right or left skewed.

plt.show()



from pandas.plotting import scatter_matrix



# scatter plot matrix

scatter_matrix(final)

plt.gcf().set_size_inches(15, 15)

plt.show()
color_list = ['red' if i=='electronic' 

              else 'green' if i=='escape room' 

              else 'blue' if i == 'hip hop' 

              else 'purple' if i == 'latin'

              else 'darksalmon' if i == 'pop'

              else 'darkcyan' if i == 'raggae'

              else 'greenyellow' for i in final.loc[:,'Genre']]

pd.plotting.scatter_matrix(final.loc[:,['Energy','Danceability','Length','Popularity']],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=1,

                                       s = 200,

                                       marker = '+',

                                       edgecolor= "black")

plt.show()
import plotly.express as px



fig = px.scatter(final, x="Beats.Per.Minute", y="Valence",size='Acousticness'

                 ,color="GeneralGenre")

fig.show()
#Box Plot Each Features



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True)



trace0 = go.Box(

    y=final['Beats.Per.Minute'],

    name = 'Beats.Per.Minute'

)

trace1 = go.Box(

    y=final['Energy'],

    name = 'Energy'

)

trace2 = go.Box(

    y=final['Danceability'],

    name = 'Danceability'

)

trace3 = go.Box(

    y=final['Loudness'],

    name = 'Loudness'

)

trace4 = go.Box(

    y=final['Liveness'],

    name = 'Liveness'

)

trace5 = go.Box(

    y=final['Valence'],

    name = 'Valence'

)

trace6 = go.Box(

    y=final['Loudness'],

    name = 'Loudness'

)

trace7 = go.Box(

    y=final['Length'],

    name = 'Length'

)

trace8 = go.Box(

    y=final['Acousticness'],

    name = 'Acousticness'

)

trace9 = go.Box(

    y=final['Speechiness'],

    name = 'Speechiness'

)

trace10 = go.Box(

    y=final['Popularity'],

    name = 'Popularity'

)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8,trace9,trace10]



fig = go.Figure(data=data)



py.offline.iplot(fig)
pd=pd.DataFrame(final['Genre'].value_counts())

pd.head(2)
pd.rename(columns={"Genre": "Count"},inplace=True)

pd['Genre'] = pd.index

pd['Ratio'] = pd['Count']/pd['Count'].sum() #Add count ratio for each genre
pd.head()
pd['Ratio'] = pd['Ratio'].astype(float)
import matplotlib.colors as colors

labels_list = pd['Genre']

colors_list = list(colors._colors_full_map.values())



# Plot

plt.figure(figsize=(20,10))

plt.pie(pd['Ratio'], colors=colors_list[0:20],

autopct='%1.1f%%', shadow=True, startangle=50,

       labels=labels_list)



plt.axis('equal')

plt.title('Distribution of the % of Genre')

plt.show()

final.info()
final2=final.drop(columns=['Track.Name', 'Artist.Name', 'GeneralGenre', 'Genre'])
final2.head()
#Rename Column

final2=final2.rename(columns={"Beats.Per.Minute": "BeatsPerMinute"})
#Standardizing to all data



from sklearn.preprocessing import StandardScaler

final2[['BeatsPerMinute', 'Energy',

        'Danceability','Loudness',

        'Liveness','Valence',

        'Length','Acousticness',

        'Speechiness','Popularity']] = StandardScaler().fit_transform(final2[['BeatsPerMinute','Energy',

                                                                              'Danceability','Loudness',

                                                                              'Liveness','Valence',

                                                                              'Length','Acousticness',

                                                                              'Speechiness','Popularity']])
final2.tail()
# KMeans Clustering

!pip install yellowbrick

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans, k=(2,10))

visualizer.fit(final2)

visualizer.poof()
# PCA variance

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline





scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(final2)



plt.bar(range(pca.n_components_), pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=7)

kmeans
k_fit=kmeans.fit(final2)
clusters=k_fit.labels_

clusters
final2['segment']=clusters+1

final2.head()
import mpl_toolkits

from mpl_toolkits.mplot3d import Axes3D



centers=kmeans.cluster_centers_



plt.rcParams['figure.figsize'] = (16,9)

fig=plt.figure()



ax=Axes3D(fig)

ax.scatter(final2.iloc[:,0], final2.iloc[:,1], final2.iloc[:,2]);
joined_df_merge = final2.merge(final, how='left', 

                                      left_index=True,

                                      right_index=True)
joined_df_merge[['segment','Genre','GeneralGenre']]
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(final2,method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size =5)

plt.show()