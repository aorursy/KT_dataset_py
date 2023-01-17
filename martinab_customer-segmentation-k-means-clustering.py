# Importing libraries:



# Importing numpy, pandas, matplotlib and seaborn:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Imports for plotly:

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots





# To keep graph within the nobebook:

%matplotlib inline



# To hide warnings

import warnings

warnings.filterwarnings('ignore')
# Read data from saved csv file:

df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
# Display first five rows of dataframe:

header = ff.create_table(df.head())



header.show()
# Function to describe variables

def desc(df):

    d = pd.DataFrame(df.dtypes,columns=['Data_Types'])

    d = d.reset_index()

    d['Columns'] = d['index']

    d = d[['Columns','Data_Types']]

    d['Missing'] = df.isnull().sum().values    

    d['Uniques'] = df.nunique().values

    return d





descr = ff.create_table(desc(df))



descr.show()
# Explore dataframe's statistics (numerical values only):

desc = ff.create_table(df.describe())



desc.show()
age_df = pd.DataFrame(df.groupby(['Gender'])['Gender'].count())

#age_df.head()
#Gender distribution of shoppers:





data=go.Bar(x = age_df.index

           , y = age_df.Gender

           ,  marker=dict( color=['#FF0000', '#0000FF'])

           )







layout = go.Layout(title = 'Number of Customers split by Gender'

                   , xaxis = dict(title = 'Gender')

                   , yaxis = dict(title = 'Volume')

                  )



fig = go.Figure(data,layout)

fig.show()
# Box plot for Annual Income by Gender:



fig = px.box(df

             , x='Gender'

             , y='Annual Income (k$)'

             , points='all'

             , color='Gender'

             , title='Box plot of Annual Income by Gender'

             #, width = 950

            )



fig.show()
# Box Plot for Spending Score split by Gender:



fig = px.box(df

             , x='Gender'

             , y='Spending Score (1-100)'

             , points="all"

             , color='Gender'

             , title='Box plot of Spending Score by Gender'

            )



fig.show()
# Scatters for Age vs. Spending Score split by Gender:



fig = px.scatter(df

                 , x='Age'

                 , y='Spending Score (1-100)'

                 , color = 'Gender'

                 , facet_col='Gender'

                 , color_continuous_scale= ['#FF0000','#0000FF']   #px.colors.sequential.Viridis

                 , render_mode="webgl"

                # , width = 950

                )



fig.show()
# Scatter for Annual income vs. Speniding Score split by Gender:



fig = px.scatter(df

                 , x='Annual Income (k$)'

                 , y='Spending Score (1-100)'

                 , color = 'Gender'

                 , facet_col='Gender'

                 , color_continuous_scale= ['#FF0000','#0000FF']   #px.colors.sequential.Viridis

                 , render_mode="webgl"

                )



fig.show()
# Histograms, Distribution of Annual Income, Age and Spending Score:



fig = make_subplots(rows=1

                    , cols=3

                    ,subplot_titles=('Annual Income', 'Age', 'Spending Score'))





trace0 = go.Histogram(x=df['Annual Income (k$)']

                      , xbins=dict(start=15

                                   , end=140

                                   , size= 5)

                      , autobinx=False

                      , opacity=0.7

                     )

trace1 = go.Histogram(x=df['Age']

                      , xbins=dict(start=18

                                   , end=98

                                   , size= 5)

                      , autobinx=False

                      , opacity=0.7

                     )

trace2 = go.Histogram(x=df['Spending Score (1-100)']

                      , xbins=dict(start=1

                                   , end=100

                                   , size= 2)

                      , autobinx=False

                      , opacity=0.7

                     )



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 1, 3)



# Update xaxis properties

fig.update_xaxes(title_text='Annual Income (k$)', row=1, col=1)

fig.update_xaxes(title_text='Age', row=1, col=2)

fig.update_xaxes(title_text='Spending Score (1-100)',  row=1, col=3)





# Update yaxis properties

fig.update_yaxes(title_text='count', row=1, col=1)



# Update title and height

fig.update_layout(title_text='Distributions of ', height=600)





fig.show()
# Scatter graph for Annual Income vs Spending Score by Gender:



fig = px.scatter(df

                 , x='Annual Income (k$)'

                 , y='Spending Score (1-100)'

                 , color= 'Gender'

                 , marginal_y='rug'

                 , marginal_x='histogram'

                )

fig.show()
# Scatter graph for Annual Income vs. Spending Score by Age:



fig = px.scatter(df

                 , x='Annual Income (k$)'

                 , y='Spending Score (1-100)'

                 , color= 'Age'

                 , marginal_y='box'

                 , marginal_x='histogram'

                )

fig.show()
# Scatter graph for Age vs. Spending Score by Annual Income:



fig = px.scatter(df

                 , x='Age'

                 , y= 'Spending Score (1-100)'

                 , color= 'Annual Income (k$)'

                 , marginal_y='box'

                 , marginal_x='histogram'

                )

fig.show()
# 3D Scatter graph for Annual Income, Spending Score and Age:



fig = px.scatter_3d(df

                    , x='Annual Income (k$)'

                    , y='Spending Score (1-100)'

                    , z='Age'

                    , color='Annual Income (k$)'

                    , size='Spending Score (1-100)'

                   )



fig.show()
# Correlation matrix for Mall dataset features:



corr = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()



fig = go.Figure(data=go.Heatmap(

                   z=corr

                 , x=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

                 , y=['Spending Score (1-100)','Annual Income (k$)', 'Age' ]

                 , hoverongaps = False))



fig.update_layout(title='Correlation for Features of Mall data')





fig.show()
# Calculate inertia for k-clusters:



from sklearn.cluster import KMeans



X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]



l = []

for i in range(2, 11):

    kmeans = KMeans(n_clusters = i, random_state = 123)

    kmeans.fit(X)

    l.append(kmeans.inertia_)



df_1= pd.DataFrame(l, columns=['Inertia'])

df_1['k'] = df_1.index+2
# Line graph for Inertia vs. k-Clusters



fig = go.Figure()

fig.add_trace(go.Scatter(x=df_1.k

                         , y=df_1.Inertia

                         , mode='lines'

                         , name='inertia lines'

                        )

             )



fig.add_trace(go.Scatter(x=df_1.k, y=df_1.Inertia,

                    mode='markers', name='inertia point'))



fig.update_layout(title='The Total Sum of Squares Method'

                  , xaxis_title='k-clusters'

                  , yaxis_title='Internia'

                 )



fig.show()
# Calculate silhouette score for k-clusters:



from sklearn.metrics import silhouette_score

from sklearn import metrics



m = []



for i in range(2,11):

    kmeans = KMeans(n_clusters = i, random_state = 123)

    k_means = kmeans.fit(X)  

    labels = k_means.labels_

    sil_coeff = metrics.silhouette_score(X, labels,metric='euclidean')

    m.append(sil_coeff)





df_2= pd.DataFrame(m, columns=['Score'])

df_2['k'] = df_2.index+2

#print(df_2)

# Line graph for silhouette score vs. k-clusters



fig = go.Figure()

fig.add_trace(go.Scatter(x=df_2.k

                         , y=df_2.Score

                         , mode='lines'

                         , name=' score lines'

                        )

             )



fig.add_trace(go.Scatter(x=df_2.k

                         , y=df_2.Score

                         , mode='markers'

                         , name='score point'))



fig.update_layout(title='The Silhouette Score Method'

                  , xaxis_title='k-clusters'

                  , yaxis_title='Score'

                 )



fig.show()
# Fit data to KMeans clustering with 5 clusters, assign labels and cetroids: 



kmeans= KMeans(n_clusters = 5)

kmeans.fit(X)



labels = kmeans.labels_

centroids = kmeans.cluster_centers_

print('Cluster membership: \n{}'.format(labels))
print('Centroids: \n{}'.format(centroids))
clusters = labels.tolist()

X['clusters'] = clusters

X['Id'] = df['CustomerID']
# Display first 5 rows of dataset X:



head = ff.create_table(X.head())

head.show()
# Scatter graph of clusters, centroids are displayed as black markers:



X['clusters'] = X['clusters'].astype(str)



fig = px.scatter(X

                 , x='Annual Income (k$)'

                 , y='Spending Score (1-100)'

                 , color='clusters'

                 , title='Customer Segmentation (k=5)'

                )



fig.add_trace(go.Scatter(x=centroids[:,1], y=centroids[:,2],

                    mode='markers',

                    name='centroids',

                        marker=dict(

            color='black')))

              

              

fig.show()