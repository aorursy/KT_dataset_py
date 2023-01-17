import pandas as pd

import numpy as np
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.head()
df.describe()
import plotly.graph_objects as go

from plotly.subplots import make_subplots



df_plots=df.select_dtypes(exclude="object")



fig=make_subplots(rows=2, cols=2,subplot_titles=df_plots.columns)



index=0



for i in range(1,3):

    for j in range(1,3):

        data=df[df_plots.columns[index]]

        trace=go.Histogram(x=data)

        fig.append_trace(trace,i,j)

        index+=1

        

fig.update_layout(height=900,width=1200,title_text="Numerical Attributes")
def plot_hist_num(df):

    df_plots=df.select_dtypes(exclude="object")



    fig=make_subplots(rows=1, cols=3,subplot_titles=df_plots.columns)



    index=0



    for i in range(1,2):

        for j in range(1,4):

            data=df[df_plots.columns[index]]

            trace=go.Histogram(x=data)

            fig.append_trace(trace,i,j)

            index+=1

        

    fig.update_layout(height=300,width=900,title_text="Numerical Attributes")

    fig.show()
df.isnull().sum()
df.skew()
df.drop(['CustomerID','Gender'],axis=1,inplace=True)

df.head(2)
df.shape
from sklearn.preprocessing import PowerTransformer



pt=PowerTransformer()



#PowerTransformer() takes the input of the form {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)

df_transformed=pt.fit_transform(df.values.reshape(-1,3))
#convert array to dataframe to plot it

pd_df_transformed=pd.DataFrame(df_transformed,columns=df.columns)



#plot the histogram to see change in distrbution

plot_hist_num(pd_df_transformed)
pd_df_transformed.describe()
from sklearn.preprocessing import QuantileTransformer



qt=QuantileTransformer(random_state=0)



#PowerTransformer() takes the input of the form {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)

df_quantile_transformed=qt.fit_transform(df.values.reshape(-1,3))
#convert array to dataframe to plot it

pd_df_quantile_transformed=pd.DataFrame(df_quantile_transformed,columns=df.columns)



#plot the histogram to see change in distrbution

plot_hist_num(pd_df_quantile_transformed)
'''

from sklearn.manifold import TSNE



# Project the data: this step will take several seconds

tsne = TSNE(n_components=2, init='random', random_state=0)



#Fit_transform() accpets input of the type array, shape (n_samples, n_features) 

sne_df_transformed = tsne.fit_transform(df_transformed)

'''
'''

#plot the clusters obtained from t-SNE

fig = go.Figure(data=go.Scatter(x=sne_df_transformed.T[0],

                                y=sne_df_transformed.T[1],

                                mode='markers')) 



fig.update_layout(title='t-SNE distribution of data')

fig.show()

'''
from sklearn.manifold import MDS

mds = MDS(n_components = 2)



mds_df_transformed = mds.fit_transform(df_transformed)
#plot the clusters obtained from K Means

fig = go.Figure(data=go.Scatter(x=mds_df_transformed.T[0],

                                y=mds_df_transformed.T[1],

                                mode='markers')) 



fig.update_layout(title='MDS Transformed data for visualization')

fig.show()

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

import plotly.express as px
silhouette_k_means=[]



for k in range(2,10):

    k_test=KMeans(n_clusters=k)

    cluster_labels=k_test.fit_predict(df_transformed)

    silhouette_avg = silhouette_score(df_transformed, cluster_labels)

    silhouette_k_means.append(silhouette_avg)

    

px.line(x=range(2,10),y=silhouette_k_means)
'''

We will keep a track of the silhouette score and the model using silhouette_score_compiled

We will keep a track of the DB score and the model using db_score_compiled

'''



silhouette_score_compiled={}

db_score_compiled={}
#plug in optimal number of clusters 



k_means=KMeans(n_clusters=6)

kmeans_labels=k_means.fit_predict(df_transformed)

silhouette_score_compiled['K Means'] = silhouette_score(df_transformed, kmeans_labels)

db_score_compiled['K Means']=metrics.davies_bouldin_score(df_transformed,kmeans_labels)

print(silhouette_score_compiled)
#plot the clusters obtained from K Means

fig = go.Figure(data=go.Scatter(x=mds_df_transformed.T[0],

                                y=mds_df_transformed.T[1],

                                mode='markers',

                                marker_color=kmeans_labels,text=kmeans_labels)) 



fig.update_layout(title='K Means')

fig.show()

from sklearn.cluster import MeanShift

from sklearn.cluster import estimate_bandwidth



est_bandwidth = estimate_bandwidth(df_transformed,quantile=0.1,n_samples=10000)

ms = MeanShift(bandwidth= est_bandwidth)

ms_labels=ms.fit_predict(df_transformed)

silhouette_score_compiled['Mean Shift'] = silhouette_score(df_transformed, ms_labels)

db_score_compiled['Mean Shift']=metrics.davies_bouldin_score(df_transformed,ms_labels)

print(silhouette_score_compiled)
#plot the clusters obtained from Mean Shift

fig = go.Figure(data=go.Scatter(x=mds_df_transformed.T[0],

                                y=mds_df_transformed.T[1],

                                mode='markers',

                                marker_color=ms_labels,text=ms_labels)) 



fig.update_layout(title='Mean Shift')

fig.show()

from sklearn.cluster import DBSCAN 

from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)

nbrs = neigh.fit(df_transformed)

distances, indices = nbrs.kneighbors(df_transformed)



#sort and plot the results

distances = np.sort(distances, axis=0)

distances = distances[:,1]

plt.plot(distances)
np.log(len(df_transformed))
# we will select the optimal values using grid search method

from sklearn import metrics



db_results=pd.DataFrame(columns=['Eps','Min_Samples','Number of Cluster','Silhouette Score'])

for i in range(1,12):

    for j in range(1,12):

        dbscan_cluster = DBSCAN(eps=i*0.2, min_samples=j)

        clusters=dbscan_cluster.fit_predict(df_transformed)

        if len(np.unique(clusters))>2:

              db_results=db_results.append({'Eps':i*0.2,

                                      'Min_Samples':j,

                                      'Number of Cluster':len(np.unique(clusters)),

                                      'Silhouette Score':metrics.silhouette_score(df_transformed,clusters),

                                      'Davies Bouldin Score':metrics.davies_bouldin_score(df_transformed,clusters)}, ignore_index=True)
db_results.sort_values('Silhouette Score',ascending=False)[:5]
#choosing min_samples as 6 and eps as 0.6

dbscan = DBSCAN(eps=0.6,min_samples=6)

dbscan_labels= dbscan.fit_predict(df_transformed)

silhouette_score_compiled['DBSCAN'] = silhouette_score(df_transformed, dbscan_labels)

db_score_compiled['DBSCAN']=metrics.davies_bouldin_score(df_transformed,dbscan_labels)

print(silhouette_score_compiled)
#plot the clusters obtained from DBSCAN

fig = go.Figure(data=go.Scatter(x=mds_df_transformed.T[0],

                                y=mds_df_transformed.T[1],

                                mode='markers',

                                marker_color=dbscan_labels,text=dbscan_labels)) 



fig.update_layout(title='DBSCAN')

fig.show()

from sklearn.mixture import GaussianMixture

from sklearn import metrics
parameters=['full','tied','diag','spherical']

n_clusters=np.arange(1,10)

results_=pd.DataFrame(columns=['Covariance Type','Number of Cluster','Silhouette Score','Davies Bouldin Score'])

for i in parameters:

    for j in n_clusters:

        gmm_cluster=GaussianMixture(n_components=j,covariance_type=i,random_state=123)

        clusters=gmm_cluster.fit_predict(df_transformed)

        if len(np.unique(clusters))>=2:

            results_=results_.append({"Covariance Type":i,'Number of Cluster':j,"Silhouette Score":metrics.silhouette_score(df_transformed,clusters),

                                    'Davies Bouldin Score':metrics.davies_bouldin_score(df_transformed,clusters)}

                                   ,ignore_index=True)
results_.sort_values('Silhouette Score',ascending=False)[:5]
gmm_labels = GaussianMixture(n_components=7,covariance_type='tied').fit_predict(df_transformed)

silhouette_score_compiled['GMM'] = silhouette_score(df_transformed, gmm_labels)

db_score_compiled['GMM']=metrics.davies_bouldin_score(df_transformed,gmm_labels)

print(silhouette_score_compiled)
#plot the clusters obtained from GMM

fig = go.Figure(data=go.Scatter(x=mds_df_transformed.T[0],

                                y=mds_df_transformed.T[1],

                                mode='markers',

                                marker_color=gmm_labels,text=gmm_labels)) 



fig.update_layout(title='GMM')

fig.show()

from sklearn.cluster import AgglomerativeClustering
parameters=['ward', 'complete', 'average', 'single']

n_clusters=np.arange(1,10)

agh_cluster_results_=pd.DataFrame(columns=['Linkage Type','Number of Cluster','Silhouette Score','Davies Bouldin Score'])

for i in parameters:

    for j in n_clusters:

        agh_cluster=AgglomerativeClustering(n_clusters=j,linkage=i)

        clusters=agh_cluster.fit_predict(df_transformed)

        if len(np.unique(clusters))>=2:

            agh_cluster_results_=agh_cluster_results_.append({"Linkage Type":i,'Number of Cluster':j,"Silhouette Score":metrics.silhouette_score(df_transformed,clusters),

                                    'Davies Bouldin Score':metrics.davies_bouldin_score(df_transformed,clusters)}

                                   ,ignore_index=True)
agh_cluster_results_.sort_values('Silhouette Score',ascending=False)[:5]
agh_labels=AgglomerativeClustering(n_clusters=8,linkage='average').fit_predict(df_transformed)

silhouette_score_compiled['Agglomerative Hierarchical Clustering'] = silhouette_score(df_transformed, agh_labels)

db_score_compiled['Agglomerative Hierarchical Clustering']=metrics.davies_bouldin_score(df_transformed,agh_labels)

print(silhouette_score_compiled)
#plot the clusters obtained from Agglomerative Hierarchical Clustering

fig = go.Figure(data=go.Scatter(x=mds_df_transformed.T[0],

                                y=mds_df_transformed.T[1],

                                mode='markers',

                                marker_color=agh_labels,text=agh_labels)) 



fig.update_layout(title='Agglomerative Hierarchical Clustering')

fig.show()

ss_df = pd.DataFrame(list(silhouette_score_compiled.items()),columns = ['Algo','Silhouette Score']) 

db_df = pd.DataFrame(list(db_score_compiled.items()),columns = ['Algo','Davies Bouldin Score']) 

final_results=pd.merge(ss_df,db_df,left_on="Algo",right_on="Algo")

final_results.sort_values('Silhouette Score',ascending=False)
df['Final Clusters']=kmeans_labels

df.head(4)
df['Final Clusters'].value_counts().index.sort_values(ascending=True)
age=[]

income=[]

spend=[]

cluster_k=[]

for i in df['Final Clusters'].value_counts().index.sort_values(ascending=True):

    df_test=df[df['Final Clusters']==i]

    cluster_k.append(i)

    age.append(round(df_test['Age'].mean(),0))

    income.append(round(df_test['Annual Income (k$)'].mean(),0))

    spend.append(round(df_test['Spending Score (1-100)'].mean(),0))
d={'CLuster':cluster_k,'Age':age,'Income(k$)':income,'Spending score':spend}

df_cluster_result=pd.DataFrame(d)

df_cluster_result
df_plots=df_cluster_result[["Age","Income(k$)","Spending score"]]

fig=make_subplots(rows=1, cols=3,subplot_titles=df_plots.columns)



index=0



for i in range(1,2):

    for j in range(1,4):

        data=df_cluster_result[df_plots.columns[index]]

        trace=go.Box(x=data)

        fig.append_trace(trace,i,j)

        index+=1

        

fig.update_layout(height=300,width=900,title_text="Boxplot of features of final Clusters")

fig.show()
trace1 = go.Scatter3d(

    x= df['Age'],

    y= df['Spending Score (1-100)'],

    z= df['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = df['Final Clusters'], 

        size= 10,

        line=dict(

            color= df['Final Clusters'],

            width= 12

        ),

        opacity=0.8

     )

)

data1 = [trace1]



layout = go.Layout(

    title = 'Character vs Gender vs Alive or not',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ),

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)



fig = go.Figure(data = data1, layout = layout)

fig.show("notebook")
