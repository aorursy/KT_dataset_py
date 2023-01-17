import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re



from scipy.stats import zscore



import scipy.cluster.hierarchy as sch

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering



from sklearn.metrics import silhouette_samples, silhouette_score



import matplotlib.cm as cm



from mpl_toolkits.mplot3d import Axes3D



import numpy as np

dataset = pd.read_csv('../input/Cricket.csv',encoding="cp1252")

dataset.head(15)
dataset.info()
dataset['HS']=dataset['HS'].str.strip('*')

dataset['4s']=dataset['4s'].str.strip('+')

dataset['6s']=dataset['6s'].str.strip('+') 
dataset['HS']=dataset.HS.astype('int64')

dataset['4s']=dataset['4s'].astype('float64')

dataset['6s']=dataset['6s'].astype('float64')
Names =[]

Country=[]

for x in dataset['Player']:

    a=  re.split("[()\xa0]", x)

    Names.append(a[0])

    b = re.split("/",a[2])

    for y in b:

        if(y!='Asia' and y!='ICC' and y!='Afr' and y!='IRE'):

            Country.append(y)







Names = pd.Series(Names, name='Player Name')

Country = pd.Series(Country, name='Country Name')

dataset['Player Name']= Names

dataset['Country']=Country

Start_Y =[]

End_Y=[]

for x in dataset['Span']:

    a=  re.split("-", x)

    Start_Y.append(int(a[0]))

    End_Y.append(int(a[1]))



Start_Y = pd.Series(Start_Y, name='Beginning Year')

End_Y = pd.Series(End_Y, name='Retire Year')



# dataset['Beginning Year']= Start_Y

# dataset['Retire Year']=End_Y



dataset['Career Span'] = End_Y - Start_Y

dataset.describe(include='all').transpose()
#Player Details, country vice

dataset.groupby(['Country','Player Name']).max()
sns.pairplot(dataset, diag_kind='kde')
#Top Runs Scored by each Country

Run_sum=dataset.groupby('Country').sum().sort_values(by='Runs').reset_index()

sns.lineplot(x='Country', y='Runs', data=Run_sum)
#Run Scored by Indian Players

##plt.figure(figsize=(16,8))

India_run = dataset[dataset['Country']=="INDIA"][['Player Name','Runs']].reset_index(drop='index')

sns.barplot(y='Player Name', x='Runs', data=India_run)
#Top Players who played maximum Matches from each country

Top_Mat=dataset[dataset['Mat'].isin(dataset.groupby('Country').max()['Mat'].values)]

sns.barplot(y='Player Name',x='Mat',data=Top_Mat,hue='Country',dodge=False)

plt.legend(loc=1)
# count = data_country.groupby('Country')['Mat'].max().reset_index()

# count



# player=[]

# data_country.reset_index(inplace=True)

# for k,l in zip(data_country.Country, data_country.Mat):

#     for i,j in zip(count.Country, count.Mat):

#         if i==k and j==l:

#             player.append(list(data_country[(data_country.Country == i) & (data_country.Mat ==j)]['Player Name']))

            

            

# play =[]

# for i in player:

#     for j in i:

#         play.append(j)

    

# count['Player Name'] = play

# count
#Top 5 Player who scored low runs

dataset['Run/Mat']=(dataset['Runs'] / dataset['Mat'])

sns.barplot(y='Player Name',x='Run/Mat', hue='Country', data=dataset.sort_values(ascending=True, by='Run/Mat').head(5), dodge=False)

plt.legend(loc=1)
#Career Span of player

sns.barplot(x='Career Span', y='Player Name', data= dataset.sort_values('Career Span',ascending=True).head(5))

plt.title('Player having short career span ')
#Player who played more in his career

sns.barplot(x='Career Span', y='Player Name', data= dataset.sort_values('Career Span',ascending=True).tail(5))

plt.title('Player having long career span ')
Top_Run=dataset[dataset['Runs'].isin(dataset.groupby(['Country'])['Runs'].max().values)].reset_index()

Top_Run=Top_Run[['Player Name','100','6s','Runs','Country']]

Top_Run
# Most 100 by batsmen

sns.barplot(x='100', y='Player Name',data=Top_Run,hue='Country',dodge=False)

plt.title('Most 100 by batsmen')
#Most 6s

sns.barplot(x='6s', y='Player Name',data=Top_Run,hue='Country',dodge=False)

plt.title('Most 6s hits')
plt.figure(figsize=(18,8))

sns.heatmap(dataset.corr(), annot=True)
dataset_sel = dataset[['Mat','Inns','Runs', 'BF','50','100','0','4s','6s','Career Span','Ave','Run/Mat','Player Name']]

dataset_sel.shape
X=dataset_sel.drop('Player Name',axis=1)
# Standardize the variables 

#using zscore for standardizing X



standard_data= X.apply(zscore)
cluster_range = range( 1, 15 )

cluster_errors = []



for num_clusters in cluster_range:

  clusters = KMeans( num_clusters )

  clusters.fit( standard_data )

  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:10]
plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
model_Km = KMeans(n_clusters=2)

model_Km.fit(standard_data)

pred_Km = model_Km.predict(standard_data)
centroids = model_Km.cluster_centers_

centroid_df = pd.DataFrame(centroids, columns = list(standard_data) )
centroid_x = centroids[:,0]

centroid_y = centroids[:,1]



xs = standard_data.iloc[:,0]

ys = standard_data.iloc[:,1]



plt.scatter(xs,ys,c=pred_Km,s=50)

plt.scatter(centroid_x,centroid_y,marker='D',c='r',s=40)



plt.xlabel('Match')

plt.ylabel('Inn')

plt.title('Match Vs Inn')

plt.show()
centroid_x = centroids[:,1]

centroid_y = centroids[:,2]



xs = standard_data.iloc[:,1]

ys = standard_data.iloc[:,2]



plt.scatter(xs,ys,c=pred_Km,s=50)

plt.scatter(centroid_x,centroid_y,marker='D',c='r',s=40)



plt.xlabel('Inn')

plt.ylabel('Runs')

plt.title('Runs Vs Inn')



plt.show()
centroid_x = centroids[:,2]

centroid_y = centroids[:,3]



xs = standard_data.iloc[:,2]

ys = standard_data.iloc[:,3]



plt.scatter(xs,ys,c=pred_Km,s=50)

plt.scatter(centroid_x,centroid_y,marker='D',c='r',s=40)



plt.xlabel('Runs')

plt.ylabel('BF')

plt.title('Runs Vs BF')

plt.legend()



plt.show()
# Compute the silhouette scores for each sample



def sil_score(n_clusters, silhouette_avg, X_scaled,cluster_labels,centers):

    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

    

    y_lower = 10

    

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)



    for i in range(n_clusters):

      # Aggregate the silhouette scores for samples belonging to

      # cluster i, and sort them

        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()



        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i





        cmap = cm.get_cmap("Spectral")

        color = cmap(float(i) / n_clusters)

        ax1.fill_betweenx(np.arange(y_lower, y_upper),

                        0, ith_cluster_silhouette_values,

                        facecolor=color, edgecolor=color, alpha=0.7)



          # Label the silhouette plots with their cluster numbers at the middle

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))



          # Compute the new y_lower for next plot

        y_lower = y_upper + 10  # 10 for the 0 samples



    ax1.set_title("The silhouette plot for the various clusters.")

    ax1.set_xlabel("The silhouette coefficient values")

    ax1.set_ylabel("Cluster label")



      # The vertical line for average silhoutte score of all the values

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")



    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])



      # 2nd Plot showing the actual clusters formed

    cmap = cm.get_cmap("Spectral")

    colors = cmap(cluster_labels.astype(float) / n_clusters)

    ax2.scatter(X_scaled.iloc[:, 0], X_scaled.iloc[:, 1], marker='.', s=30, lw=2, c=colors)

    

      # Labeling the clusters

      # Draw white circles at cluster centers

    ax2.scatter(centers[:, 0], centers[:, 1],

              marker='o', c="white", alpha=1, s=200)



    for i, c in enumerate(centers):

          ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)



    ax2.set_title("The visualization of the clustered data.")

    ax2.set_xlabel("Feature space for the 1st feature")

    ax2.set_ylabel("Feature space for the 2nd feature")



    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "

                "with n_clusters = %d" % n_clusters),

               fontsize=14, fontweight='bold')



    plt.show()


Inertia =[]



def sil_coeff(no_clusters):

    # Apply your clustering algorithm of choice to the reduced data 

    clusterer_1 = KMeans(n_clusters=no_clusters, random_state=101 )

    clusterer_1.fit(standard_data)

    

    # Predict the cluster for each data point

    preds_1 = clusterer_1.predict(standard_data)

    

    # Find the cluster centers

    centers_1 = clusterer_1.cluster_centers_

    

    # Calculate the mean silhouette coefficient for the number of clusters chosen

    score = silhouette_score(standard_data, preds_1)

    

#     #Cluster Error

#     error = clusterer_1.inertia_

#     Inertia.append(error)

    

    print("silhouette coefficient for `{}` clusters => {:.4f}".format(no_clusters, score))

    sil_score(no_clusters, score, standard_data,preds_1,centers_1)
clusters_range = range(2,15)

for i in clusters_range:

    sil_coeff(i)
linked=sch.linkage(standard_data, method="ward")

plt.figure(figsize=(15, 5))  

sch.dendrogram(linked,  

            orientation='top',

            distance_sort='descending',

            show_leaf_counts=True)

plt.title('Dendrogram')

plt.show()





# cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)

# sns.clustermap(standard_data, cmap=cmap, linewidths=.5)
model_Aglo = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  

pred=model_Aglo.fit_predict(standard_data)
x=standard_data.iloc[:,[0,1]].values
plt.scatter(x[pred==0,0],x[pred==0,1],c='r',label='cluster1')

plt.scatter(x[pred==1,0],x[pred==1,1],c='g',label='cluster2')

plt.scatter(centroids[:,0],centroids[:,1],marker='D',c='c',s=55)

plt.legend()

plt.show()
standard_data['label']=model_Km.labels_

standard_data['Player']=dataset_sel['Player Name']
standard_data.head(10)