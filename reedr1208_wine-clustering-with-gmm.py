%matplotlib notebook

import numpy as np

import pandas as pd

import random

from IPython.display import display #for displaying dataframes nicely

import matplotlib.pyplot as plt





random.seed(137)
#combine red and white datasets

reds= pd.read_csv('../input/wine-quality-selection/winequality-red.csv')

reds['type']= 'red'

whites= pd.read_csv('../input/wine-quality-selection/winequality-white.csv')

whites['type']='white'



#drop the "quality" column as it is not a physical property that we are interested in

wines= pd.concat([reds, whites])

wines.drop(columns=['quality'], inplace=True)
print('rows, columns: {}'.format(wines.shape))

display(wines.head())
wines.describe()
#Use a heatmap to visualize the greatest correlations

plt.style.use('default')

plt.figure(figsize=(7, 7))

plt.imshow(wines.corr(), cmap='Reds', interpolation= 'nearest')

plt.xticks(np.arange(len(wines.corr().index.values)), wines.corr().index.values, fontsize=12, rotation=-60)

plt.yticks(np.arange(len(wines.corr().index.values)), wines.corr().index.values, fontsize=12)

plt.title('Heatmap of Correlations')

plt.tight_layout()

plt.show()
#View the numeric data corresponding to the above

wines.corr()
#Fetch top correlations



#Return a copy of dataframe with only metric columns

def drop_dims(df):

    df= df.copy()

    for i, e in zip(df.columns.values, df.dtypes):

        if e not in [np.float64, np.int64]:

            df.drop(i, inplace=True, axis=1)

    return df



#Every pair of metrics shows up twice

#This function removes one version of each pair (along with pricipal axis in which all correlations equal 1.0)

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



#return the highest correlations

def get_top_abs_correlations(df, n=3):

    df= drop_dims(df)

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations\n")

display(get_top_abs_correlations(wines))
#create a subplot showing a histogram for each metric

def show_metric_dist(df, title):

    plt.style.use('ggplot')

    plt.rcParams['figure.figsize']= [10, 8]

    fig = plt.figure()

    plt.subplots_adjust(hspace=0.4)

    

    fig.suptitle(title, fontsize=20, y=0.98)

    j=1

    for i, e in zip(df.columns.values, df.dtypes):

        if e not in [np.float64, np.int64]:

            continue

        ax=fig.add_subplot(3, 4, j)

        df[i].hist(bins=50, color='maroon')

        ax.set_title(i)

        j+=1

        

    plt.show()





show_metric_dist(wines, title='Metric Distributions (Before Normalizing & Scaling)')

#test alternative normalization

from sklearn.preprocessing import MinMaxScaler

from scipy import stats



#scale to gaussian

wines_mets= drop_dims(wines)

wines_scaled = wines_mets.apply(lambda x: stats.boxcox(x+1e-8)[0], axis=0)



scaler = MinMaxScaler()



wines_norm = pd.DataFrame(wines_scaled)



wines_norm = pd.DataFrame(scaler.fit_transform(wines_scaled), columns=wines_scaled.columns)



# Show an example of a record with scaling applied

display(wines_norm.describe())
show_metric_dist(wines_norm, title='Metric Distributions (After Normalizing & Scaling)')

# Apply PCA by fitting the data with the same number of dimensions as features

from sklearn.decomposition import PCA

pca = PCA(n_components=wines_norm.shape[1], random_state=51)

pca.fit(wines_norm)



#Transform wines_norm using the PCA fit above

pca_samples = pca.transform(wines_norm)
for i in range(wines_norm.shape[1]):



    first_n = pca.explained_variance_ratio_[0:i+1].sum()*100

    print('Percent variance explained by first {} components: {:.1f}%'.format(i+1, first_n))





print('\nFirst principle component contributions:\n')

first_comp= zip(wines_norm.columns.values, pca.components_[0])



for i, j in first_comp:

    print(i, '%.3f' % j)

pca_3d = PCA(n_components=3, random_state=51)

pca_3d.fit(wines_norm)



#transform wines_norm using the PCA fit above

pca_samples_3d = pca_3d.transform(wines_norm)
#visualize our PCA transformed data

from mpl_toolkits.mplot3d import Axes3D



plt.style.use('fivethirtyeight')

fig = plt.figure(figsize= (10, 10))

ax = Axes3D(fig)



ax.scatter(pca_samples_3d[:,0], pca_samples_3d[:,1], pca_samples_3d[:,2], alpha=0.4, color= 'b')

ax.set_xticklabels([])

ax.set_yticklabels([])

ax.set_zticklabels([])



plt.show()



from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score



max_k=20



sil_scores=[]

for i in range(2, max_k+1):

    clusterer = GaussianMixture(n_components=i, random_state=51, n_init=5)

    clusterer.fit(wines_norm)



    #Predict the cluster for each data point

    preds = clusterer.predict(wines_norm)



    #Find the cluster centers

    centers = clusterer.means_



    #Predict the cluster for each transformed sample data point

    sample_preds = clusterer.predict(wines_norm)



    #Calculate the mean silhouette coefficient for the number of clusters chosen

    score = silhouette_score(wines_norm, preds)

    sil_scores.append(score)

    

sil_scores= pd.Series(sil_scores, index= range(2,max_k+1))

max_score= sil_scores.max()

n_clusters= sil_scores.idxmax()

print('Max Silhouette Score: {:.3f}'.format(max_score))

print('Number of clusters: {}\n'.format(max_k))



print('First 3 Silhouette Scores')

print(sil_scores[0:3])



#refit the model to the K with the max silhouette score

clusterer = GaussianMixture(n_components=n_clusters, random_state=51, n_init=5)

clusterer.fit(wines_norm)



#Predict the cluster for each data point

preds = clusterer.predict(wines_norm)



#Find the cluster centers

centers = clusterer.means_
plt.style.use('ggplot')

plt.figure(figsize=(10,8))

plt.title('Silhouette Score vs. Number of Clusters', fontsize=14)

plt.ylabel('Silhouette Score')

plt.xlabel('Number of Clusters')

plt.xticks(np.arange(2, max_k+1, 1))

plt.plot(sil_scores.index.values, sil_scores)

#append cluster labels

pca_3d_clusters= np.append(pca_samples_3d, preds.reshape(-1, 1), axis=1)

#append wine type (red, white)

pca_3d_clusters= np.append(pca_3d_clusters, np.asarray(wines['type']).reshape(-1, 1), axis=1)

from mpl_toolkits.mplot3d import Axes3D



plt.style.use('fivethirtyeight')

fig = plt.figure(figsize= (10, 10))

ax = Axes3D(fig)



mapping= {0:'b', 1:'c', 2:'g'}

mapping= {0:'r', 1:'c', 2:'b'}

colors= [mapping[x] for x in preds]

ax.scatter(pca_3d_clusters[:,0], pca_3d_clusters[:,1], pca_3d_clusters[:,2], alpha=0.4, color= colors, marker= 'o')

ax.set_xticklabels([])

ax.set_yticklabels([])

ax.set_zticklabels([])



plt.show()
from mpl_toolkits.mplot3d import Axes3D



plt.style.use('ggplot')

plt.rcParams['figure.figsize']= [10, 3.5]

fig = plt.figure()



for i in range(3):



    ax=fig.add_subplot(1, 3, i+1, projection='3d')

    

    cluster_subset= pca_3d_clusters[pca_3d_clusters[:,3]==i]

    type_colors= np.where(cluster_subset[:,4]=='red', 'r', 'y')

    ax.scatter(cluster_subset[:,0], cluster_subset[:,1], cluster_subset[:,2], alpha=0.4, color= type_colors, marker= 'o')

    

    ax.set_title('Cluster {}'.format(i))

    

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_zticklabels([])

    

plt.tight_layout(pad=2.0)



plt.show()
wines['cluster']= pca_3d_clusters[:, 3]
for i in range(3):

    print('Cluster {}'.format(i))

    subset= wines[wines['cluster']==i]

    display(subset.describe())