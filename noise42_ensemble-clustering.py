import pandas as pd

import seaborn as sns
### Set path to the data set

dataset_path = "../input/77_cancer_proteomes_CPTAC_itraq.csv"

clinical_info = "../input/clinical_data_breast_cancer.csv"

pam50_proteins = "../input/PAM50_proteins.csv"
## Load data

data = pd.read_csv(dataset_path,header=0,index_col=0)

clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample

pam50 = pd.read_csv(pam50_proteins,header=0)
print('data\n', data.head())

print('clinical\n', clinical.head())
#see that the names from data are different than those from clinical

print('pam50\n', pam50.head())

print('genes', len(pam50.GeneSymbol.unique()))

print('proteins', len(pam50.RefSeqProteinID.unique()))

## Drop unused information columns

data.drop(['gene_symbol','gene_name'],axis=1,inplace=True)
## Change the protein data sample names to a format matching the clinical data set

import re



data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)
#Check if names are ok

data.head()
## Transpose data for the clustering algorithm since we want to divide patient samples, not proteins

data = data.T
#do samples match?

print(len(data.index))

print(len(clinical.index))
## Add clinical meta data to our protein data set, note: all numerical features for analysis start with NP_ or XP_

merged = data.merge(clinical,left_index=True,right_index=True)

"""

left_index : boolean, default False

Use the index from the left DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels

right_index : boolean, default False

Use the index from the right DataFrame as the join key. Same caveats as left_index

"""
len(merged.index)
## Change name to make it look nicer in the code!

processed = merged



#some columns contains other information, like pre-made clusters, we will use them as reference

processed.columns
## Numerical data for the algorithm, NP_xx/XP_xx are protein identifiers from RefSeq database

##in this case it corresponds to all the columns

numerical_cols = [x for x in processed.columns if bool(re.search("NP_|XP_", x)) == True ]

#label indexing

processed_numerical = processed.loc[:, numerical_cols]
bool_pam_50 = processed_numerical.columns.isin(pam50['RefSeqProteinID'])

#boolean indexing

processed_numerical_p50 = processed_numerical.iloc[:, bool_pam_50]
#are there missing values?

processed_numerical_p50.isnull().sum()
processed_numerical_p50.head(20)
#how many proteins?

len(processed_numerical_p50.columns)
#NaN

## Impute missing values (maybe another method would work better? mean, or drop columns?)

from sklearn.preprocessing import Imputer, StandardScaler





imputer = Imputer(missing_values='NaN', strategy='median', axis=1).fit(processed_numerical_p50)

processed_numerical_p50 = imputer.transform(processed_numerical_p50)

scaler = StandardScaler()

processed_numerical_p50= scaler.fit_transform(processed_numerical_p50)

# Bewarem imputer.transform returns a numpy array, not a dataframe
#if dimensions > 10 (quite always), try different models (for KMeans different clusters).

## Check which number of clusters works best, 20 and 79 are just for fun and comparison.

n_clusters = [2,3,4,5,6,7,8,10,20,79]
from sklearn.cluster import KMeans

from sklearn import metrics



def compare_k_means(k_list, data):

    ## Run clustering with different k and check the metrics

    for k in k_list:

        clusterer = KMeans(n_clusters=k, random_state=0)

        clusterer.fit(data)

        ##

        print("Silhouette Coefficient for k == {}: {}".format(k, metrics.silhouette_score(data, clusterer.labels_)))

        print("Homogeneity score for k == {}: {}".format(k, metrics.homogeneity_score(processed['PAM50 mRNA'], clusterer.labels_)))

        

        print("------------------------------------")
#Non-perfect labelings that further split classes into more clusters can be perfectly homogeneous

compare_k_means(n_clusters,processed_numerical_p50)
processed['PAM50 mRNA']
#What would have happened with random proteins?

## Create a random numerical matrix with imputation:

from numpy import random



rnd_indexing = random.choice(range( processed_numerical.shape[1]) , 43)

processed_numerical_random = processed_numerical.iloc[:, rnd_indexing]

#NaN inside!

imputer_rnd = imputer.fit(processed_numerical_random)

processed_numerical_random = imputer_rnd.transform(processed_numerical_random)
compare_k_means(n_clusters, processed_numerical_random)
#The scores are pretty much lower than p50 (are they significantly lower?)

processed['PAM50 mRNA'].values
## Visualize data using k==3, show the heatmap of protein expression for the used PAM50 proteins (43 available in our data)

clusterer_final = KMeans(n_clusters=3, random_state=0).fit(processed_numerical_p50)

processed_p50_plot = pd.DataFrame(processed_numerical_p50) #to use pandas methods

## add a column with the predictions

processed_p50_plot['KMeans_cluster'] = clusterer_final.labels_



from sklearn.preprocessing import LabelEncoder





le = LabelEncoder()



processed_p50_plot['ref_cluster']= le.fit_transform(processed['PAM50 mRNA'].values)

## sort the samples (axis 0) by cluster

processed_p50_plot.sort_values('KMeans_cluster', axis = 0, inplace=True)



from sklearn.metrics import adjusted_rand_score as rn

rn(le.fit_transform(processed['PAM50 mRNA'].values), clusterer_final.labels_)
processed_p50_plot.tail()
processed_p50_plot.index.name = 'Patient'

sns.heatmap(processed_p50_plot, cmap='YlGnBu')
le.inverse_transform([0,1,2,3])
## Let's do ensemble clustering

##First we will try other cluster methods

eps = [0.1, 0.3, 0.5]

min_samples= [10, 15, 20]



import itertools

from sklearn.cluster import DBSCAN

import numpy as np

db_params=itertools.product(eps, min_samples)



def compare_DBSCAN(param_list, data):

    ## Run clustering with different k and check the metrics

    for eps, min_samples in param_list:

        print(eps, min_samples)

        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

        

        clusterer.fit(data)

        ##

        if len(np.unique(clusterer.labels_)) > 1:

            print("Silhouette Coefficient for eps == {}, min_samples == {}: {}".format(eps, min_samples, metrics.silhouette_score(data, clusterer.labels_)))

        else:

            print("no clustering can be made")

        print("Homogeneity score for eps == {}, min_samples == {}: {}".format(eps, min_samples, metrics.homogeneity_score(processed['PAM50 mRNA'], clusterer.labels_)))

        

        print("------------------------------------")



compare_DBSCAN(db_params, processed_numerical_p50)
from sklearn.cluster import SpectralClustering



def compare_spclust(klist, data):

    ## Run clustering with different k and check the metrics

    for k in klist:

        

        clusterer = SpectralClustering(n_clusters=k)

        

        clusterer.fit(data)

        ##

        if len(np.unique(clusterer.labels_)) > 1:

            print("Silhouette Coefficient for k == {}: {}".format(k, metrics.silhouette_score(data, clusterer.labels_)))

        else:

            print("no clustering can be made")

        print("Homogeneity score for k == {}: {}".format(k , metrics.homogeneity_score(processed['PAM50 mRNA'], clusterer.labels_)))

        

        print("------------------------------------")



compare_spclust(n_clusters, processed_numerical_p50)
## Visualize data using k==3, show the heatmap of protein expression for the used PAM50 proteins (43 available in our data)

clusterer_final2 = SpectralClustering(n_clusters=3, random_state=0).fit(processed_numerical_p50)

processed_p50_plot = pd.DataFrame(processed_numerical_p50) #to use pandas methods

## add a column with the predictions

processed_p50_plot['Spectral_cluster'] = clusterer_final2.labels_



from sklearn.preprocessing import LabelEncoder





le = LabelEncoder()



processed_p50_plot['ref_cluster']= le.fit_transform(processed['PAM50 mRNA'].values)

## sort the samples (axis 0) by cluster

processed_p50_plot.sort_values('Spectral_cluster', axis = 0, inplace=True)



processed_p50_plot.index.name = 'Patient'

sns.heatmap(processed_p50_plot, cmap='YlGnBu')
from sklearn.cluster import MeanShift



## Visualize data using k==3, show the heatmap of protein expression for the used PAM50 proteins (43 available in our data)

clusterer_final3 = MeanShift().fit(processed_numerical_p50)

processed_p50_plot = pd.DataFrame(processed_numerical_p50) #to use pandas methods

## add a column with the predictions

processed_p50_plot['MeanShift_cluster'] = clusterer_final3.labels_



from sklearn.preprocessing import LabelEncoder





le = LabelEncoder()



processed_p50_plot['ref_cluster']= le.fit_transform(processed['PAM50 mRNA'].values)

## sort the samples (axis 0) by cluster

processed_p50_plot.sort_values('MeanShift_cluster', axis = 0, inplace=True)



processed_p50_plot.index.name = 'Patient'

sns.heatmap(processed_p50_plot, cmap='YlGnBu')
### Ensemble clustering with different random_states

def ensemble_kmeans(data, rnd_states, k_list):

    ## Run clustering with different k and check the metrics

    labs=[]

    for r in rnd_states:

        for k in k_list:

            print(k, r)

            clusterer = KMeans(n_clusters=k, random_state=r)

            clusterer.fit(data)

            labs.append(clusterer.labels_)

    return np.array(labs)
rnd_states=[0,1,2,3,4,42,2371]

klist=[3,4,5,6]

cl_data=ensemble_kmeans(processed_numerical_p50, rnd_states, klist)
print(cl_data)
#construct a cooccurrence (consensus) matrix

def cons_matrix(labels):

    C=np.zeros([labels.shape[1],labels.shape[1]], np.int32)

    for label in labels:

        for i, val1 in enumerate(label):

            for j, val2 in enumerate(label):

                #filling C_ij

                

                if val1 == val2 :

                    C[i,j] += 1 

                    

                ##and with a list comprehension?

                

    

    return pd.DataFrame(C)

            
C=cons_matrix(cl_data)

C.columns= processed.index

C.index=processed.index



g=sns.clustermap(C)

plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

plt.show()
from scipy.cluster.hierarchy import dendrogram

den = dendrogram(g.dendrogram_col.linkage,

                                         labels = C.index,

                                         color_threshold=100)

list(den.keys())

den['ivl']
from collections import defaultdict

cluster_idxs = defaultdict(list)

for c, pi in zip(den['color_list'], den['icoord']):

    for leg in pi[1:3]:

        i = (leg - 5.0) / 10.0

        if abs(i - int(i)) < 1e-5:

            cluster_idxs[c].append(den['ivl'][int(i)])

cluster_idxs
##is this division accurate? 

labels_pred=[]



#Messy part: we need to re-order the elements in order to compare them with the original labels

for k in cluster_idxs:

    for el in cluster_idxs[k]:

        labels_pred.append([el,k])
#create a dataframe and set the index in order to merge the indexes with the reference

labels_pred=pd.DataFrame(labels_pred)

labels_pred.index=labels_pred[0]

ref=pd.DataFrame(processed['PAM50 mRNA'])

final=labels_pred.merge(ref,left_index=True,right_index=True)

final.head()
from sklearn.metrics import adjusted_rand_score as rn

from sklearn.metrics import adjusted_mutual_info_score as mi

rand=rn(final['PAM50 mRNA'], final[1])

info=mi(final['PAM50 mRNA'], final[1])

print('rand: {}\nmutual_info: {}'.format(rand,info))
pd.crosstab(final['PAM50 mRNA'], final[1])