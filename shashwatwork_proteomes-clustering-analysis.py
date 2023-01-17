import sklearn, re

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer,StandardScaler

from sklearn import preprocessing

from sklearn.decomposition import PCA

import seaborn as se

from fancyimpute import KNN

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score,adjusted_mutual_info_score,adjusted_rand_score,homogeneity_score
df_proteomes = pd.read_csv('../input/breastcancerproteomes/77_cancer_proteomes_CPTAC_itraq.csv',index_col = 0)

clinical = pd.read_csv('../input/breastcancerproteomes/clinical_data_breast_cancer.csv',index_col=0)

PAM50  = pd.read_csv('../input/breastcancerproteomes/PAM50_proteins.csv',header = 0)
df_proteomes
clinical
PAM50
proteomes = df_proteomes.drop(['gene_symbol','gene_name'], axis=1)
proteomes.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)

proteomes = proteomes.transpose()
proteomes
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in proteomes.index],:]
clinical
merged_data = proteomes.merge(clinical,left_index=True,right_index=True)
numerical_data = merged_data.loc[:,[x for x in merged_data.columns if bool(re.search("NP_|XP_",x)) == True]]

pam50_data = numerical_data.iloc[:,numerical_data.columns.isin(PAM50['RefSeqProteinID'])]
pam50_data
pam50_data_ = KNN(k=3).fit_transform(pam50_data)
pam50_data_
scaler = StandardScaler().fit(pam50_data_)

pam50_data_ = scaler.transform(pam50_data_)

np.set_printoptions(precision=3)

print(pam50_data_[0:5,:])
n_clusters = [2,3,4,5,6,7,8,10,20,79]



def compare_k_means(k_list,data):

    ## Run clustering with different k and check the metrics

    for k in k_list:

        clusterer = KMeans(n_clusters=k, n_jobs=4)

        clusterer.fit(data)

        ## The higher (up to 1) the better

        print("Silhouette Coefficient for k == %s: %s" % (

        k, round(silhouette_score(data, clusterer.labels_), 4)))

        ## The higher (up to 1) the better

        print("Homogeneity score for k == %s: %s" % (

        k, round(homogeneity_score(merged_data['PAM50 mRNA'], clusterer.labels_),4)))

        print("Ajusted Rand score for k == %s: %s" % (

        k, round(adjusted_rand_score(merged_data['PAM50 mRNA'], clusterer.labels_),4)))

        print("------------------------")
imputer = Imputer()

processed_numerical_random = numerical_data.iloc[:,np.random.choice(range(numerical_data.shape[1]),43)]

imputer_rnd = imputer.fit(processed_numerical_random)

processed_numerical_random = imputer_rnd.transform(processed_numerical_random)
## Check different numbers of clusters for the PAM50 proteins, there are 4 subtypes of cancer in this data

## 3 samples of healthy patients were dropped at the beginning...

compare_k_means(n_clusters,pam50_data_)

## seems that k==3 works good, the silhouette score is still high and the homogeneity score jumps ~2-fold

## this is what they report in the paper to be the best number of clusters!

## k == 79 has homogeneity score of 1.0, no wonder since the algorithm can assign all the points their separate clusters!

## However, for our application, such clustering would be worthless.
## Use random proteins for comparison

compare_k_means(n_clusters,processed_numerical_random)

## The scores should be significantly lower than for the PAM50 proteins!
## Visualize data using k==3

clusterer_final = KMeans(n_clusters=3, n_jobs=4)

clusterer_final = clusterer_final.fit(pam50_data_)

processed_p50_plot = pd.DataFrame(pam50_data_)

processed_p50_plot['KMeans_cluster'] = clusterer_final.labels_

processed_p50_plot.sort_values('KMeans_cluster',axis=0,inplace=True)

fig = plt.figure(figsize=(12,8))



plt.scatter(pam50_data_[:,0], pam50_data_[:,1], c=clusterer_final.labels_, cmap="Set1_r", s=25)

plt.scatter(clusterer_final.cluster_centers_[:,0] ,clusterer_final.cluster_centers_[:,1], color='black', marker="x", s=250)

plt.title("Kmeans Clustering \n Breast cancer preteomics", fontsize=16)

plt.show()
for n_component in range(1, 16, 2):

    pca = PCA(n_components=n_component)

    reduced_data = pca.fit_transform(pam50_data_)

    print('\nComponents: {}'.format(n_component))

    print('% of Variance Explained: {}'.format(sum(pca.explained_variance_ratio_)))

    compare_k_means(n_clusters, reduced_data)
pca=PCA(n_components=5)

ProteomicsX_pca=pca.fit(pam50_data_)

ProteomicsX_pca2=ProteomicsX_pca.transform(pam50_data_)

print(pca.explained_variance_ratio_)
n_clusters = 5

KMeansModel=KMeans(n_clusters=n_clusters, init='k-means++')

KMeanData=ProteomicsX_pca2

KMeansModel.fit(KMeanData)

labels=KMeansModel.labels_

centroids=KMeansModel.cluster_centers_

print("LABELS",labels)

print("----------------------------")

print("Centroids",centroids)
fig = plt.figure(figsize=(12,8))



plt.scatter(KMeanData[:,0], KMeanData[:,1], c=KMeansModel.labels_, cmap="Set1_r", s=25)

plt.scatter(KMeansModel.cluster_centers_[:,0] ,KMeansModel.cluster_centers_[:,1], color='black', marker="x", s=250)

plt.title("Kmeans Clustering \n Breast cancer preteomics", fontsize=16)

plt.show()