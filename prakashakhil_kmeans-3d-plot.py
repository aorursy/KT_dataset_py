import pandas as pd

from sklearn.cluster import KMeans

import re

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA



def print_full(x):

    pd.set_option('display.max_rows', len(x))

    print(x)

    pd.reset_option('display.max_rows')

    

### Set path to the data set

dataset_path = "../input/77_cancer_proteomes_CPTAC_itraq.csv"

clinical_info = "../input/clinical_data_breast_cancer.csv"

pam50_proteins = "../input/PAM50_proteins.csv"



## Load data

data = pd.read_csv(dataset_path,header=0,index_col=0)

clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample

pam50 = pd.read_csv(pam50_proteins,header=0)



## Drop unused information columns

data.drop(['gene_symbol','gene_name'],axis=1,inplace=True)





## Change the protein data sample names to a format matching the clinical data set

data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)



## Transpose data for the clustering algorithm since we want to divide patient samples, not proteins

data = data.transpose()



clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index],:]



merged = data.merge(clinical,left_index=True,right_index=True)





processed_numerical = merged.loc[:,[x for x in merged.columns if bool(re.search("NP_|XP_",x)) == True]]    

                                    

## Select only the PAM50 proteins - known panel of genes used for breast cancer subtype prediction

processed_numerical_p50 = processed_numerical.ix[:,processed_numerical.columns.isin(pam50['RefSeqProteinID'])]



                                                 

imputer = Imputer(missing_values='NaN', strategy='median', axis=1)

imputer = imputer.fit(processed_numerical_p50)

processed_numerical_p50 = imputer.transform(processed_numerical_p50)

clusterer = KMeans(copy_x=True, init='k-means++', max_iter=30000, n_clusters=2, n_init=100,

    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,

    verbose=0)

clusterer.fit(processed_numerical_p50)

clusterer.labels_

centroids= clusterer.cluster_centers_





fignum = 1



fig = plt.figure(fignum, figsize=(4, 3))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



plt.cla()

clusterer.fit(processed_numerical_p50)

labels = clusterer.labels_

clusterer.cluster_centers_

clusterer.inertia_



#ax.scatter(processed_numerical_p50[:, 3], processed_numerical_p50[:, 0], processed_numerical_p50[:, 2], c=labels.astype(np.float))



colors=['#0000FF','#FFFFFF']

col_map=dict(zip(set(labels),colors))

label_color = [col_map[l] for l in labels]

n_component=2

pca = PCA(n_components=n_component)

reduced_data = pca.fit_transform(processed_numerical_p50)

reduced_center = pca.fit_transform(centroids)

print('Variance: {}'.format(sum(pca.explained_variance_ratio_)))

plt.scatter( reduced_data[:,0], reduced_data[:,1], c=label_color)

                                                