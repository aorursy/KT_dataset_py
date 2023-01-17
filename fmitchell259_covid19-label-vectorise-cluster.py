import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import glob

import time

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import matplotlib.pyplot as plt



import matplotlib.cm as cm

import os

from sklearn.preprocessing import MinMaxScaler



from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns



from covid19_functions import *
corona_data = pd.read_csv("../input/kagglecovid19/kaggle_covid-19.csv")

corona_data = corona_data.drop(columns=['abstract'])

corona_data = corona_data.fillna("Unknown")

corona_data['risk_label'] = 'Unlabelled'
coronas_d2v_model = Doc2Vec.load("../input/covidvectors/COVID_MEDICAL_DOCS_w2v_MODEL.model")
doc_folder = {"risk": return_doc_index("risk", corona_data),



              "preg": return_doc_index("pregnant", corona_data),



               "smoking": return_doc_index("smoking", corona_data),



               "co_infection": return_doc_index("co infection", corona_data),



                "neonates": return_doc_index("neonates", corona_data),



               "transmission": return_doc_index("transmission dynamics", corona_data),



                "high_risk": return_doc_index("high-risk patient", corona_data)

             }
print(f"Number of Documents that Mention Risk: {len(doc_folder['risk'][0])}")



print(f"Number of Documents that Mention Pregnancy: {len(doc_folder['preg'][0])}")



print(f"Number of Documents that Mention Smoking: {len(doc_folder['smoking'][0])}")



print(f"Number of Documents that Mention Neonates: {len(doc_folder['neonates'][0])}")



print(f"Number of Documents that Mention Transmission Dynamics: {len(doc_folder['transmission'][0])}")



print(f"Number of Documents that Mention High Risk Patients: {len(doc_folder['high_risk'][0])}")
doc_folder['risk'][0]
corona_data = assign_label(doc_folder['risk'][0], corona_data, "risk")

corona_data = assign_label(doc_folder['preg'][0], corona_data, "preg")

corona_data = assign_label(doc_folder['smoking'][0], corona_data, "smoking")

corona_data = assign_label(doc_folder['neonates'][0], corona_data, "neonates")

corona_data = assign_label(doc_folder['transmission'][0], corona_data, "transmission")

corona_data = assign_label(doc_folder['high_risk'][0], corona_data, "high_risk")
le = LabelEncoder()

corona_data['risk_label_encode'] = le.fit_transform(corona_data['risk_label'])
corona_data['title_vector'] = corona_data['title'].apply(create_body_vector, args=[coronas_d2v_model])
vectors = [x for x in corona_data['title_vector']]
vec_df = pd.DataFrame(vectors)
corona_data = corona_data.drop(columns=['title_vector'])
scaler = MinMaxScaler()

vec_df_s = scaler.fit_transform(vec_df)
# First we need to normalise the feature vectors before clustering,



silhouette_plot(vec_df_s, 2, 20)
best_cents = return_opt_weights(vec_df_s)
kmeans_optimised = KMeans(n_clusters=15, init=best_cents, max_iter=20)

kmeans_optimised.fit(vec_df_s)
corona_data['cluster_labels'] = kmeans_optimised.labels_
print(f"Instance 0 CLuster Label: {corona_data['cluster_labels'][0]}")

print(f"Instance 156 CLuster Label: {corona_data['cluster_labels'][9875]}")

print(f"Instance 5689 CLuster Label: {corona_data['cluster_labels'][5689]}")

print(f"Instance 12 CLuster Label: {corona_data['cluster_labels'][12]}")


reshaped_list = [

    

    (0, vec_df_s[0].reshape(-1, 1).T),

    (9875, vec_df_s[9875].reshape(-1, 1).T),

    (5689, vec_df_s[5689].reshape(-1, 1).T),

    (12, vec_df_s[12].reshape(-1, 1).T)

    

]







for r in reshaped_list:

    print('\n------------------------\n')

    ind_arr = list(kmeans_optimised.transform(r[1]))

    print(f"Instance {r[0]} Distance Array:\n\n{ind_arr}\n\n")

    print(f"Instance Actual CLuster Label: {corona_data['cluster_labels'][r[0]]}")

    print(f"Instance Lowest Distance Index: {np.argmin(ind_arr)}")

    print('\n-----------------------\n')



cluster_features = create_cluster_df(kmeans_optimised, vec_df_s)
cluster_features = rename_cluster_cols(cluster_features)
vec_df = rename_vec_df(vec_df)
num_covidDoc_repe = pd.concat([cluster_features, vec_df], axis=1)
doc_id_series = pd.Series(corona_data['doc_id'])

doc_source_series = pd.Series(corona_data['source'])



num_covidDoc_repe["doc_id"] = num_covidDoc_repe.insert(0, "doc_id", doc_id_series)

num_covidDoc_repe["source"] = num_covidDoc_repe.insert(0, "source", doc_id_series)

num_covidDoc_repe["cluster_label"] = num_covidDoc_repe.insert(0, "cluster_label", doc_id_series)
num_covidDoc_repe["doc_id"] = corona_data["doc_id"]

num_covidDoc_repe['source'] = corona_data['source']

num_covidDoc_repe['cluster_label'] = corona_data['cluster_labels']
clust_0_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==0]

clust_1_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==1]

clust_2_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==2]

clust_3_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==3]

clust_4_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==4]

clust_5_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==5]

clust_6_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==6]

clust_7_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==7]



ind_0 = list(clust_0_d.index)

ind_1 = list(clust_1_d.index)

ind_2 = list(clust_2_d.index)

ind_3 = list(clust_3_d.index)

ind_4 = list(clust_4_d.index)

ind_5 = list(clust_5_d.index)

ind_6 = list(clust_6_d.index)

ind_7 = list(clust_7_d.index)

# We can see that cluster 1 documents deal with virus modelling. 



# print_doc_title(corona_data, ind_0)
# CLuster 1 appears to deal with EPidemiology 



# print_doc_title(corona_data, ind_1)
# Think cluister 2 appears to be about co-infection 



# print_doc_title(corona_data, ind_2)
# cLUSTER 3 appears to be about bio-checmical interactions



# print_doc_title(corona_data, ind_3)
# CLuster 4 appears to deal with viral evolution. 



# print_doc_title(corona_data, ind_4)
# Unsure about cluster 5, maybe some domain knowldge would help. 



# print_doc_title(corona_data, ind_5)
# Cluster 6 is mostly about detection 



# print_doc_title(corona_data, ind_6)
# CLuster 7 appears to be mostly about Transmission data. 



# print_doc_title(corona_data, ind_7)
pca = PCA(n_components=3)

three_d_vectors = pca.fit_transform(vec_df_s)
pca_df = pd.DataFrame()

pca_df['pca_one'] = three_d_vectors[:, 0]

pca_df['pca_two'] = three_d_vectors[:, 1]

pca_df['pca_three'] = three_d_vectors[:, 2]
plot_vectors(pca_df, 'cluster', corona_data)
plot_vectors(pca_df, 'risk_label', corona_data)