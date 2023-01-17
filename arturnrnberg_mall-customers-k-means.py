import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

import yellowbrick.cluster

from copy import deepcopy

import seaborn as sns 

sns.set()



from pandas_profiling import ProfileReport
mall_customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
mall_report = ProfileReport(mall_customers, title='Mall Report')
mall_report
plt.plot('CustomerID', 'Annual Income (k$)', data=mall_customers)

plt.show()
mall_customers.set_index('CustomerID', inplace=True)

mall_customers['Woman'] = mall_customers['Gender']=='Female'

mall_customers.drop('Gender', axis=1, inplace = True)
mall_customers.describe()
mall_customers
mall_customers_normalized = (mall_customers - mall_customers.mean())/mall_customers.std()
def plot_inercia_per_k (X, k_list, ax = None):



    ax = ax or plt.gca()

    

    kmeans_per_k = [KMeans(n_clusters=k, random_state=0).fit(X) for k in k_list]



    inercias = [m.inertia_ for m in kmeans_per_k]



    ax.plot(k_list, inercias,ls='-',marker='*')



    ax.set_xlabel('$k$')

    ax.set_ylabel('inércia')

    

    return kmeans_per_k
_ = plot_inercia_per_k(mall_customers_normalized, range(2,20))
def plot_silhueta_per_k(X, k_list, ax = None):



    ax = ax or plt.gca()

    

    kmeans_per_k = [KMeans(n_clusters=k, random_state=0).fit(X)

                    for k in k_list]



    silhuetas = [silhouette_score(X, m.labels_) for m in kmeans_per_k]



    ax.plot(k_list, silhuetas,ls='-',marker='*')



    ax.set_xlabel('$k$')

    ax.set_ylabel('silhueta média')

    

    return kmeans_per_k
kmeans_per_k = plot_silhueta_per_k(mall_customers_normalized,range(2,20))
def plot_silhueta(X, k_list):

        

    n_lines = int(np.ceil(len(k_list)/2))    

    

    fig, ax = plt.subplots(n_lines, 2, figsize = (16,4*n_lines))



    for i in range(len(k_list)):



        m = KMeans(k_list[i])

        yellowbrick.cluster.silhouette_visualizer(m, X, show=False, ax=ax.ravel()[i]);



    fig.tight_layout()
plot_silhueta(mall_customers_normalized, [2,3,4,5,6,7])
def plot_cluster_attributes(df, k_list):

    fig, ax = plt.subplots(1,len(k_list), figsize=(20,5))

    

    for i in range(len(k_list)):



        df_tmp = deepcopy(df)

        df_tmp['ID'] = df.index

        kmeans_per_k = [KMeans(n_clusters=k, random_state=0).fit(df) for k in k_list]

        df_tmp['Cluster'] = kmeans_per_k[i].labels_

        df_tmp_melt = pd.melt(df_tmp.reset_index(),

                              id_vars=['ID', 'Cluster'],

                              value_vars=df.columns,

                              var_name='Attribute',

                              value_name='Value')



        sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_tmp_melt, 

                     ax=ax[i], palette = sns.color_palette("husl", k_list[i]))



        ax[i].set_title(f'$k={k_list[i]}$')

        

    return kmeans_per_k
plot_cluster_attributes(mall_customers_normalized, [4, 6])
mall_customers_no_gender = mall_customers_normalized.drop('Woman', axis = 1)
_ = plot_silhueta_per_k(mall_customers_no_gender,range(2,20))
_  = plot_inercia_per_k(mall_customers_no_gender, range(2,20))
plot_silhueta(mall_customers_no_gender, [2,3,4,5,6,7])
k_means_6, k_means_7 = plot_cluster_attributes(mall_customers_no_gender, [6, 7])
mall_customers_no_gender['cluster'] = k_means_6.predict(mall_customers_no_gender)
mall_customers_no_gender['cluster'].value_counts().sort_index()