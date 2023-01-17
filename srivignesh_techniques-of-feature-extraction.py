import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.decomposition import PCA

from sklearn.manifold import MDS, TSNE



pd.set_option('max.rows',500)

pd.set_option('max.columns',80)
preprocessed_train = pd.read_csv('../input/preprocessed-train-data/preprocessed_train_data.csv')

x_train, y_train = preprocessed_train[preprocessed_train.columns[:-1]], preprocessed_train[preprocessed_train.columns[-1]]

preprocessed_train.head()
def PCA_FE(x_train,y_train, n):

    """PCA - Feature Extraction"""

    pca = PCA(n_components= n)

    column_list = []

    for i in range(1,n+1):

        column_list = column_list + ['pc'+str(i)]

    x_train_pca = pd.DataFrame(pca.fit_transform(x_train,y_train),columns = column_list,index=x_train.index)

    return x_train_pca, pca



'''PCA with two dimensions'''

x_train_pca, pca = PCA_FE(x_train, y_train, 2)

display(x_train_pca.head())



''' PCA with three dimensions'''

x_train_pca_3, pca = PCA_FE(x_train, y_train, 3)

display(x_train_pca_3.head())
def plot_pca(x_train_pca):

    '''2D data visualization for PCA'''

    plt.figure(1, figsize=(10,10))

    plt.title('Scatter Plot for PCA',fontsize = 15)

    plt.xlabel('PC1',fontsize =12)

    plt.ylabel('PC2',fontsize =12)

    plt_obj = plt.scatter(x_train_pca['pc1'], x_train_pca['pc2'])

    

plot_pca(x_train_pca)
'''3D data visualization for PCA'''

px.scatter_3d(x_train_pca_3, x = 'pc1', y ='pc2', z= 'pc3')
def MDS_FE(x_train, y_train, n):

    """MDS - Feature Extraction"""

    mds = MDS(n_components = n)

    column_list = []

    for i in range(1,n+1):

        column_list = column_list + ['component'+str(i)]

    x_train_mds = pd.DataFrame(mds.fit_transform(x_train,y_train),columns = column_list,index=x_train.index)

    return x_train_mds, mds



'''MDS with two dimensions'''

x_train_mds, mds = MDS_FE(x_train, y_train, 2)

display(x_train_mds.head())



''' MDS with three dimensions'''

x_train_mds_3, mds = MDS_FE(x_train, y_train, 3)

display(x_train_mds_3.head())
def plot_mds(x_train_mds):

    '''2D data visualization for MDS'''

    plt.figure(1, figsize=(10,10))

    plt.title('Scatter Plot for MDS',fontsize = 15)

    plt.xlabel('Component1',fontsize =12)

    plt.ylabel('Component2',fontsize =12)

    plt_obj = plt.scatter(x_train_mds['component1'], x_train_mds['component2'])



plot_mds(x_train_mds)
'''3D data visualization for MDS'''

px.scatter_3d( x_train_mds_3, x='component1', y ='component2', z='component3')
def TSNE_FE(x_train, y_train, n):

    """T-SNE - Feature Extraction"""

    tsne = TSNE(n_components = n)

    column_list = []

    for i in range(1,n+1):

        column_list = column_list + ['component'+str(i)]

    x_train_tsne = pd.DataFrame(tsne.fit_transform(x_train,y_train),columns = column_list,index=x_train.index)

    return x_train_tsne, tsne



'''TSNE with two dimensions'''

x_train_tsne, tsne = TSNE_FE(x_train, y_train, 2)

display(x_train_tsne.head())



'''TSNE with three dimensions'''

x_train_tsne_3, tsne = TSNE_FE(x_train, y_train, 3)

display(x_train_tsne_3.head())
def plot_tsne(x_train_tsne):

    '''2D data visualization for MDS'''

    plt.figure(1, figsize=(10,10))

    plt.title('Scatter Plot for TSNE',fontsize = 15)

    plt.xlabel('Component1',fontsize =12)

    plt.ylabel('Component2',fontsize =12)

    plt_obj = plt.scatter(x_train_tsne['component1'], x_train_tsne['component2'])



plot_mds(x_train_tsne)
'''3D data visualization for TSNE'''

px.scatter_3d( x_train_tsne_3, x='component1', y ='component2', z='component3')