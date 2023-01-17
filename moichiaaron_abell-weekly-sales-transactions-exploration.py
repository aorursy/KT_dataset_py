# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data  = pd.read_csv('../input/Sales_Transactions_Dataset_Weekly.csv')
data.head()
data.describe()
data
data.isnull().values.any()
data_norm = data.copy()

data_norm[['Normalized {}'.format(i) for i in range(0,52)]].head()
data_norm = data_norm[['Normalized {}'.format(i) for i in range(0,52)]]
data_norm.head()
data_norm.diff(axis=1).head()
data_norm_diff = data_norm.diff(axis=1).drop('Normalized 0', axis=1).copy()
data_norm_diff.head()
data_norm_diff.head()
data_norm_diff_prod1 =  data_norm_diff.values - data_norm_diff.values[0,:]
data_norm_diff_prod1
data_norm_diff_prod1_sum = (data_norm_diff_prod1**2).sum(axis=1)
data_norm_diff_prod1_sum.shape
import seaborn as sb
sb.jointplot(x = np.arange(0,811,1), 
             y = data_norm_diff_prod1_sum,
            kind='scatter')
#plt.scatter(range(0,811),data_norm_diff_prod1_sum)
prod1_velocities = pd.DataFrame(data_norm_diff_prod1_sum**2, columns=["Vel_total_diff"])
prod1_velocities.sort_values(by="Vel_total_diff")
def getWeeklyDiffs(products_sales_table):
    
    return products_sales_table.diff(axis=1).drop(products_sales_table.columns[0], axis=1).copy()

def getProductErrors(product_index, products_diffs):
    
    return products_diffs - products_diffs.iloc[product_index]
    
def getTotalSquaredError(per_product_error):
    
    return pd.DataFrame((per_product_error**2).sum(axis=1), columns=["Total Error"])
    
def makeProductVelErrorMatrix(products_diffs, nproducts):
    
    product_error_matrix = pd.DataFrame()
    
    for i in range(0,nproducts):
    
        product_errors_table = getProductErrors(i, product_diffs)
        
        product_errors_sumsq = getTotalSquaredError(product_errors_table)
        
        product_error_matrix[i] = product_errors_sumsq
        
    return product_error_matrix
        
        
    
product_diffs  = getWeeklyDiffs(data_norm)
error_matrix = makeProductVelErrorMatrix(product_diffs, 811)


plt.figure(figsize=(15,15))

sb.heatmap(error_matrix, 
           square=True)

def getTotalSquaredError(per_product_error, signed = True):
    
    if signed == False:
        return pd.DataFrame((per_product_error**2).sum(axis=1), columns=["Total Error"])
    else:
        return pd.DataFrame((per_product_error).sum(axis=1)**2, columns=["Total Error"])
error_matrix_signed = makeProductVelErrorMatrix(product_diffs, 811)

plt.figure(figsize=(15,15))

sb.heatmap(error_matrix_signed, 
           square=True)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_data_norm = PCA(n_components=2)
pca_data_norm.fit_transform(data_norm.T)
print(pca_data_norm.explained_variance_ratio_)
print(pca_data_norm.explained_variance_ratio_.sum())
def determineNComponents(data, variance_threshold=0.90):
    
    n_components = 0
    sum_explained_variance = 0
    sum_list = []
    
    while sum_explained_variance < variance_threshold:
        n_components += 1
        pca_data_norm = PCA(n_components=n_components)
        pca_data_norm.fit_transform(data)
        sum_explained_variance = pca_data_norm.explained_variance_ratio_.sum()
        
        sum_list.append(sum_explained_variance)
        
    plt.scatter(np.arange(1,n_components+1,1),
               sum_list)
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Total explained variance ratio")
    
    print("At least {} components needed to explain {}% of the variance.".format(n_components,variance_threshold*100))
                
    return n_components

min_components  = determineNComponents(data_norm.T)
pca_data_norm = PCA(n_components=2)
pca_data_norm.fit_transform(data_norm.T)
sb.jointplot(x=pca_data_norm.components_[0,:], 
             y=pca_data_norm.components_[1,:],
            kind='hex')
sb.jointplot(x=pca_data_norm.components_[0,:], 
             y=pca_data_norm.components_[1,:],
            kind='scatter')
components_data_norm = pca_data_norm.components_
from sklearn.cluster import KMeans  
kmeans = KMeans(n_clusters=min_components)  
kmeans.fit(data_norm)  
print(kmeans.cluster_centers_)  
plt.scatter(x= components_data_norm[0,:],
            y= components_data_norm[1,:],
            c=kmeans.labels_, 
            cmap='rainbow') 

plt.xlabel("1st Princiapl Component")
plt.xlabel("2nd Princiapl Component")
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
components_tsne = tsne.fit_transform(data_norm)
plt.scatter(x= components_tsne[:,0],
            y= components_tsne[:,1],
            cmap='rainbow') 
kmeans_tsne = KMeans(n_clusters=3, random_state=42)  
kmeans_tsne.fit(data_norm)  
plt.scatter(x    = components_tsne[:,0],
            y    = components_tsne[:,1],
            cmap = 'rainbow',
            c    = kmeans_tsne.labels_) 
kmeans_tsne = KMeans(n_clusters=42, random_state=42)  
kmeans_tsne.fit(data_norm)  

plt.scatter(x    = components_tsne[:,0],
            y    = components_tsne[:,1],
            cmap = 'rainbow',
            c    = kmeans_tsne.labels_) 









