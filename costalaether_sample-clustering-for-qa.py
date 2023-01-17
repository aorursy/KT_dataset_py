# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # pretty pictures

#from sklearn.cluster import AgglomerativeClustering  

from scipy.cluster.hierarchy import dendrogram, linkage  # linkage analysis and dendrogram for visualization

from scipy.cluster.hierarchy import fcluster  # simple clustering

from scipy.cluster.hierarchy import inconsistent # inconsistancy metric. see link above why this isn't very good





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/SC_expression.csv',index_col = 0 )
Z = linkage(data.transpose(),  method='single', metric='euclidean')  # Performs hierarchical/agglomerative clustering on the condensed distance matrix y.
plt.figure(figsize=(15, 5))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('sample index')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',

    p=22,

    

    show_leaf_counts=False,  # otherwise numbers in brackets are counts

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,  # to get a distribution impression in truncated branches

    

)

plt.show()
clusters = fcluster(Z, 15 , criterion='maxclust')   # clustering wit a maximum of 15 clusters (Corresponding to the 15 groups of accession numbers)
clusters
i = 0

cluster_columns = {}

for element in data.columns:

    cluster_columns[element] = clusters[i]

    #print element , ' in cluster ' , clusters[i]

    i+=1

    



meta_samples = {}

for key, val in cluster_columns.items():

    # the 6 letter column names correspond to sample origin as accession code. 

    # Considering only the first half of accession gives a set of 15 origins for the data.

    # corresponding clusters 

    

    

    metaval = str(key)[:3]  

    

    if not metaval in meta_samples:

        meta_samples[metaval] = []

    

    meta_samples[metaval].append(val)

    

    
for key, value in meta_samples.items():

    print(key , value)
#  This actually looks OK  

# the samples SICXXX seems to be without replicates, SARXXX, QCFXXX and FABXXX have one replicate, and so on. 

# extensive accessions are most likely from the same source.


