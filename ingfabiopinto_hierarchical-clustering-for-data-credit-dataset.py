import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

# las de clustering 

from scipy.spatial import distance_matrix

from scipy.cluster.hierarchy import dendrogram, linkage
datacredit = pd.read_csv("../input/ccdata/CC GENERAL.csv")

datacredit.head()
## let's see the columns with nas

datacredit.isna().sum()
## We fill the na's on the dataset with the mean of each columns 

mp_mean = datacredit['MINIMUM_PAYMENTS'].mean()

datacredit['MINIMUM_PAYMENTS'].fillna(value = mp_mean, inplace = True)

datacredit['CREDIT_LIMIT'].fillna(value = mp_mean, inplace = True)
datacredit.hist(figsize = (15,20));
# after the imputation 

datacredit.head()
datacredit.drop("CUST_ID", axis = 1, inplace = True)

datacredit.head()
datacredit.describe()
## escalamiento 

from sklearn.preprocessing import StandardScaler



escala = StandardScaler()



copiadata = escala.fit_transform(datacredit)

datacopia = pd.DataFrame(copiadata, columns= datacredit.columns)

datacopia.head()
enlaces = linkage(datacopia, method = "ward")

enlaces
dendrogram(enlaces);



# se identifican 4 clusters 
## let see minimized dendrogram 

plt.figure(figsize=(25,10))

dend = dendrogram(enlaces, truncate_mode="lastp", p = 10, show_leaf_counts= True, show_contracted= True) 
## automatic cut of the dendrogram 



from scipy.cluster.hierarchy import inconsistent
## inconsistent method 



depth = 5

incons = inconsistent(enlaces, depth)

incons[-10:]
## método del codo 

last = enlaces[-10:,2]

last_rev = last[::-1]

print(last_rev)



idx = np.arange(1, len(last)+1)

plt.plot(idx, last_rev)



acc = np.diff(last,2)

acc_rev = acc[::-1]

print(acc_rev)

plt.plot(idx[:-2]+1, acc_rev)
## visualization of cluster 



from scipy.cluster.hierarchy import fcluster



## Put the tags in the elements 

clusteres = fcluster(enlaces, 4, criterion="maxclust")

datacredit['Cluster'] = clusteres

datacopia['Cluster'] = clusteres

datacredit.head()


datacopia.boxplot(figsize = (25,25), fontsize = 8, by='Cluster', rot =45, autorange = True );

plt.ylim(-5, 30)
datacopia.boxplot(column='PURCHASES', by='Cluster' )

plt.ylim(-1,30)
datacopia.boxplot(column='ONEOFF_PURCHASES', by='Cluster' )

plt.ylim(-1,30)
datacopia.boxplot(column='CREDIT_LIMIT', by='Cluster' )

plt.ylim(-1,30)
datacopia.boxplot(column='BALANCE', by='Cluster' )

plt.ylim(-1,20)