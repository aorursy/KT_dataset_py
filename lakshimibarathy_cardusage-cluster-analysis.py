import fun_py

import pandas as pd

cardData=pd.read_csv(r"../input/card-usage/CreditCardUsage.csv")



fun_py.welcome_msg()

    



    
cardData.isnul
cardData.corr()['BALANCE'].sort_values()
X = cardData[:]

import numpy as np

from sklearn.preprocessing import StandardScaler



X.drop('CUST_ID',axis=1,inplace=True)

fun_py.data_groupcols(X)
Clus_dataSet = StandardScaler().fit_transform(X)
X_BAL_PUR = X.iloc[:,0:3].values

X_BAL_PUR
X['BALANCE'].max()
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('BALANCE AND PURCHASES')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X_BAL_PUR, method = 'ward'))

plt.show()
X.head(5)
import pandas_profiling
pr = pandas_profiling.ProfileReport(X)
pr
X['PURCHASES_FREQUENCY'].min()
X['TENURE'].value_counts()
X.head()