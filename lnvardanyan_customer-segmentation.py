import numpy as np 

import pandas as pd 

from sklearn.cluster import KMeans
customers = pd.read_excel('../input/customers.xlsx', sheetname=2)
customers.head()
cluster = KMeans(5)
X = customers.iloc[:,1:]
customers["cluster"] = cluster.fit_predict(X)
customers.head()
customers["cluster"].value_counts()
print(customers.iloc[:,[0,-1]])