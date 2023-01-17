from sklearn.cluster import KMeans 

import numpy as np 

X=np.array([[-3,-2],[-4,-5],[3,4],[4,5]]) 

model = KMeans(n_clusters=2, random_state=0).fit(X) 

print(model.predict([[1,2],[1,-1]])) 

print(model.cluster_centers_)