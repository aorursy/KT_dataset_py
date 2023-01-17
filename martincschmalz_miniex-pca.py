from sklearn.decomposition import PCA 

import numpy as np 

X=np.array([[3,2],[4,5],[-3,-4],[-4,-5],[0,0],[-4,4],[3,-4]])

import matplotlib.pyplot as plt

x=X[:,0]

y=X[:,1]

plt.scatter(x,y)

model=PCA(n_components=2) 

model.fit(X) 

print(model.explained_variance_ratio_)

print(model.singular_values_)