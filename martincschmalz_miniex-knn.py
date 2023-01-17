from sklearn.neighbors import NearestNeighbors 

import numpy as np 

X = np.array([[-1,-2],[-2,-2],[1,1],[-3,-5],[2,2],[4,4]]) 

model = NearestNeighbors(n_neighbors=2).fit(X) 

distances = model.kneighbors([[0,0]]) 

distances
from sklearn.neighbors import KNeighborsRegressor 

import numpy as np 

X = [[-1], [-2], [-3], [1], [2], [4]]

y = [-2 , -2 ,-5 ,1 ,2 ,4]



model = KNeighborsRegressor(n_neighbors=2)

model.fit(X, y) 

print(model.predict([[0]]))