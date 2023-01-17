from sklearn.ensemble import RandomForestClassifier 

import numpy as np 

import matplotlib.pyplot as plt

#X=np.array([[-4],[-3],[-2],[-1],[0],[1],[3],[4]]) 

#y=np.array([1,1,-2,-2,4,4,2,2]) 



X=np.array([[-5],[-4],[-3],[-2],[-1],[0],[1],[2],[3],[4],[5]]) 

y=np.array([1,0,1,0,1,0,1,0,1,0,1] ) 



plt.scatter(X,y)

model=RandomForestClassifier(n_estimators=5) 

model.fit(X,y) 

print(model.predict([[0]])) 