import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_rows', 10)

df = pd.read_csv("../input/ex2data1.txt",header=None)
df.head()
df.plot.scatter(x=0,y=1,c=df[2].map({0:'b', 1:'r'}))
X = df.iloc[:,0:2].values
y = df.iloc[:,2].values
class KNN:
    def __init__(self, X, y, k):
        m,n = X.shape
        self.m = m
        self.n = n
        self.k = k
        self.X = X
        self.y = y

    def calcDistance(self,x):
        distance = np.zeros((self.m))
        for i in range(self.m):
            distance[i] = np.linalg.norm(self.X[i,:]-x)
        neighbor_y = self.y[np.argsort(distance)[:self.k]]
        return np.argmax(np.bincount(neighbor_y))
    
    def predict(self,test_data):
        pred_y = []
        for x in test_data:
            pred_y.append(self.calcDistance(x))
        return pred_y

from sklearn.metrics import accuracy_score

y_pred = KNN(X, y, 5).predict(X)
accuracy_score(y, y_pred)
