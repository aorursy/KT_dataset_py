# from sklearn.datasets import load_wine
# data = load_wine()
# print(data.data)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

watermelon = pd.read_csv('../input/watermelon3.0.csv')
X = watermelon.iloc[:,1:-1]

le = LabelEncoder()
X.iloc[:,:-2] = X.iloc[:,:-2].apply(le.fit_transform)
columns_name = X.columns.values
X
X = X.values
Y = watermelon.iloc[:,-1]
Y = le.fit_transform(Y)
print(Y)
def calcDistance(X):
    m,n = X.shape
    distance = np.zeros((m,m))
    for i in range(m):
        # 自己的距离设为最大值
        distance[i,i] = float('inf')
        for j in range(i+1,m):
            distance[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            distance[j,i] = distance[i,j]
    return distance
distance = calcDistance(X)
# 初始化各属性权值
m,n = X.shape

delta = np.zeros((n))
py = [1,1,1,1,1,1,0,0]
for j in range(n):
    for i in range(m):
        x = X[i]
        x_distance = distance[i]
        # 猜中临近
        y_distance = x_distance[Y == 1]
        nh = X[np.argsort(y_distance)[0]]
        # 猜错临近
        y_distance = x_distance[Y == 0]
        nm = X[np.argsort(y_distance)[0]]

        if py[j] ==1:
            # 离散属性
            if x[j] != nh[j]:
                delta_nh = 1
            else:
                delta_nh = 0
                
            if x[j] != nm[j]:
                delta_nm = 1
            else:
                delta_nm = 0
        else:
            # 连续属性
            delta_nh = abs(x[j] - nh[j])**2
            delta_nm = abs(x[j] - nm[j])**2
            
        if Y[i] == 1:
            delta[j] = delta[j] - delta_nh + delta_nm
        else:
            delta[j] = delta[j] + delta_nh - delta_nm
print(delta)
index = np.argsort(delta)[::-1]
print(columns_name[index])
