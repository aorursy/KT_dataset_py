import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy import optimize
import math
from scipy.stats import logistic
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('../input/train.csv')
#df = df[df['LotArea']<30000] # remove outliers
df.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics).fillna(0)
x = newdf[newdf.columns[1:-1]].values
y = df['SalePrice'].values

plt.figure(figsize=(10,10))
plt.scatter(x[:,2],y, alpha=0.6)
plt.show()
def S2_Cost(c,*args):
    X,y = args
    n = len(X)
    mask = X > c
    if len(y[mask]) == 0:
        y1 = 0
        x1 = 0
    else:
        y1 = y[mask].var()
        x1 = X[mask].var()
    if len(y[~mask]) == 0:
        y2 = 0
        x2 = 0
    else:
        y2 = y[~mask].var()
        x2 = X[~mask].var()
    return 1/(1+x1)*y1 + 1/(1+x2)*y2


class AdvancedNode:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.d = len(self.X.T)
        self.value = np.nan
        self.left = None
        self.right= None
    
    def __str__(self):
        return str(len(self.X))
    
    def train(self,depth=0,max_depth=100):
        cuts = self.X.mean(axis=0)
        s = np.zeros(self.d)
        for i in range (0,self.d):
            x_i = self.X[:,i]
            args = (x_i,self.y)
            best_i = S2_Cost(cuts[i],*args)
            s[i] = best_i 
        self.x_selected = np.argmin(s)
        self.c = cuts[self.x_selected]
        indices = self.X[:,self.x_selected] > self.c
        print(self.X[:,self.x_selected],self.c)
        x_left = self.X[indices]
        y_left = self.y[indices]
        x_right = self.X[~indices]
        y_right = self.y[~indices]
        print("numbers",len(x_left),len(x_right),len(y_left),len(y_right))
        left = AdvancedNode(x_left,y_left)
        right = AdvancedNode(x_right,y_right)
        self.value = self.y.mean()
        print("pro",s[self.x_selected], self.x_selected,self.X[:,self.x_selected].var())
        if s[self.x_selected] == 0 or depth == max_depth or self.X[:,self.x_selected].var()==0:
            print(self.y.mean(), self.x_selected, s[self.x_selected], depth)
        else:
            if len(left.X) > 0:
                self.left = left
                self.left.train(depth+1, max_depth)
            else:
                print("Leaf: ",self.value)
            if len(right.X) > 0:
                self.right = right
                self.right.train(depth+1, max_depth)
            else:
                print("Leaf: ",self.value)
    
    def regress(self,X_test):
        X_test_n = X_test
        if self.left is None and self.right is None:
            return self.value
        
        if X_test_n[self.x_selected] > self.c:
            if self.left is not None:
                return self.left.regress(X_test)
            else:
                return self.value
        else:
            if self.right is not None:
                return self.right.regress(X_test)
            else:
                return self.value

precision = 1/(1024*1024)
max_depth = int(1 - np.log2(precision))
print(max_depth)
tree = AdvancedNode(x, y)

tree.train(max_depth=max_depth)
n = len(x)
y_pred = np.zeros(n)
for i,x_i in enumerate(x):
    y_pred[i] = tree.regress(x_i)

plt.figure(figsize=(10,10))
plt.scatter(x[:,2],y_pred, alpha=0.5, c='r')
plt.scatter(x[:,2],y, alpha=0.5, c='b')

plt.show()

F = 1/ (y.var() / y_pred.var())
LSE = np.sum(np.square(y_pred - y))/len(x)
SSres = LSE
SStot = y.var()*(n-1)
R2 = 1 - (SSres / SStot)
adjR2 = 1 - (1 - R2)*((n-1)/(n-2))
print("F:" + str(F))
print("LSE: " + str(LSE))
print("R2: " + str(R2))
print("Adj R2: " + str(adjR2))

df_test = pd.read_csv('../input/test.csv')
newdf = df_test.select_dtypes(include=numerics).fillna(0)
x = newdf[newdf.columns[1:-1]].values

n = len(x)
y_pred = np.zeros(n)
for i,x_i in enumerate(x):
    y_pred[i] = tree.regress(x_i)

df_result = pd.DataFrame({"Id":df_test['Id'].values, "SalePrice": y_pred})
df_result.head()
df_result.to_csv('output.csv', index=False)