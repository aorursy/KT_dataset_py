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
df[df.select_dtypes(include=numerics).columns] = newdf
df.head()
x = df[df.columns[1:-1]].values
y = df['SalePrice'].values

plt.figure(figsize=(10,10))
plt.scatter(x[:,3],y, alpha=0.6)
plt.show()
def S2_Cost(c,*args):
    X,y,categorical = args
    n = len(X)
    y = y / (y.max() - y.min())
    if categorical:
        mask = X == c
    else:
        mask = X > c
        if X.max()!=X.min():
            X = X / (X.max()-X.min())
    if len(y[mask]) == 0:
        y1 = 0
        x1 = 0
    else:
        y1 = y[mask].var()
        if categorical:
            x1 = 0
        else:
            x1 = X[mask].var()
    if len(y[~mask]) == 0:
        y2 = 0
        x2 = 0
    else:
        y2 = y[~mask].var()
        if categorical:
            count = len(X[~mask])
            p_j = {}
            for x_i in X[~mask]:
                if x_i not in p_j:
                    p_j[x_i] = 0
                p_j[x_i] +=1
            x2 = 0
            for j in p_j.keys():
                x2 = np.square(p_j[j]/count)
            x2 = 1-x2
        else:
            x2 = X[~mask].var()
    return x1*y1 + x2*y2


class AdvancedNode:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.d = len(self.X.T)
        self.value = np.nan
        self.left = None
        self.right= None
        self.categorical = False
    def __str__(self):
        return str(len(self.X))
    
    def train(self,depth=0,max_depth=100):
        s = np.zeros(self.d)
        cuts = np.zeros(self.d)
        categorical = np.zeros(self.d)
        c_cuts=[]
        for i in range (0,self.d):
            x_i = self.X[:,i]
            string = False
            for x in x_i:
                if type(x) is str:
                    string = True
                    break
            if string:
                for k in range(0,len(x_i)):
                    x_i[k] = str(x_i[k])
                categorical[i] = 1
                c_cuts.append(stats.mode(x_i)[0][0])
                args = (x_i,self.y,True)
                cuts[i] = len(c_cuts) -1
                best_i = S2_Cost(c_cuts[int(cuts[i])],*args)
                s[i] = best_i
            else:
                args = (x_i,self.y,False)
                cuts[i] = x_i.mean()
                best_i = S2_Cost(cuts[i],*args)
                s[i] = best_i
        s[s==0]=1
        self.x_selected = np.argmin(s)
        if categorical[self.x_selected] == 1:
            self.c = c_cuts[int(cuts[self.x_selected])]
            self.categorical = True
            indices = self.X[:,self.x_selected] == self.c
        else:
            self.c = cuts[self.x_selected]
            indices = self.X[:,self.x_selected] > self.c
        x_left = self.X[indices]
        y_left = self.y[indices]
        x_right = self.X[~indices]
        y_right = self.y[~indices]
        
        left = AdvancedNode(x_left,y_left)
        right = AdvancedNode(x_right,y_right)
        self.value = self.y.mean()
                
        if s[self.x_selected] == 0 or depth == max_depth:
            a = 0
        else:
            if len(left.X) > 0:
                self.left = left
                self.left.train(depth+1, max_depth)
            if len(right.X) > 0:
                self.right = right
                self.right.train(depth+1, max_depth)
    
    def regress(self,X_test):
        X_test_n = X_test
        if self.left is None and self.right is None:
            return self.value
        if (not self.categorical and X_test_n[self.x_selected] > self.c) or (self.categorical and X_test_n[self.x_selected] == self.c):
            if self.left is not None:
                return self.left.regress(X_test)
            else:
                return self.value
        else:
            if self.right is not None:
                return self.right.regress(X_test)
            else:
                return self.value
precision = 1/(512)
max_depth = 1 - np.log2(precision)
trees = 10
BOOSTEDTREES = []
n = len(x)
y_pred = []
res = y.copy()
for i in range(0,trees):
    tree = AdvancedNode(x, res)
    tree.train(max_depth=max_depth)
    BOOSTEDTREES.append(tree)
    y_pred.append(np.zeros(n))
    for j,x_j in enumerate(x):
        y_pred[i][j] = tree.regress(x_j)
    y_pred_boosted = y_pred[0].copy()
    for k in range(1, len(y_pred)):
        y_pred_boosted += y_pred[k]
    res = y - y_pred_boosted
    print(np.sum(np.square(res)))

y_pred_f = np.zeros(n)
for j,x_j in enumerate(x):
    for t in BOOSTEDTREES:
        y_pred_f[j] += t.regress(x_j)


plt.figure(figsize=(10,10))
print(len(y_pred_f), y_pred_f.shape)
plt.scatter(x[:,3],y_pred_f, alpha=0.5, c='r')
plt.scatter(x[:,3],y, alpha=0.5, c='b')

plt.show()

F = 1/ (y.var() / y_pred_f.var())
LSE = np.sum(np.square(y_pred_f - y))/len(x)
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
df_test[df_test.select_dtypes(include=numerics).columns] = newdf

x_test = df_test[df_test.columns[1:]].values
n_test = len(x_test)
y_pred_test = np.zeros(n_test)
for j,x_j in enumerate(x_test):
    for t in BOOSTEDTREES:
        y_pred_test[j] += t.regress(x_j)


    
plt.figure(figsize=(10,10))
plt.scatter(x_test[:,3],y_pred_test, alpha=0.5, c='r')
df_result = pd.DataFrame({"Id":df_test['Id'].values, "SalePrice": y_pred_test})
df_result.head()
df_result.to_csv('output.csv',index=False)