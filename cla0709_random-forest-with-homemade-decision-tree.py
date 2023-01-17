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

df = pd.read_csv("../input/Iris.csv")
le = preprocessing.LabelEncoder()
le.fit(df['Species'])
df['Species'] = le.transform(df['Species'])
X = df[df.columns[1:-1]].values
y = df['Species'].values
n = len(X)
d = len(X.T)
mask = np.random.rand(n)
mask = mask<=0.7
X_train = X[mask]
X_test = X[~mask]
y_train = y[mask]
y_test = y[~mask]

# enc = OneHotEncoder()
# y_b = enc.fit_transform(y.reshape(-1, 1)).toarray()

palette = ['b','r','g']
colors = list(map(lambda y_i: palette[y_i], y_test))
plt.scatter(X_test[:,0],X_test[:,1],c=colors)
plt.show()    
l = 3

def H(y,l):
    p = np.zeros(l)
    for i in range(0,l):
        p[i] = np.sum((y==i))/len(y)
        if p[i] != 0:
            p[i] *= np.log2(p[i])
    return -np.sum(p) 

def H_cost(c, *args):
    x,y,l = args
    indices = x > c
    if len(y[indices]) > 0:
        h1 = H(y[indices],l) * (len(y[indices]) / len(y))
    else:
        h1 = 0
    if len(y[~indices]) > 0:
        h2 = H(y[~indices],l) * (len(y[~indices]) / len(y))
    else:
        h2 = 0
    print(h1,h2)
    return h1 + h2
class NaiveNode:
    def __init__(self, X, y, l):
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.d = len(self.X.T)
        self.l = l
        self.classe = -1
    
    def train(self):
        H_father = H(self.y,self.l)
        IG = np.zeros(self.d)
        cuts = np.zeros(self.d)
        for i in range (0,self.d):
            x_i = self.X[:,i]
            x_i_mean = x_i.mean()
            args = (x_i,self.y,self.l)
            c_best_i = optimize.fmin_bfgs(H_cost,x_i_mean,args=args)
            H_best_i = H_cost(c_best_i,*args)
            cuts[i] = c_best_i
            IG[i] = H_father - H_best_i
            
        self.x_selected = np.argmax(IG)
        self.c = cuts[self.x_selected]
        indices = self.X[:,self.x_selected] > self.c
        new_X = np.delete(self.X,self.x_selected,1)
        x_left = new_X[indices]
        y_left = self.y[indices]
        x_right = new_X[~indices]
        y_right = self.y[~indices]
        
        self.left = NaiveNode(x_left,y_left,self.l)
        self.right = NaiveNode(x_right,y_right,self.l)
        
        if len(new_X.T) != 0:
            self.left.train()
            self.right.train()
        else:
            self.classe = stats.mode(self.y)[0][0]
    
    def classify(self,X_test):
        if self.classe != -1:
            return self.classe
        else:
            if X_test[self.x_selected] > self.c:
                return self.left.classify(X_test)
            else:
                return self.right.classify(X_test)
class AdvancedNode:
    def __init__(self, X, y, l, features):
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.d = len(self.X.T)
        self.l = l
        self.features = features
        self.classe = -1
    
    def train(self,depth=0,max_depth=100):
        H_father = H(self.y,self.l)
        IG = np.zeros(self.d)
        cuts = np.zeros(self.d)
        for i in range (0,self.d):
            x_i = self.X[:,i]
            x_i_mean = x_i.mean()
            args = (x_i,self.y,self.l)
            c_best_i = optimize.fmin_bfgs(H_cost,x_i_mean,args=args)
            print(c_best_i)
            H_best_i = H_cost(c_best_i,*args)
            print(H_best_i)
            cuts[i] = c_best_i
            IG[i] = H_father - H_best_i
            
        self.x_selected = np.argmax(IG)
        self.c = cuts[self.x_selected]
        indices = self.X[:,self.x_selected] > self.c
        x_left = self.X[indices]
        y_left = self.y[indices]
        x_right = self.X[~indices]
        y_right = self.y[~indices]
        
        self.left = AdvancedNode(x_left,y_left,self.l,self.features)
        self.right = AdvancedNode(x_right,y_right,self.l,self.features)
        
        print(IG[self.x_selected],H_father,max_depth)
        if IG[self.x_selected] == 0 or depth == max_depth:
            print(stats.mode(self.y)[0][0])
            self.classe = stats.mode(self.y)[0][0]
        else:
            self.left.train(depth+1, max_depth)
            self.right.train(depth+1, max_depth)
    
    def classify(self,X_test):
        X_test_n = X_test.T[self.features].T
        if self.classe != -1:
            return self.classe
        else:
            if X_test_n[self.x_selected] > self.c:
                return self.left.classify(X_test)
            else:
                return self.right.classify(X_test)
y = y_train.copy()

tree = NaiveNode(X_train,y,l)
tree.train()

i = 0;
n_test = len(X_test)
pred_y = np.zeros(n_test)
for x in X_test:
    pred_y[i] = tree.classify(x)
    i+=1

print(pred_y)
palette = ['b','r','g']
colors = list(map(lambda y_i: palette[int(y_i)], pred_y))
plt.scatter(X_test[:,0],X_test[:,1],c=colors)
plt.show()  


print("Accuracy: ", np.sum(pred_y==y_test)/len(y_test))



from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pred = np.zeros(n_test)
i = 0
for x in X_test:
    pred[i] = clf.predict([x])[0]
    i += 1

    
colors = list(map(lambda y_i: palette[int(y_i)], pred))
plt.scatter(X_test[:,0],X_test[:,1],c=colors)
plt.show()  
print("Accuracy: ", np.sum(pred==y_test)/len(y_test))

y = y_train.copy()

advanced_tree = AdvancedNode(X_train,y,l,features=[0,1,2,3])
advanced_tree.train()

i = 0;
n_test = len(X_test)
pred_y = np.zeros(n_test)
for x in X_test:
    pred_y[i] = advanced_tree.classify(x)
    i+=1

print(pred_y)
palette = ['b','r','g']
colors = list(map(lambda y_i: palette[int(y_i)], pred_y))
plt.scatter(X_test[:,0],X_test[:,1],c=colors)
plt.show()  


print("Accuracy: ", np.sum(pred_y==y_test)/len(y_test))
# Hyperparameters
trees = 100
max_depth = 100
m = int(np.sqrt(d))
bootstrap = 0.5

forest = []
for i in range(0,trees):
    mask = np.random.rand(len(X_train))
    mask = mask<= bootstrap

    features = sorted(np.random.choice(d,m, replace=False))
    X_sample = X_train[mask].T[features].T
    y_sample = y_train[mask]
    tree = AdvancedNode(X_sample,y_sample,l,features)
    tree.train(max_depth=max_depth)
    forest.append(tree)
    
    
i = 0;
n_test = len(X_test)
pred_y = np.zeros(n_test)
for x in X_test:
    voting = np.zeros(trees)
    for j,t in enumerate(forest):
        voting[j] = t.classify(x)
    pred_y[i] = stats.mode(voting)[0]
    i+=1

palette = ['b','r','g']
colors = list(map(lambda y_i: palette[int(y_i)], pred_y))
plt.scatter(X_test[:,0],X_test[:,1],c=colors)
plt.show()  

print("Accuracy: ", np.sum(pred_y==y_test)/len(y_test))  
