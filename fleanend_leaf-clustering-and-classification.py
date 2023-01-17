# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn import preprocessing
from sklearn import manifold
from sklearn import decomposition
from sklearn import discriminant_analysis
from scipy import optimize
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
le = preprocessing.LabelEncoder()
df['species'] = le.fit_transform(df['species'])
scaler = preprocessing.MinMaxScaler()
df[df.columns[2:]] = scaler.fit_transform(df[df.columns[2:]])

j2 = df[(df['species']==0) | (df['species']==1) | (df['species']==2) | (df['species']==3) | (df['species']==4) | (df['species']==5) | (df['species']==6)]
y = j2[j2.columns[1]].values
palette = ['b','r', 'g', 'k', 'y', 'brown', 'violet']
colors = list(map(lambda y_i: palette[y_i], y))
X = j2[j2.columns[2:]].values
n = len(X)
d = len(X.T)
beta = 2

def gaussian_kernel(x_i,x_j,beta):
    return np.exp(-beta*np.linalg.norm(x_i-x_j))

K_i_j = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if i <= j:
            k = gaussian_kernel(X[i],X[j],beta)
            K_i_j[i,j] = k
            K_i_j[j,i] = k
one_over_n = 1/n * np.ones((n,n))
NK_i_j = K_i_j - 2*one_over_n*K_i_j + one_over_n*K_i_j*one_over_n

val, vec = np.linalg.eig(NK_i_j)
k_pca = vec[:,:2]
X_kpca = np.dot(K_i_j,k_pca)
X_emb = manifold.TSNE().fit_transform(X)
X_pca = decomposition.PCA().fit_transform(X)
X_lda = discriminant_analysis.LinearDiscriminantAnalysis().fit_transform(X,y)
fig = plt.figure(figsize=(20, 20)) 
plt.subplot(2,2,1)
plt.scatter(X_emb[:,0], X_emb[:,1],c=colors)
plt.subplot(2,2,2)
plt.scatter(X_pca[:,0], X_pca[:,1],c=colors)
plt.subplot(2,2,3)
plt.scatter(X_lda[:,0], X_lda[:,1],c=colors)
plt.subplot(2,2,4)
plt.scatter(X_kpca[:,0], X_kpca[:,1],c=colors)
plt.show()

def k_means_iter(k, X, max_iter=12, verbose=False):
    n = len(X)
    d = len(X.T)
    
    # INIT
    def kmeans_pp(X,k):
        n = len(X)
        d = len(X.T)
        indices = [np.random.choice(n,1)[0]]
        C = [X[indices[0]]]
        for i in range(0,k):
            dist = np.zeros(n)
            for j in range(0,n):
                d = np.zeros(len(C))
                for c in range(0,len(C)):
                    d[c] = np.linalg.norm(C[c]-X[j])
                dist[j] = np.min(d)
            for h in indices:
                dist[h] = 0
            dist /= np.sum(dist)
            index = np.random.choice(n, 1, p=dist)
            indices.append(index)
            C.append(X[index])
        return C
        
    C = np.array(kmeans_pp(X,k))

    y_old = np.ones(n)
    y = np.zeros(n)
    if verbose:
        fig = plt.figure(figsize=(10, 10)) 
    for it in range(0,max_iter):
    # ASSIGNMENT
        if (y_old == y).all():
            break
        else:
            y_old = np.copy(y)
        y = np.zeros(n)

        for i in range(0,n):
            dist = np.zeros(k)
            for j in range(0,k):
                dist[j] = np.linalg.norm(C[j]-X[i])
            y[i] = np.argmin(dist)

        # UPDATE
        C[::] = 0
        for i in range(0,n):
            for j in range(0,k):
                if int(y[i]) == j:
                    C[j] += X[i]
        for i in range(0,k):
            n_i = len(y[y==i])
            if n_i == 0:
                ind = np.random.choice(n,1)
                C[i] = X[ind]
                n_i = 1
            C[i] /= n_i
        if verbose:
            if it < 12:
                plt.subplot(3,4,it+1)
                colors = list(map(lambda y_i: palette[int(y_i)], y))
                plt.scatter(X_emb[:,0], X_emb[:,1],c=colors)
            else:
                plt.subplot(3,4,12)
                colors = list(map(lambda y_i: palette[int(y_i)], y))
                plt.scatter(X_emb[:,0], X_emb[:,1],c=colors)
    if verbose:
        plt.show()
    return y, C
y, C = k_means_iter(7,X,verbose=True)
def f_test(X, y, C):
    n = len(X)
    k = len(C)
    d = len(X.T)
    M = X.mean(axis = 0)
    BGV = 0
    for i in range(0,k):
        n_i = len(y[y==i])
        BGV += n_i*np.linalg.norm(C[i]-M)
    BGV/=(k-1)
    WGV=np.zeros(k)
    for i in range(0,n):
        WGV[int(y[i])] += np.linalg.norm(X[i]-C[int(y[i])])
    WGV = np.sum(WGV)
    WGV/=(n-k)
    return BGV/WGV

def k_means(k, X, max_iter=100, verbose=False):
    iterations = max_iter
    f_cand = 1000000
    for i in range(0, 100):
        y,C = k_means_iter(k, X, max_iter=1000000, verbose=False)
        f = f_test(X, y, C)
        if f < f_cand:
            y_cand = y
            C_cand = C
    return y_cand, C_cand
    
# ELBOW METHOD
n = len(X)
t_max = int(n/2)
elb = np.zeros((t_max-2,2))
for i in range(2, t_max):
    elb[i-2][0] = i
    y, C = k_means(i,X, max_iter=1)
    elb[i-2][1] = f_test(X, y, C)
    
plt.plot(elb[:,0],elb[:,1])
plt.show()
y,C=k_means(7,X,max_iter=1000)
colors = list(map(lambda y_i: palette[int(y_i)], y))
plt.scatter(X_emb[:,0], X_emb[:,1],c=colors)
plt.show()
y,C=k_means(7,X_lda,max_iter=1000)
colors = list(map(lambda y_i: palette[int(y_i)], y))
plt.scatter(X_emb[:,0], X_emb[:,1],c=colors)
plt.show()
K_i_j = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if i <= j:
            k = gaussian_kernel(X[i],X[j],beta)
            K_i_j[i,j] = k
            K_i_j[j,i] = k
one_over_n = 1/n * np.ones((n,n))
NK_i_j = K_i_j - 2*one_over_n*K_i_j + one_over_n*K_i_j*one_over_n

val, vec = np.linalg.eig(NK_i_j)
k_pca = vec.T
X_kpca = np.dot(NK_i_j,k_pca)
y,C=k_means(7,X_kpca,max_iter=10000)
colors = list(map(lambda y_i: palette[int(y_i)], y))
plt.scatter(X_emb[:,0], X_emb[:,1],c=colors)
plt.show()
# Exploration with TSNE
X = df[df.columns[2:]].values
y = df[df.columns[1]].values

X_emb = manifold.TSNE().fit_transform(X)

plt.scatter(X_emb[:,0], X_emb[:,1])
plt.show()
# Classification with Neural Network
mask = np.random.rand(len(X)) < 0.8
X_train = X[mask]
X_test = X[~mask]
y_train = y[mask]
y_test = y[~mask]

n = len(X_train)
n_test = len(X_test)
d = len(X_train.T)
k = len(set(y_train))

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu',
                    hidden_layer_sizes=(int(d/2)), random_state=1)

clf.fit(X_train, y_train)
y_pred_nn = clf.predict(X_test)
acc_nn = y_test == y_pred_nn
acc_nn = len(acc_nn[acc_nn==True])/len(acc_nn)
print(acc_nn)
# Classification with Naive Bayes

# P(C_k)
p_c_k = np.zeros(k)
mean_c_k = np.zeros((k,d))
var_c_k = np.zeros((k,d))

for i in range(0,k):
    mask = y_train==i
    p_c_k[i] = len(y_train[mask])/n
    mean_c_k[i] = X_train[mask].mean(axis=0)
    var_c_k[i] = X_train[mask].var(axis=0)
    
# P( x_i | C_k)
p_x_i_c_k = np.zeros((n,k,d))

for i in range(0,n_test):
    for j in range(0,k):
        for t in range(0,d):
            var = var_c_k[j][t]
            mean = mean_c_k[j][t]
            exponent = -np.square(X_test[i][t]-mean)/(2*var)
            p_x_i_c_k[i][j][t] = (1/np.sqrt(2*np.pi*var)) * np.exp(exponent)
            
p_x_i_c_k[np.isnan(p_x_i_c_k)]=1
p_x_i_c_k[p_x_i_c_k==0]=0.00000000000001

aux = np.zeros((n_test,k))
for i in range(0, n_test):
    for j in range(0, k):
        aux[i][j] = p_c_k[j] * np.prod(p_x_i_c_k[i][j])
y_pred_nb = np.argmax(aux,axis=1)
acc_nb = y_test == y_pred_nb
acc_nb = len(acc_nb[acc_nb==True])/len(acc_nb)
print(acc_nb)
# K-Nearest Neighbour with LDA

from scipy import stats

K = 10

lda_map = discriminant_analysis.LinearDiscriminantAnalysis().fit(X_train,y_train)
X_train_lda = lda_map.transform(X_train)
X_test_lda = lda_map.transform(X_test)

KNN = np.zeros((n_test,K))

for j in range(0,n_test):
    dist_label_j = []
    for i in range(0,n):
        dist = np.linalg.norm(X_train_lda[i]-X_test_lda[j])
        label = y_train[i]
        dist_label_j.append((dist,label))
    dist_label_j.sort(key=lambda tup: tup[0])
    knn_label = []
    for dist_label in dist_label_j[:K]:
        knn_label.append(dist_label[1])
    KNN[j] = np.array(knn_label)
y_pred_knn = stats.mode(KNN,axis=1)[0].reshape(n_test)
acc_knn = y_test == y_pred_knn
acc_knn = len(acc_knn[acc_knn==True])/len(acc_knn)
print(acc_knn)
# Ensemble

mode = stats.mode([y_pred_knn, y_pred_nb, y_pred_nn])
y_pred_ens = mode[0].reshape(n_test)
mask = (mode[1]<2).reshape(n_test)

y_pred_ens[mask] = y_pred_knn[mask]

acc_ens = y_test == y_pred_ens
acc_ens = len(acc_ens[acc_ens==True])/len(acc_ens)
print(acc_ens)
