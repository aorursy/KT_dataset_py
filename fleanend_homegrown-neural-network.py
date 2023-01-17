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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../input/Iris.csv")
le = preprocessing.LabelEncoder()
le.fit(df['Species'])
df['Species'] = le.transform(df['Species'])
X = df[df.columns[1:-1]].values
y = df['Species'].values
enc = OneHotEncoder()
y_b = enc.fit_transform(y.reshape(-1, 1)).toarray()
palette = ['b','r','g']
colors = list(map(lambda y_i: palette[y_i], y))
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()
#Define L= #layers, s= #neurons
L = 3
s = [len(X.T),10,3]
epochs = 1000
lam = 0.5
alpha = 0.1

#Init W
W = []
for i in range(1,L):
    W.append(math.sqrt(2/s[i])*np.random.randn(s[i],s[i-1]+1))

vsig = np.vectorize(logistic.cdf)
vsigprime = np.vectorize(lambda a: a * (1-a))

def flatten(matrix_list):
    parameter_list = np.empty([1,])
    for matrix in matrix_list:
        n=matrix.size
        parameter_list=np.concatenate([parameter_list,matrix.reshape(n,)])
    return np.array(parameter_list[1:])
        
def matricise(parameter_list):
    matrix_list = []
    for i in range(1,L):
        l = s[i]*(s[i-1]+1)
        to_add_array = np.array(parameter_list[:l])
        parameter_list=np.array(parameter_list[l:])
        to_add = np.reshape(to_add_array,(s[i],s[i-1]+1))
        matrix_list.append(to_add)
    return matrix_list

def regularise(W_flat):
    to_sum = []
    W = matricise(W_flat)
    for W_l in W:
        mask = np.ones((W_l.shape[1]))
        mask[0] = 0
        to_sum.append(np.sum(np.dot(np.square(W_l),mask)))
    return(sum(to_sum))

def der_regularise(Del_flat,lam):
    to_sum = []
    Del = matricise(Del_flat)
    for Del_l in Del:
        mask = lam*np.ones(Del_l.shape)
        mask[:,0]=0
        Del_l=np.multiply(Del_l,mask)
    flattened_del = flatten(Del)
    return (1/len(flattened_del)) * flattened_del

W_flattened = flatten(W)
args = (X,y_b,lam)

def predict(x,W):
    def iter_predict(x,level):
        a = np.dot(W[level],x)
        a = vsig(a)
        return a
    activations = []
    a = x
    activations.append(a)
    for l in range(0,len(W)):
        a = np.append(np.ones((1,)),a)
        a = iter_predict(a,l)
        activations.append(a)
    return activations

def backpropagate(del_L,W,activations):
    def iter_backpropagate(del_l_1,l):
        first_term = np.dot(W[l].T,del_l_1)
        second_term = vsigprime(activations[l])
        return np.multiply(first_term[1:],second_term)
    deltas = []
    del_l_1 = del_L
    for l in range(len(W)-1,-1,-1):
        deltas.append(np.outer(del_l_1,np.append(np.ones((1,)),activations[l])))
        del_l_1=iter_backpropagate(del_l_1,l)
    deltas.reverse()
    return flatten(deltas)

def J(W_flattened, *args):
    W = matricise(W_flattened)
    X,y,lam=args
    m = len(y)
    err = []
    for i in range(0,m):
        activations=predict(X[i],W)
        h_th = activations[-1]
        for j in range(0,len(h_th)):
            err.append(y[i][j]*math.log(h_th[j])+(1-y[i][j])*(math.log(1-h_th[j])))
    error = -1/m*sum(err) + (lam/(2*m))*regularise(W_flattened)
    return error

def der_J(W_flattened, *args):
    W = matricise(W_flattened)
    X,y,lam=args
    m = len(y)
    Del = np.zeros(W_flattened.shape)
    for i in range(0,m):
        activations=predict(X[i],W)
        del_L=y[i]-activations[-1]
        Del = np.add(Del,backpropagate(del_L,W,activations))
    return der_regularise(-alpha*Del,lam)

err_iter = []
# Stochastic Gradient Descent
for i in range(0,epochs):
    ind = np.random.choice(len(X), 32)
    args = (np.take(X, ind, axis=0),np.take(y_b, ind, axis=0),lam)
    jac = der_J(W_flattened, *args)
    W_flattened = np.add(W_flattened,-jac)
    err_iter.append(math.sqrt(J(W_flattened, *args)))

index = list(range(0,epochs))
#plt.figure(figsize=(10,10))
plt.plot(index,err_iter)
plt.show()

W_trained = matricise(W_flattened)
y_prova=[]
for x in X:
    y_prova.append(predict(x,W_trained)[-1])

y_pred = np.argmax(np.array(y_prova), axis=1)
acc=np.sum(np.apply_along_axis(lambda a: 1 if a[0] == a[1] else 0, 0, np.dstack((y, y_pred)).T))/y_pred.size
print("Accuracy:", acc)
colors = list(map(lambda y_i: palette[y_i], y_pred))
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()

