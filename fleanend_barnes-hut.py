import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import time
import numpy.linalg

n = 3
c = 10
X = np.zeros((2,3*n))
X[0] = c*(np.append([np.random.randn(n),np.random.randn(n)-(1/c)*np.sqrt(n)],np.random.randn(n)+(1/c)*np.sqrt(n)))
X[1] = c*(np.append([np.random.randn(n)-(1/c)*np.sqrt(n),np.random.randn(n)],np.random.randn(n)))

plt.scatter(X[0],X[1])
plt.show()

beg = time.time()
deltas = np.zeros((2,3*n))
for i in range(0,3*n):
    for j in range(0,3*n):
        if i != j:
            v = (X[:,i]-X[:,j])
            d = np.linalg.norm(v)
            d = c*c*c/(d*d*d)
            v*=d
            deltas[:,i] += v
X1=X+deltas
plt.scatter(X1[0],X1[1])
plt.show()
print(time.time() - beg)
class SpaceNode:
    def __init__(self, limits):
        self.type = 'Empty'
        self.limits = limits
        self.mass = 0
        self.com = self.limits.mean(axis = 0)
    def add(self, point, mass):
        center = self.limits.mean(axis = 0)
        if self.type == 'Empty':
            self.type = 'External'
            self.com = point
            self.mass = mass
        elif self.type == 'External':
            self.type = 'Internal'
            limits_nw = self.limits.copy()
            limits_nw[0][1] = center[1]
            limits_nw[1][0] = center[0]
            limits_ne = self.limits.copy()
            limits_ne[0][1] = center[1]
            limits_ne[0][0] = center[0]
            limits_se = self.limits.copy()
            limits_se[1][1] = center[1]
            limits_se[0][0] = center[0]
            limits_sw = self.limits.copy()
            limits_sw[1][1] = center[1]
            limits_sw[1][0] = center[0]
            self.children = {'nw':SpaceNode(limits_nw), 'ne':SpaceNode(limits_ne), 'se':SpaceNode(limits_se),'sw':SpaceNode(limits_sw)}
            self.add(self.com, self.mass)
        if self.type == 'Internal': 
            if point[1] >= center[1]: # N
                if point[0] < center[0]: # NW
                    self.children['nw'].add(point, mass)
                else: # NE
                    self.children['ne'].add(point, mass)
            else: # S
                if point[0] >= center[0]: # SE
                    self.children['se'].add(point, mass)
                else: # SW
                    self.children['sw'].add(point, mass)
        self.compute_mass()
        self.compute_com()
    def compute_mass(self):
        if self.type != "Internal":
            return self.mass
        else:
            mass = 0
            for k, v in self.children.items():
                mass += v.compute_mass()
            self.mass = mass
            return mass
    def compute_com(self):
        if self.type != "Internal":
            return self.com
        else:
            com = np.zeros(2)
            mass = 0
            for k, v in self.children.items():
                com += v.compute_com()
                mass += v.mass
            self.com = com/mass
            return com
    def __str__(self, n=0):
        string = "Depth: " + str(n) + " mass: " + str(self.mass) + " type: " + str(self.type) +"\n"
        if self.type != "Internal":
            return string
        for k, v in self.children.items():
            string += v.__str__(n+1)
        return string

def barnes_hut(space_node, space_root):
    if space_node.type == 'Internal':
        X = np.zeros((space_node.mass,2))
        i = 0
        for k, v in space_node.children.items():
            f = i+v.mass
            if f > i:
                X[i:f] = barnes_hut(v, space_root)
                i = f
        return X
    elif space_node.type == 'External':
        return space_node.com + delta(space_node, space_root)
    return

def delta(space_node_i, space_node_j):
        v = (space_node_i.com-space_node_j.com)
        d = np.linalg.norm(v)
        if d == 0:
            return np.zeros(2)
        w = (space_node_j.limits[1,0]-space_node_j.limits[0,0])
        s = np.linalg.norm(w)
        if space_node_j.type == 'Internal':
            if s/d <= 0.5:
                d = space_node_j.mass*c*c*c/(d*d*d)
                v*=d
                return v
            else:
                delta_total = np.zeros(2)
                for k, v in space_node_j.children.items():
                    if v.mass != 0:
                        delta_total+=delta(space_node_i, v)
                return delta_total
        elif space_node_j.type == 'External':
            d = c*c*c/(d*d*d)
            v*=d
            return v
def barnes_hut_gravity(X):
    limits = np.array([X.min(axis = 1),X.max(axis = 1)])
    space_root = SpaceNode(limits)
    for i in range(0, len(X.T)):
        space_root.add(X[:,i],1)
    return barnes_hut(space_root, space_root)

X2 = barnes_hut_gravity(X)
plt.subplot(1,2,1)
plt.scatter(X2[:,0],X2[:,1])
plt.subplot(1,2,2)
plt.scatter(X1[0],X1[1])
plt.show()
#TSNE
from scipy.optimize import minimize
def TSNE(X, new_d, perplexity):
    n = len(X)
    d = len(X.T)
    
    N = np.zeros((n,n))
    P = np.zeros(int((n*(n-1))/2))
    
    def g_distance(x_i,x_j):
        d = np.linalg.norm(x_i-x_j)
        return np.exp(-(d*d)/(2*perplexity))
    
    def t_distance(x_i,x_j):
        d = np.linalg.norm(x_i-x_j)
        return 1/(1+(d*d))
    
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                if i < j:
                    N[i][j] = g_distance(X[i],X[j])
                else:
                    N[i][j] = N[j][i]
                    
    c = 0
    D = np.sum(N, axis = 1)
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                Pij = N[i][j]/D[i]
                Pji = N[i][j]/D[j]
                P[c] = (Pij + Pji) / (2 * n)
                c += 1
    Y = np.random.randn(n,new_d)
    def KLD(Y, *args):
        P, n, d = args
        N = np.zeros((n,n))
        Q = np.zeros(int((n*(n-1))/2))
        Y = np.reshape(Y,(n,d))
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    if i < j:
                        N[i][j] = t_distance(Y[i],Y[j])
                    else:
                        N[i][j] = N[j][i]

        c = 0
        D = np.sum(N, axis = 1)
        for i in range(0, n):
            for j in range(0, n):
                if i < j:
                    Qij = N[i][j]/D[i]
                    Qji = N[i][j]/D[j]
                    Q[c] = (Qij + Qji) / (2 * n)
                    c += 1
        KL = 0
        for i in range(0,c):
            KL += P[i] * np.log(P[i]/Q[i])
        return KL
    args = (P, n, new_d)
    Y_0 = np.reshape(Y,-1)
    res = minimize(KLD, Y_0, args=args, method='BFGS')
    if res.success:
        print('SUCCESS')
        return np.reshape(res.x,(n,new_d))
    return False
plt.subplot(1,2,1)
plt.scatter(X[0],X[1])
plt.subplot(1,2,2)
X_TSNE = TSNE(X.T, 2, 5)
plt.scatter(X_TSNE[:,0],X_TSNE[:,1])
plt.show()