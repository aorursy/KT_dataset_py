import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

target = [
    [5,6],
    [-3,-4]
]

source = [
    [3,2],
    [-1,-2],
    [9,4],
    [-4,0],
    [4,-1],
    [0,-3],
    [0,5],
    [-2,2],
    [1,-3],
    [-3,3]
]

X = np.array(source)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.cluster_centers_ = np.array(target)
res = kmeans.predict(X)
print(res + 1)
A = np.array([[2,4], [3,2]])
B = np.array([[-4,-2], [-1,-3], [-5,0]])

X = np.array([
    [0,5], 
    [1,4], 
    [-1, 3], 
    [1,1], 
    [2,1], 
    [1,2],
    [-3,2], 
    [-2,-4],
    [2,-5], 
    [-2,5],
    [6,-2],
    [3,4], 
    [-2,2], 
    [-3,-3], 
    [1,-5],
    [0,4], 
    [0,-3], 
    [-2, 0], 
    [4, 0], 
    [0,3]
])

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.cluster_centers_ = np.array([np.mean(A, axis=0), np.mean(B, axis=0)])
res = kmeans.predict(X)
print(res + 1)
X = np.array([
    [0,5], 
    [1,4], 
    [-1, 3], 
    [1,1], 
    [2,1], 
    [1,2],
    [-3,2], 
    [-2,-4],
    [2,-5], 
    [-2,5],
    [6,-2],
    [3,4], 
    [-2,2], 
    [-3,-3], 
    [1,-5],
    [0,4], 
    [0,-3], 
    [-2, 0], 
    [4, 0], 
    [0,3]
])


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.cluster_centers_ = np.array([[-3,-4], [-3.5,-2.8], [2,4]])
res = kmeans.predict(X)
print(res + 1)
def solveFunc(b, a):
	return (a[0]*b[0] + a[1]*b[1] - 0.5*(b[0]**2 + b[1]**2))


target = [
    [5,6],
    [-3,-4]
]

source = [
    [3,2],
    [-1,-2],
    [9,4],
    [-4,0],
    [4,-1],
    [0,-3],
    [0,5],
    [-2,2],
    [1,-3],
    [-3,3]
]

def solveFunc(b, a):
	return (a[0]*b[0] + a[1]*b[1] - 0.5*(b[0]**2 + b[1]**2))

print(np.array(list((list(map(lambda x: solveFunc(x, t), target))) for t in source)) + 1)
target = [
    [5,6],
    [-3,-4]
]

source = [
    [3,2],
    [-1,-2],
    [9,4],
    [-4,0],
    [4,-1],
    [0,-3],
    [0,5],
    [-2,2],
    [1,-3],
    [-3,3]
]

def solveFunc(b, a):
	return (a[0]*b[0] + a[1]*b[1] - 0.5*(b[0]**2 + b[1]**2))

print(np.array(list(np.argmax(list(map(lambda x: solveFunc(x, t), target))) for t in source)) + 1)
A = np.array([[2,4], [3,2]])
B = np.array([[-4,-2], [-1,-3], [-5,0]])
target = [A, B]
source = [
    [0,5], 
    [1,4], 
    [-1, 3], 
    [1,1], 
    [2,1], 
    [1,2],
    [-3,2], 
    [-2,-4],
    [2,-5], 
    [-2,5],
    [6,-2],
    [3,4], 
    [-2,2], 
    [-3,-3], 
    [1,-5],
    [0,4], 
    [0,-3], 
    [-2, 0], 
    [4, 0], 
    [0,3]]
def solveFunc(b, a):
	return (a[0]*b[0] + a[1]*b[1] - 0.5*(b[0]**2 + b[1]**2))

np.array(list(np.argmax([np.array(list(map(lambda g: solveFunc(g, t),  d))).max() for d in target]) for t in source)) + 1
A = np.array([[2,4], [3,2]])
B = np.array([[-4,-2], [-1,-3], [-5,0]])
target = [np.mean(A, axis=0), np.mean(B, axis=0)]

source = [
    [0,5], 
    [1,4], 
    [-1, 3], 
    [1,1], 
    [2,1], 
    [1,2],
    [-3,2], 
    [-2,-4],
    [2,-5], 
    [-2,5],
    [6,-2],
    [3,4], 
    [-2,2], 
    [-3,-3], 
    [1,-5],
    [0,4], 
    [0,-3], 
    [-2, 0], 
    [4, 0], 
    [0,3]]
def solveFunc(b, a):
	return (a[0]*b[0] + a[1]*b[1] - 0.5*(b[0]**2 + b[1]**2))
print(np.array(list(1 if solveFunc(target[0], t) - solveFunc(target[1], t) > 0 else 2 for t in source)).astype(int))