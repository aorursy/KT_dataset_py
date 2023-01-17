import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/unsupervised-learning-on-country-data/Country-data.csv")
df.head()
X = np.array((df[['income', 'gdpp']]).astype(float))
X.shape
n = X.shape[1]
for j in range(n):
    X[:, j] -= np.mean(X[:, j])
    print(np.mean(X[:, j]))
k = 3
(m, n) = X.shape
mu = np.random.randint(1, 10, (k, n))
print(mu)
def find_closest_centroids(X, mu):
    m = X.shape[0]
    k = mu.shape[0]
    c = np.zeros([m, 1])
    distance = np.zeros([m, k])
    for i in range(m):
        for j in range(k):
            distance[i, j] = np.sum((X[i, :] - mu[j, :])**2)
        dist = list(distance[i, :])
        c[i, 0] = dist.index(min(dist))
    return c
c = find_closest_centroids(X, mu)
print(c[:5])
def compute_centroids(X, c, mu):
    (k, n) = mu.shape
    m = X.shape[0]
    for i in range(k):
        points = []
        for j in range(m):
            if c[j, 0] == i:
                points.append(j)
        for j in range(n):
            mu[i, j] = np.mean(X[points, j])
    return mu
compute_centroids(X, c, mu)
def cost_function(X, mu, c):
    m = X.shape[0]
    J = 0
    for i in range(m):
        idx = int(c[i, 0])
        J += np.sum((X[i, :] - mu[idx, :])**2)
    return J / m
k = 3
max_iters = 25
np.random.seed(0)
mu = np.random.randint(1, 10, (k, n))
for i in range(max_iters):
    idx = find_closest_centroids(X, mu)
    centroids = compute_centroids(X, idx, mu)
    plt.figure(figsize = (12, 8))
    color = ['r', 'g', 'b']
    mark = ['+', 'o', '*']
    for i in range(k):
        points = []
        for j in range(m):
            if idx[j, 0] == i:
                points.append(j)
        plt.scatter(X[points, 0], X[points, 1], c = color[i], marker = mark[i], s = 100)
    plt.xlabel("Income")
    plt.ylabel("GDP")
plt.figure(figsize = (12, 8))
color = ['r', 'g', 'b']
mark = ['+', 'o', '*']
for i in range(k):
    points = []
    for j in range(m):
        if idx[j, 0] == i:
            points.append(j)
    plt.scatter(X[points, 0], X[points, 1], c = color[i], marker = mark[i], s = 100)
plt.xlabel("Income")
plt.ylabel("GDP")
cost_function(X, centroids, idx)
k = 3
(m, n) = X.shape
indexes = []
costs = []
for i in range(100):
    mu = np.random.randint(1, 10, (k, n))
    idx = find_closest_centroids(X, mu)
    if (0 not in idx) or (1 not in idx) or (2 not in idx):
        pass
        #print("something's missing")
    else:
        centroids = compute_centroids(X, idx, mu)
        J = cost_function(X, centroids, idx)
        #print(J)
        costs.append(J)
        indexes.append(idx)
i_min = costs.index(min(costs))
best_clusters = indexes[i_min]
print(f"minimum cost: {costs[i_min]}")

plt.figure(figsize = (12, 8))
color = ['r', 'g', 'b']
mark = ['+', 'o', '*']
for i in range(k):
    points = []
    for j in range(m):
        if best_clusters[j, 0] == i:
            points.append(j)
    plt.scatter(X[points, 0], X[points, 1], c = color[i], marker = mark[i], s = 100)
plt.xlabel("Income")
plt.ylabel("GDP")
features = np.array(df.drop(['country'], axis = 1))
print(features.shape)
m = features.shape[0]
sigma = np.dot(features.T, features) / m
print(sigma.shape)
u, s, v = np.linalg.svd(sigma)
print(u.shape, s.shape, v.shape)
dim = range(1, 9)
variance = []
for i in dim:
    v = np.sum(s[:i]) / np.sum(s)
    variance.append(v)
print(variance)
d = 2
u_reduce = u[:, 0:d]
print(u_reduce.shape)
z = np.dot(features, u_reduce)
print(z.shape)
max_iters = 5
k = 3
(m, n) = z.shape
np.random.seed(0)
mu = np.random.randn(k, n)
for i in range(max_iters):
    idx = find_closest_centroids(z, mu)
    centroids = compute_centroids(z, idx, mu)
    plt.figure(figsize = (12, 8))
    color = ['r', 'g', 'b']
    mark = ['+', 'o', '*']
    for i in range(k):
        points = []
        for j in range(m):
            if idx[j, 0] == i:
                points.append(j)
        plt.scatter(z[points, 0], z[points, 1], c = color[i], marker = mark[i], s = 100)