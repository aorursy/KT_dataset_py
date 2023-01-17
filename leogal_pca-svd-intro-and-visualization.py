import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
df = pd.read_csv('../input/IRIS.csv')
df['species'] = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df = df.sample(frac=1.0)
df.head(3)
X = df.iloc[:, :-1].as_matrix().T

X_mean = np.mean(X, axis=1).reshape(-1, 1)
X_std = np.std(X, axis=1).reshape(-1, 1)

X -= X_mean
X /= X_std
df1 = df.iloc[:, 1:]

class0 = df1[df1['species']==0]
class1 = df1[df1['species']==1]
class2 = df1[df1['species']==2]

class0_mat = (df[df['species']==0].iloc[:, :-1].as_matrix().T - X_mean) / X_std
class1_mat = (df[df['species']==1].iloc[:, :-1].as_matrix().T - X_mean) / X_std
class2_mat = (df[df['species']==2].iloc[:, :-1].as_matrix().T - X_mean) / X_std

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class0.iloc[:, 0], class0.iloc[:, 1], class0.iloc[:, 2], c='blue')
ax.scatter(class1.iloc[:, 0], class1.iloc[:, 1], class1.iloc[:, 2], c='red')
ax.scatter(class2.iloc[:, 0], class2.iloc[:, 1], class2.iloc[:, 2], c='green')
U, S, Vh = np.linalg.svd(X)
U = U[:, :-1]
Y_0 = U.T @ class0_mat
Y_1 = U.T @ class1_mat
Y_2 = U.T @ class2_mat
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y_0[0], Y_0[1], Y_0[2], c='blue')
ax.scatter(Y_1[0], Y_1[1], Y_1[2], c='red')
ax.scatter(Y_2[0], Y_2[1], Y_2[2], c='green')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
U, S, Vh = np.linalg.svd(X)
U = U[:, :-2]
Y_0 = U.T @ class0_mat
Y_1 = U.T @ class1_mat
Y_2 = U.T @ class2_mat
plt.scatter(Y_0[0], Y_0[1], c='blue')
plt.scatter(Y_1[0], Y_1[1], c='red')
plt.scatter(Y_2[0], Y_2[1], c='green')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
print('Number of P.C.s:', len(S))
print('First P.C. explains', np.sum(S[0]) / np.sum(S), 'of the total variance.')
print('Second P.C. explains', np.sum(S[1]) / np.sum(S), 'of the total variance.')
print('Third P.C. explains', np.sum(S[2]) / np.sum(S), 'of the total variance.')