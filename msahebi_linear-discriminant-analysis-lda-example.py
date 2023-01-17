import numpy as np
import matplotlib.pyplot as plt
A = np.array([[4, 2],[2, 4],[2, 3],[3, 6],[4, 4]])
B = np.array([[9,10],[6, 8],[9, 5],[8, 7],[10,8]])
fig = plt.figure()
ax = fig.gca()

plt.plot(A[:,0], A[:,1], 'ro')
plt.plot(B[:,0], B[:,1], 'bo')
plt.grid()
muA = np.mean(A, axis=0)
muB = np.mean(B, axis=0)

sA  = np.cov(A.transpose())
sB  = np.cov(B.transpose())

# Within class
Sw  = sA + sB

# Between class
mu = (muA - muB)
mu = mu.reshape((2, 1))
Sb  = np.dot(mu, mu.transpose())
wStar = np.matmul( np.linalg.pinv(Sw), mu )
projA = np.dot(A, wStar)
projB = np.dot(B, wStar)

fig = plt.figure()
ax = fig.gca()

plt.plot(projA, np.zeros(5), 'ro')
plt.plot(projB, np.zeros(5), 'bo')
plt.grid()
x=plt.title("Projected Points")


