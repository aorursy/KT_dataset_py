import numpy as np
import matplotlib.pyplot as plt
X1 = np.random.normal((2,2), 0.5, (100, 2))
X2 = np.random.normal((5,5), 0.5, (100, 2))
X3 = np.random.normal((5,2), 0.5, (100, 2))

plt.plot(X1[:,0], X1[:,1], 'ro')
plt.plot(X2[:,0], X2[:,1], 'bo')
plt.plot(X3[:,0], X3[:,1], 'go')
mu1 = np.mean(X1, axis=0).reshape((2, 1))
mu2 = np.mean(X2, axis=0).reshape((2, 1))
mu3 = np.mean(X1, axis=0).reshape((2, 1))

mu  = (mu1 + mu2 + mu3) / 3

N1 = X1.shape[0]
N2 = X1.shape[0]
N3 = X1.shape[0]

Sb1 = N1 * np.dot((mu1 - mu), (mu1 - mu).transpose())
Sb2 = N2 * np.dot((mu2 - mu), (mu2 - mu).transpose()) 
Sb3 = N3 * np.dot((mu3 - mu), (mu3 - mu).transpose())

Sb  = Sb1 + Sb2 + Sb3
S1 = np.cov(X1.transpose())
S2 = np.cov(X2.transpose())
S3 = np.cov(X3.transpose())

Sw = S1 + S2 + S3
Sw_Sb = (np.linalg.pinv(Sw) * Sb)
w, wStars = np.linalg.eig(Sw_Sb)
wStar = wStars[0]
projX1 = np.dot(X1, wStar)
projX2 = np.dot(X2, wStar)
projX3 = np.dot(X3, wStar)


plt.plot(projX1, np.zeros(100), 'ro')
plt.plot(projX2, np.zeros(100), 'bo')
plt.plot(projX3, np.zeros(100), 'go')
x=plt.title("Projected Points on W1")
wStar = wStars[1]
projX1 = np.dot(X1, wStar)
projX2 = np.dot(X2, wStar)
projX3 = np.dot(X3, wStar)


plt.plot(projX1, np.zeros(100), 'ro')
plt.plot(projX2, np.zeros(100), 'bo')
plt.plot(projX3, np.zeros(100), 'go')
x=plt.title("Projected Points on W2")
