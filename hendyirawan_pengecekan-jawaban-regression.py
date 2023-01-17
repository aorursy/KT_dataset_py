import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
%matplotlib inline 
X = np.array([-2.907, -2.745, -2.637, -2.42, -2.276, -2.101, -1.973, -1.857, -1.643, -1.408, -1.248, -0.963, -0.766, -0.654, -0.494, -0.43, -0.277, -0.108, 0.112, 0.235, 0.498, 0.566, 0.869, 0.958, 1.015, 1.207, 1.427, 1.905, 1.946, 2.133, 2.438, 2.71, 2.896, 2.994])
y = np.array([61.108, 56.501, 55.173, 46.509, 33.646, 36.807, 29.749, 32.362, 15.746, 5.789, -2.241, 7.106, 9.78, 0.253, -8.779, -4.613, -7.886, -5.105, -1.629, 0.215, -2.538, 7.553, -3.64, 6.078, -2.236, 3.084, -18.917, -21.356, -32.504, -30.397, -44.834, -77.244, -86.026, -94.611])


X2 = np.array([[-14.741, -14.99],[-13.909, -13.588],[-13.511, -12.161],[-11.676, -11.112],
[-11.182, -9.973],[-10.355, -8.59],[-10.153, -8.248],[-8.75, -7.99],
[-8.033, -7.722],[-6.502, -6.199],[-4.612, -5.304],[-3.93, -5.129],
[-2.946, -4.6],[-2.768, -4.305],[-1.903, -3.065],[-1.351, -1.861],
[-0.877, 0.691],[1.104, 1.226],[2.476, 1.577],[3.179, 2.78],
[3.326, 3.684],[3.748, 5.785],[5.558, 7.346],
[6.812, 7.455],[7.384, 7.898],[7.863, 8.777],[9.153, 9.309],
[10.053, 9.794],[10.707, 9.97],[11.527, 11.768],[11.999, 12.885],
[12.966, 13.21],[13.908, 14.414],[14.984, 14.98]])
y2 = np.array([126.74, 107.4, 75.963, 55.152, 55.653, 37.074, 43.837, 16.993, 28.95, 16.498, 0.815, 5.401, 2.013, 6.667, 10.605, -7.207, 10.536, 10.965, -1.448, -0.223, 2.722, 3.54, -1.318, -4.409, -16.689, -20.469, -15.055, -31.662, -29.46, -61.319, -65.428, -77.927, -90.615, -131.86])
plt.scatter(X,y)
plt.show()
w_soal = np.array([1.591, -17.824, -8.144])
Xb = np.array([X/X, X,X**2]).T
XtX = (Xb.T).dot(Xb)
Xty = (Xb.T).dot(y.T)
w_hat = np.round(inv(XtX).dot(Xty),3)
print('w_soal = ')
print(w_soal)
print('w_hat = ')
print(w_hat)
y_hat = w_hat.dot(Xb.T)
y_soal = w_soal.dot(Xb.T)

print('jika menggunakan bobot hasil perhitungan')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X,y_hat, c='b', marker="s", label='prediction')
ax1.scatter(X,y, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()

print('jika menggunakan bobot dari soal')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X,y_soal, c='b', marker="s", label='soal')
ax2.scatter(X,y,  c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()
SSE_soal = 43454.587
w_soal = np.array([-2.7, 5.9])
Xb = np.array([X,X/X]).T
y_pred = w_soal.dot(Xb.T)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X,y_pred, c='b', marker="s", label='prediction')
ax1.scatter(X,y, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()
SSE = np.sum((y-y_pred)**2)
print('SSE = ',np.round(SSE,3))
print('SSE soal = ',SSE_soal)
w_soal = np.array([1.087,-0.844,-0.492,-9.198])
Xb = np.array([X/X, X,X**2, X**3]).T
XtX = (Xb.T).dot(Xb)
Xty = (Xb.T).dot(y.T)
w_hat = np.round(inv(XtX).dot(Xty),3)
print('w_soal = ')
print(w_soal)
print('w_hat = ')
print(w_hat)
y_hat = w_hat.dot(Xb.T)
y_soal = w_soal.dot(Xb.T)

print('jika menggunakan bobot hasil perhitungan')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X,y_hat, c='b', marker="s", label='prediction')
ax1.scatter(X,y, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()

print('jika menggunakan bobot dari soal')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X,y_soal, c='b', marker="s", label='soal')
ax2.scatter(X,y,  c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()
w_soal = np.array([-18.756, -2.851])
Xb = np.array([X, X/X]).T
XtX = (Xb.T).dot(Xb)
Xty = (Xb.T).dot(y.T)
w_hat = np.round(inv(XtX).dot(Xty),3)
print('w_soal = ')
print(w_soal)
print('w_hat = ')
print(w_hat)
y_hat = w_hat.dot(Xb.T)
y_soal = w_soal.dot(Xb.T)

print('jika terhadap bobot hasil perhitungan')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X,y_hat, c='b', marker="s", label='prediction')
ax1.scatter(X,y, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()

print('jika menggunakan bobot dari soal')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X,y_soal, c='b', marker="s", label='soal')
ax2.scatter(X,y,  c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()
X_soal = np.array([-2.864,1.489 ,-0.235,1.092 ,2.898])
y_soal = np.array([ 73.128, 20.271,-3.220 , 0.347 ,-75.691 ])
w_soal = np.array([[21.274, -10.608, 0.921]])
SSE_soal = 6666.126

Xb = np.array([X_soal/X_soal, X_soal, X_soal**2]).T
y_pred = w_soal.dot(Xb.T)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X_soal,y_pred, c='b', marker="s", label='prediction')
ax1.scatter(X_soal,y_soal, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()

SSE = np.sum((y_soal-y_pred)**2)
print('SSE = ',np.round(SSE,3))
print('SSE soal = ',SSE_soal)

X2_soal = np.array([[-7.767,-7.111],[-0.333, 0.857]])
w_soal = np.array([4.443, -8.112, 9.028, -8.025])
y_soal = np.array([44.536,9.746])

Xb = np.array([X2_soal[:,0]/X2_soal[:,0], X2_soal[:,0], X2_soal[:,1],X2_soal[:,0]*X2_soal[:,1]]).T
y_pred = w_soal.dot(Xb.T)

print('y prediksi = ',np.round(y_pred,3))
print('y soal = ',y_soal)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X2_soal[:,0],y_pred, c='b', marker="s", label='prediction')
ax1.scatter(X2_soal[:,0],y_soal, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()
w_soal = np.array([4.373,-0.022,-0.015])

Xb = np.array([X2[:,0]/X2[:,0], X2[:,0]**3,X2[:,1]**3]).T
XtX = (Xb.T).dot(Xb)
Xty = (Xb.T).dot(y2.T)
w_hat = np.round(inv(XtX).dot(Xty),3)
print('w_soal = ')
print(w_soal)
print('w_hat = ')
print(w_hat)
y_hat = w_hat.dot(Xb.T)
y_soal = w_soal.dot(Xb.T)

print('jika menggunakan bobot hasil perhitungan')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X2[:,0],y_hat, c='b', marker="s", label='prediction')
ax1.scatter(X2[:,0],y2, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()

print('jika menggunakan bobot dari soal')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X2[:,0],y_soal, c='b', marker="s", label='soal')
ax2.scatter(X2[:,0],y2,  c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()
w_soal = np.array([4.514,-1.678, 1.842,-0.002,-0.005])

Xb = np.array([X2[:,0]/X2[:,0], X2[:,0],X2[:,1], X2[:,0]*X2[:,1], (X2[:,0]+X2[:,1])**3]).T
XtX = (Xb.T).dot(Xb)
Xty = (Xb.T).dot(y2.T)
w_hat = np.round(inv(XtX).dot(Xty),3)
print('w_soal = ')
print(w_soal)
print('w_hat = ')
print(w_hat)
y_hat = w_hat.dot(Xb.T)
y_soal = w_soal.dot(Xb.T)

print('jika menggunakan bobot hasil perhitungan')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X2[:,0],y_hat, c='b', marker="s", label='prediction')
ax1.scatter(X2[:,0],y2, c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()

print('jika menggunakan bobot dari soal')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X2[:,0],y_soal, c='b', marker="s", label='soal')
ax2.scatter(X2[:,0],y2,  c='r', marker="o", label='target')
plt.legend(loc='lower left')
plt.show()