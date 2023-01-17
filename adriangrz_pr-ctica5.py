import matplotlib.pyplot as plt
import numpy as np
plt.plot([0.1, 1.05, 2.01, 3.003, 4.11, 4.99, 5.89], [-0.01, 0.978, 1.4, 1.74, 2.1, 2.3, 2.5])
plt.grid()
plt.show()
plt.plot([ 0.316, 1.024, 1.417, 1.732, 2.027, 2.233, 2.426],  [-0.01, 0.978, 1.4, 1.74, 2.1, 2.3, 2.5])
plt.grid()
plt.show()
X = np.array([[0.316], [1.024], [1.417], [1.732], [2.027], [2.233], [2.426]])
X = np.asmatrix(X)
unos = np.ones(X.shape)
X1 = np.column_stack((unos, X))
X1
Y = np.array([[-0.01], [0.978], [1.4], [1.74], [2.1], [2.3], [2.5]])
Y = np.asmatrix(Y)
Y
theta = (X1.T * X1).I * X1.T * Y
theta
plt.plot([0.1, 1.05, 2.01, 3.003, 4.11, 4.99, 5.89], [ 0.065,  0.899, 1.361, 1.731, 2.078, 2.320, 2.548])
plt.grid()
plt.show()
xy = np.array([[-0.6861007046, -2.7319977278],
               [0.1845728382, 0.451296404],
               [-2.3158346172, -0.6509207096],
               [-1.0254515503, -0.0809364635],
               [0.4311453719, 0.8026880194],
               [-3.7719061095, 0.814367149],
               [0.8427785188, -1.0716033783],
               [2.2864584569, -2.9176860601],
               [-0.2791914176, -0.8346607406],
               [-0.9005654622, -2.2787579969],
               [0.1421184987, -1.213108765],
               [3.1509795878, -3.7415509261],
               [0.6174679771, -0.5790232979],
               [2.0068530366, -1.4894149248],
               [3.5943855252, -0.8032932784]])
y = np.array([[0],[1],[0],[1],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[0]])

colors = ["#5a9e63","#ff4233"]
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.scatter(xy[:,0], xy[:,1], color=np.vectorize(lambda n: colors[n])(y.ravel()))
plt.grid()
plt.show()
x1 = np.array([[-0.6861007046], 
               [0.1845728382],
               [-2.3158346172],
               [-1.0254515503],
               [0.4311453719],
               [-3.7719061095],
               [0.8427785188],
               [2.2864584569],
               [-0.2791914176],
               [-0.9005654622],
               [0.1421184987],
               [3.1509795878],
               [0.6174679771],
               [2.0068530366],
               [3.5943855252]])
x2 = np.array([[-2.7319977278],
               [0.451296404],
               [-0.6509207096],
               [-0.0809364635],
               [0.8026880194],
               [0.814367149],
               [-1.0716033783],
               [-2.9176860601],
               [-0.8346607406],
               [-2.2787579969],
               [-1.213108765],
               [-3.7415509261],
               [-0.5790232979],
               [-1.4894149248],
               [-0.8032932784]])

unos = np.ones(x1.shape)
x1c = x1**2; x2x = x2**2; x1px2 = x1*x2
xfunc = np.column_stack((unos, x1, x2, x1px2, x1c, x2c))
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

y = y.ravel(); y = np.array([y]).T

theta = np.array([xfunc[1]]).T; itera = 20000; errores = np.zeros(itera)

def descPorGrad(theta, iteraciones):
    alpha = 0.03
    xc = xfunc
    yc = y
    for i in range(0,iteraciones):
        e = sigmoid(np.dot(xc, theta))
        errores[i] = - np.sum(y * np.log(e) + (1 - y) * np.log(1 - e))
        grad = np.dot(xc.T, e - yc)/xc.size
        theta = theta - alpha * grad
    return theta
                
thetaR = descPorGrad(theta, itera)
def zF(x1, x2, thetaO):
    theta0 = thetaO.item(0); theta1 = thetaO.item(1); theta2 = thetaO.item(2); theta3 = thetaO.item(3); theta4 = thetaO.item(4); theta5 = thetaO.item(5)
    return theta0+theta1*x1+theta2*x2+theta3*x1*x2+theta4*x1**2+theta5*x2**2   
def solucion(X, Y, theta):
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
    plt.scatter(X[:,0], X[:,1], color=np.vectorize(lambda n: colors[n])(Y.ravel()))
    x = np.linspace(-5,5,400); y = np.linspace(-5,5,400); x,y = np.meshgrid(x,y)
    z = zF(x,y,thetaR)
    plt.grid()
    plt.contour(x,y,z, levels = [0])
    
solucion(xy, y, thetaR)   
