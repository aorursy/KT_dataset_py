import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler


X = np.random.randn(100).reshape(100,1)
y = 3*X + 2*np.random.randn(100).reshape(100,1)+1

plt.scatter(X,y)
plt.title("Dataset")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.show()
linspace = np.arange(-1,1,0.01)
plt.plot(linspace,np.square(linspace))
plt.xlabel("Error")
plt.ylabel("MSE")
plt.show()
n = len(X)
d = 1

# Add column of ones to X
X_t = np.reshape(np.dstack((np.ones((n,1)),X)),(n,d+1))

# One Shot Solution
theta_star = np.dot(np.linalg.inv(np.dot(X_t.T,X_t)),np.dot(X_t.T,y))

# Plot Data
plt.scatter(X,y)

# Plot Regressor Line
linspace = np.arange(-4,3,0.1).reshape(-1,1)
y_line = np.dot(np.dstack((np.ones((70,1)),linspace)).reshape(70,2),theta_star)
plt.plot(linspace,y_line, c='r')

plt.title("Dataset")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.show()

def f(X, Y):
    return X*X + (Y*Y)*1.5

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121, projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 100)
ys = np.linspace(-1, 1, 100)
X_3d, Y_3d = np.meshgrid(xs, ys)

# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-0.5, 2)


Z = f(X_3d, Y_3d)
wframe = ax.plot_surface(X_3d, Y_3d, Z, rstride=2, cstride=2,zorder=10, cmap=cm.coolwarm)
wframe = ax.contour(X_3d, Y_3d, Z, zdir='z', offset=-0.5, cmap=cm.coolwarm)
plt.title("f(x⃗)",fontsize=20)
plt.xlabel("X1",fontsize=15)
plt.ylabel("X2",fontsize=15)
ax1 = fig.add_subplot(122)
plt.title("f(x⃗)",fontsize=20)
plt.xlabel("X1",fontsize=15)
plt.ylabel("X2",fontsize=15)
wframe = ax1.contour(X_3d, Y_3d, Z, cmap=cm.coolwarm)
plt.show()
alpha = 0.1

# f(x⃗ )
def f(x):
    return x[0]*x[0] + x[1]*x[1]*1.5

# ∇f(x⃗ )
def grad_f(x):
    return np.array([2*x[0],3*x[1]])

iter_max = 12

x = -0.75*np.ones((iter_max,2))

# ∇f(x⃗ ) on many points at once
def fvec(x_array):
    output = []
    for x in x_array:
        output.append(f(x))
    return np.array(output)
fig = plt.figure(figsize=(30,int(iter_max/3) * 10))
for i in range(0,iter_max):
    
    ax1 = fig.add_subplot(int(iter_max/3),3,i+1)
    plt.title("f(x⃗)",fontsize=20)
    plt.xlabel("X1",fontsize=15)
    plt.ylabel("X2",fontsize=15)
    
    wframe = ax1.contour(X_3d, Y_3d, Z, cmap=cm.coolwarm, zorder = -2)
    wframe = ax1.plot(x[:i+1,0],x[:i+1,1],c='k', zorder = -1)
    wframe = ax1.scatter(x[:i+1,0],x[:i+1,1], zorder = 1)
    if i < iter_max-1:
        x[i+1] = x[i] - alpha * grad_f(x[i])

plt.show()
# Scale original data
Xn = StandardScaler().fit_transform(X)
print("Original Data Mean", X.mean(axis = 0), "\nOriginal Data Variance", X.var(axis = 0)
      , "\nNormalised Data Mean", Xn.mean(axis = 0), "\nNormalised Data Variance", Xn.var(axis = 0))
# Add column of ones
Xn = np.reshape(np.dstack((np.ones((n,1)),Xn)),(n,d+1))
yn = StandardScaler().fit_transform(y)
# Plot Normalised data
plt.scatter(Xn[:,1],yn)
plt.show()

# Learning rate
alpha = 0.1

max_iterations = 100

# Vector of errors at each iteration of the desccent
err_iter = np.zeros(max_iterations)

def J(X, y, theta):
    e = y - np.dot(X,theta)
    return (1/n) * np.dot(e.T,e)

def grad_J(X, y, theta):
    return (2/n) * (np.dot(np.dot(X.T,X),theta) - np.dot(X.T,y))

# Initialise theta at random
theta = np.random.randn(d+1).reshape(d+1,1)

for i in range(0, max_iterations):
    # Compute J(theta_k-1)
    err_iter[i] = J(Xn, yn, theta)
    # Update theta_k
    theta -= alpha * grad_J(Xn, yn, theta)
    
fig = plt.figure(figsize=(30,10))
ax1 = fig.add_subplot(121)
plt.plot(np.arange(1,101,1),err_iter)
plt.title("Gradient Descent Optimisation",fontsize=15)
plt.xlabel("Iterations",fontsize=10)
plt.ylabel("Error",fontsize=10)


ax1 = fig.add_subplot(122)

# Plot Data
plt.scatter(Xn[:,1],yn)

# Plot Regressor Line with theta from the gradient descent
linspace = np.arange(-4,3,0.1).reshape(-1,1)
y_line = np.dot(np.dstack((np.ones((70,1)),linspace)).reshape(70,2),theta)
plt.plot(linspace,y_line, c='r')

plt.title("Dataset")
plt.xlabel("Scaled Feature")
plt.ylabel("Scaled Label")
plt.show()

print("Scaled data Coefficients:\n", theta)
import scipy.stats as stat

beta = theta[1][0]
print ("beta:",beta)

# compute residuals y - y_hat

y_hat = np.dot(Xn,theta)
e = yn - y_hat

# compute the standard error

se = np.sqrt((np.dot(e.T,e)/(n-2))/(Xn[:,1].var()*(n-1)))[0][0]

# compute t-score

tscore = beta/(se)

print("t-score(beta):", tscore)

# compute probability through the t-student probability density function

pvalue = 2*stat.t.pdf(tscore,n-2)

print("p-value:", pvalue)

# yn is normalised and as such there's no need to subtract the mean when computing the sum of squares

SStot = np.dot(yn.T,yn)[0][0]

SSres = np.dot(e.T,e)[0][0]

R2 = 1 - (SSres/SStot)

print("R2 coefficient is:",R2)
mse = np.dot(e.T,e)[0][0]

print("MSE:", mse)
fig = plt.figure(figsize=(30,10))
ax1 = fig.add_subplot(121)
plt.plot(np.arange(0,len(X),1),yn, label='Real Value')
plt.plot(np.arange(0,len(X),1),y_hat,label='Prediction')
ax1.legend()
plt.title("Labels VS Prediction point by point")

ax2 = fig.add_subplot(122)

plt.plot(np.arange(0,len(X),1),np.sort(yn,axis=0), label='Real Value')
plt.plot(np.arange(0,len(X),1),np.sort(y_hat,axis=0),label='Prediction')
ax2.legend()
plt.title("Labels VS Prediction point by point sorted")
plt.show()
y_poly = np.exp(-2*Xn[:,1]+1).reshape(n,1) + 10*np.random.randn(100).reshape(100,1)+1
y_poly = StandardScaler().fit_transform(y_poly)
slope, intercept, r_value, p_value, std_err = stat.linregress(Xn[:,1],y_poly.reshape(n,))

lin_poly_hat = slope * Xn[:,1] + intercept

plt.scatter(Xn[:,1],y_poly)
plt.plot(Xn[:,1],lin_poly_hat, 'r')
plt.title("Dataset 2")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.show()

print("p-value: ",p_value, "\nR2: ", r_value*r_value)
alpha = 0.01
max_iterations = 200

def J_poly(X, y, theta):
    e = y - np.dot(X,theta)
    return (1/n) * np.dot(e.T,e)

def grad_J_poly(X, y, theta):
    e = y - np.dot(X,theta)
    grad = -(2/n) * np.array([np.sum(e), np.dot(X[:,1],e), np.dot(X[:,2],e), np.dot(X[:,3],e)])
    return grad.reshape(4,1)

# Initialise theta at random
theta_poly = np.random.randn(4).reshape(4,1)

chi = np.array(list(map(lambda x: [1,x,x*x,x*x*x], list(Xn[:,1])))).reshape(n,4)


err_iter = np.zeros(max_iterations)

for i in range(0, max_iterations):
    # Compute J(theta_k-1)
    err_iter[i] = J_poly(chi, y_poly, theta_poly)
    # Update theta_k
    theta_poly -= alpha * grad_J_poly(chi, y_poly, theta_poly)
    
fig = plt.figure(figsize=(30,10))
ax1 = fig.add_subplot(121)
plt.plot(np.arange(1,max_iterations+1,1),err_iter)
plt.title("Gradient Descent Optimisation",fontsize=15)
plt.xlabel("Iterations",fontsize=10)
plt.ylabel("Error",fontsize=10)
    
ax1 = fig.add_subplot(122)

# Plot Data
plt.scatter(Xn[:,1],y_poly)

# Plot Regressor Line with theta from the gradient descent
linspace = np.arange(-4,3,0.1).reshape(-1,1)
linspace_poly = np.array(list(map(lambda x: [1,x,x*x,x*x*x], list(linspace)))).reshape(-1,4)
y_line = np.dot(linspace_poly,theta_poly).reshape(-1)
plt.plot(linspace,y_line, c='r')

plt.title("Dataset")
plt.xlabel("Scaled Feature")
plt.ylabel("Scaled Label")
plt.show()

alpha = 0.01
max_iterations = 200

# Initialise theta at random
theta_poly = np.random.randn(4).reshape(4,1)

err_iter = np.zeros(max_iterations)

for i in range(0, max_iterations):
    # Compute J(theta_k-1)
    err_iter[i] = J_poly(chi, yn, theta_poly)
    # Update theta_k
    theta_poly -= alpha * grad_J_poly(chi, yn, theta_poly)
    
fig = plt.figure(figsize=(30,10))
ax1 = fig.add_subplot(121)
plt.plot(np.arange(1,max_iterations+1,1),err_iter)
plt.title("Gradient Descent Optimisation",fontsize=15)
plt.xlabel("Iterations",fontsize=10)
plt.ylabel("Error",fontsize=10)
    
ax1 = fig.add_subplot(122)

# Plot Data
plt.scatter(Xn[:,1],yn)

# Plot Regressor Line with theta from the gradient descent
linspace = np.arange(-4,3,0.1).reshape(-1,1)
linspace_poly = np.array(list(map(lambda x: [1,x,x*x,x*x*x], list(linspace)))).reshape(-1,4)
y_line = np.dot(linspace_poly,theta_poly).reshape(-1)
plt.plot(linspace,y_line, c='r')

plt.title("Dataset")
plt.xlabel("Scaled Feature")
plt.ylabel("Scaled Label")
plt.show()

e = yn - np.dot(chi,theta_poly)

print("Linear MSE:", mse,"\nPolynomial MSE:", np.dot(e.T,e)[0,0])
alpha = 0.001
max_iterations = 20000
Lambda = 5

def J_poly(X, y, theta):
    e = y - np.dot(X,theta)
    return (1/n) * np.dot(e.T,e) + Lambda * np.dot(theta[1:].T,theta[1:])

def grad_J_poly(X, y, theta):
    e = y - np.dot(X,theta)
    grad = -(2/n) * np.array([np.sum(e), np.dot(X[:,1],e)-Lambda*theta[1], np.dot(X[:,2],e)-Lambda*theta[2], np.dot(X[:,3],e)-Lambda*theta[3]])
    return grad.reshape(4,1)

# Initialise theta at random
theta_poly = np.random.randn(4).reshape(4,1)

err_iter = np.zeros(max_iterations)

for i in range(0, max_iterations):
    # Compute J(theta_k-1)
    err_iter[i] = J_poly(chi, yn, theta_poly)
    # Update theta_k
    theta_poly -= alpha * grad_J_poly(chi, yn, theta_poly)
    
fig = plt.figure(figsize=(30,10))
ax1 = fig.add_subplot(121)
plt.plot(np.arange(1,max_iterations+1,1),err_iter)
plt.title("Gradient Descent Optimisation",fontsize=15)
plt.xlabel("Iterations",fontsize=10)
plt.ylabel("Error",fontsize=10)
    
ax1 = fig.add_subplot(122)

# Plot Data
plt.scatter(Xn[:,1],yn)

# Plot Regressor Line with theta from the gradient descent
linspace = np.arange(-4,3,0.1).reshape(-1,1)
linspace_poly = np.array(list(map(lambda x: [1,x,x*x,x*x*x], list(linspace)))).reshape(-1,4)
y_line = np.dot(linspace_poly,theta_poly).reshape(-1)
plt.plot(linspace,y_line, c='r')

plt.title("Dataset")
plt.xlabel("Scaled Feature")
plt.ylabel("Scaled Label")
plt.show()

print(theta_poly)
e = yn - np.dot(chi,theta_poly)

print("Linear MSE:", mse,"\nPolynomial MSE:", np.dot(e.T,e)[0,0])
n, p = 10, 0.5  # number of trials, probability of each trial
X = np.random.binomial(n, p, 100000)
ax = sns.countplot(X, color='teal')
ax.set_title('B(10,0.5) Distribution over 100000 samples')
ax.set_ylabel('Occurrences')
ax.set_xlabel('Number of Heads')

plt.show()
ax = sns.countplot(X, palette = sns.color_palette(['red']*2+['teal']*7+['red']*2))
ax.set_title('B(10,0.5) Distribution over 100000 samples')
ax.set_ylabel('Occurrences')
ax.set_xlabel('Number of Heads')

plt.show()

sigma = 2
mu = 1
experiments = 1000

X = sigma*np.random.randn(10000)+mu

exp_gauss_0 = sigma*np.random.randn(10000,1000)+mu

Z = (exp_gauss_0.mean(axis=0)-mu)/(sigma/np.sqrt(experiments))

x = sigma * np.random.randn(1000,10) + mu
t = (x.mean(axis=1)-mu)/np.sqrt((x.var(axis=1)/10))

x_2 = sigma * np.random.randn(1000,100) + mu
t_2 = (x_2.mean(axis=1)-mu)/np.sqrt((x_2.var(axis=1)/100))

fig = plt.figure(figsize=(20,20))
plt.subplot(221)
ax = sns.distplot(X, color='teal', hist = False, kde=True)
ax.set_title('N(1,4) Distribution over 10000 samples')
ax.set_xlabel('X')

plt.subplot(222)
ax1 = sns.distplot(Z, color='teal', hist = False, norm_hist = True )
ax1.set_title('N(0,1) Distribution over 1000 samples')
ax1.set_xlabel('X')

plt.subplot(223)
ax2 = sns.distplot(t, color='teal', hist = False, kde=True, norm_hist = True)
ax2.set_title('t-distribution with 9 degrees of freedom over 1000 samples')
ax2.set_xlabel('X')

plt.subplot(224)
ax3 = sns.distplot(t_2, color='teal', hist = False, kde=True, norm_hist = True)
ax3.set_title('t-distribution with 99 degrees of freedom over 1000 samples')
ax3.set_xlabel('X')

plt.show()
