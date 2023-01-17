import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, IntSlider
import scipy as sc
# Generate data 
x = np.linspace(0,10,100)
y = np.random.normal(x)

# Prepare matrix form to get X.A = Y
X = np.stack([x,np.ones_like(x)], -1)

# Solve using OLS / pinv
A = np.linalg.inv(X.T@X)@X.T@y

plt.scatter(x,y)
plt.plot(x,X@A, c='r', label=f'A : [{A[0]:.2f},{A[1]:.2f}]')
plt.legend()
x_c = 3
y_c = 4
r_c = 2

theta = np.linspace(0, 2*np.pi, 100)
plt.plot(x_c + r_c*np.cos(theta), y_c + r_c*np.sin(theta), c='r')
plt.axis('equal')

x_m, y_m = np.meshgrid(np.linspace(x_c-1.2*r_c,x_c+1.2*r_c), np.linspace(y_c-1.2*r_c,y_c+1.2*r_c))
loss = lambda x, y : ((x_c - x)**2 + (y_c - y)**2 - r_c**2)**2
plt.pcolormesh(x_m, y_m, loss(x_m,y_m))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Loss for this given circle')
## Generate data 
theta = np.linspace(0, 2*np.pi, 100)
x = np.random.normal(x_c + r_c * np.cos(theta), 0.2)
y = np.random.normal(y_c + r_c * np.sin(theta), 0.2)
plt.scatter(x,y)
plt.axis('equal')
# Matrix form
A = np.stack([-2*x, -2*y, np.ones_like(x)], axis=-1)
b = -(x**2 + y**2)

# Pseudo-Inverse 
v = np.linalg.inv(A.T@A)@A.T@b
x_p = v[0]
y_p = v[1]
r_p = np.sqrt(-v[2] + x_p**2 + y_p**2)

plt.scatter(x,y)
plt.plot(x_p + r_p*np.cos(theta), y_p + r_p * np.sin(theta), c='r')
plt.axis('equal')
w = np.ones_like(x)
w[10] = 200
W = np.diag(w)
@interact
def show_weighted_ols(i=IntSlider(min=1, max=300, step=30, value=1)) :
    w = np.ones_like(y)
    w[10] = i
    W = np.diag(w)
    A = np.linalg.inv(X.T@W@X)@X.T@W.T@y
    plt.plot(x,X@A, c='r', label=f'A : [{A[0]:.2f},{A[1]:.2f}]')
    plt.scatter(x,y, s=w)
    plt.title("Weighted OLS with one weighted point")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
A = np.linalg.inv(X.T@W@X)@X.T@W.T@y
w = np.random.rand(*y.shape)*100
W = np.diag(w)
A = np.linalg.inv(X.T@W@X)@X.T@W.T@y
plt.plot(x,X@A, c='r', label=f'A : [{A[0]:.2f},{A[1]:.2f}]')
plt.scatter(x,y, s=w)
plt.title("Weighted OLS with one weighted point")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
p = 2
z = np.linspace(-1.5,1.5,1000)
plt.plot(z, np.abs(z)**(p-2), label = r'$p=2\quad |x|^{(p-2)}$')
plt.legend()
plt.show()

p = 10
z = np.linspace(-1.5,1.5,1000)
plt.plot(z, np.abs(z)**(p-2), label = r'$p=5\quad |x|^{(p-2)}$')
plt.legend()
plt.show()


p = 0.5
z = np.linspace(-1.5,1.5,1000)
plt.plot(z, np.abs(z)**(p-2), label = r'$p=0.5\quad |x|^{(p-2)}$')
plt.legend()
plt.show()

p = 0.5

w = np.ones_like(y) # Initial weights 


for i in range(5) :
    # Fit Weighted OLS : 
    W = np.diag(w)
    A = np.linalg.inv(X.T@W@X)@X.T@W.T@y
    plt.plot(x,X@A, c='r', label=f'A : [{A[0]:.2f},{A[1]:.2f}]')
    plt.scatter(x,y, s=w)
    plt.title(f'Step : {i}')
    plt.legend()
    plt.show()

    # Compute Error and weight function : 
    R = X@A - y
    w = np.abs(R)**(p-2)
## Problem Setup : 

I = 30 # Number of points to optimize
#n = 1 # Number of dimension in the vector we search

# The lambda matrix tells us for which point we want to keep the initial value 
lam = np.round(np.random.rand(I))

# The w matrix give similarity between points w_i,j : how close v_j have to be from v_i 
w = np.zeros((I,I))
mean_color = np.random.rand(I)
for i in range(I) :
    for iprim in range(I) :
        w[i,iprim] = w[iprim,i] = 1 - (mean_color[i] - mean_color[iprim])**2
        if np.random.rand() > 0.3 and iprim != i: 
            w[i,iprim] = w[iprim,i] = 0
            

# Initial values for v 
v_init = np.random.rand(I)*lam

print(f'w : {w.shape}  v : {v_init.shape} lambda : {lam.shape}')
params = {'w':w, 'v_init': v_init, 'lambda' : lam}
# Easy / Slow Loss function : Find a balance between keeping the initial guess and the smoothness constrain

def loss(v,params = params, verbose=False) :
    w, v_init, lam = params['w'], params['v_init'], params['lambda']
    s =0
    for i in range(I) : 
        for iprim  in range(I) : 
            s +=w[i, iprim]*((v[i]-v[iprim])**2)
    if verbose : 
        print(f'Smoothness Penalty : {s} ')
    return np.sum( lam * (v_init  - v)**2) + s
# Try out random potential solution just for fun : 
losses = []
for t in range(100) :
    v = np.random.rand(I)
    losses.append(loss(v))
plt.plot(losses)
plt.xlabel('Random V')
plt.ylabel('Loss')
print(f' Minimum loss random : {min(losses).round(1)}')
w, v_init, lam = params['w'], params['v_init'], params['lambda']
A = np.diag(lam) + np.diag(w.sum(axis=0)) - w
b = np.diag(lam) @ v_init
v_bar = np.linalg.inv(A)@b
loss(v_bar, verbose=True)
v_bar
res = sc.optimize.minimize(loss, np.random.rand(I,1), options={'disp': True})
loss(res.x, verbose=True)
res.x