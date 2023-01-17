import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from scipy import integrate as integrate
from scipy.integrate import odeint
from scipy.integrate import quad

# Test: 1-dressed gluon
def ab(a,b,phia,phib):
    return 1-np.cos(a)*np.cos(b)-np.cos(phia-phib)*np.sin(a)*np.sin(b)

def AB(a,b,phia,phib):
    return 1+np.cos(a)*np.cos(b)-np.cos(phia-phib)*np.sin(a)*np.sin(b)

def U(a,b,j,phia,phib,phij,L):
    return 2**(L/2)*np.cos(j)**L*np.sqrt(ab(a,b,phia,phib)/AB(a,j,phia,phij)/ab(j,b,phij,phib))**L

def integrand(a,b,j,phia,phib,phij,L):
    return ab(a,b,phia,phib)/ab(a,j,phia,phij)/ab(j,b,phij,phib)*(U(a,b,j,phia,phib,phij,L)-1)*np.sin(j)

def int1(a,b,j,phia,phib,L):
    #return quad(integrand, 0, 2*np.pi, args=(a,b,j,phia,phib,L))[0]
    return quad(lambda phij: integrand(a,b,j,phia,phib,phij,L), 0, 2*np.pi)[0]

def int2(a,b,phia,phib,L):
    return quad(lambda j: int1(a,b,j,phia,phib,L), 10**-5, np.pi/2)[0]

#https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
# testmap function over numpy array
x = np.array([1, 2, 3, 4, 5])
f = lambda x: x ** 2
squares = f(x)
print(squares)

A = np.array([[1, 1], [2, 1], [3, -3]])
print(A[2,0])

def fun(a, b):
    f = a + b
    return f

fun_l = [[fun(a, b) for a in range(4)] for b in range(5)]
fun_np = np.array(fun_l)
print(type(fun_np))
print(fun_np)
print(fun_np[:, 1])

#replace quad by sum
def intsum(expr, a, b, N=100):
    value = 0
    value2 = 0
    for n in range(1, N+1):
        value += expr(a + (( n - 1 / 2) * ((b - a) / N)))
    value2 = (b - a) / N *value
    return value2


f1 = lambda x:x**2
print(intsum(f1, 0, 1))
f2 = lambda x, y:x**2 + y
#def f2y(y):
    #return intsum(f2, 100, 0, 1)
#print(f2(1, 1))


def int_sum(b, L):
    a=0.
    #b=np.pi
    phia=0.
    phib=np.pi
    #L=1
    int1 = lambda j: integrand(a,b,j,phia,phib,0,L)/2
    int2 = intsum(int1, 10**-5, np.pi/2, 100) 
    return int2

print(int2(0,np.pi,0,np.pi,3)/4/np.pi)
print(int_sum(np.pi, 3))
    
#ODEint

#print(integrand(0,np.pi,np.pi/2,0,np.pi,np.pi,2))
#print(int1(0,np.pi,np.pi/2,0,np.pi,2)/4/np.pi)
#print(int2(0,np.pi,0,np.pi,2)/4/np.pi)

# function that returns dy/dt
def model(y, t):
    a=0.
    b=np.pi
    phia=0.
    phib=np.pi
    dydt = int2(a,b,phia,phib,t)/4/np.pi #integration from scipy
    return dydt

def model_s(y, t, b=np.pi):
    dydt = int_sum(b, t) #integration by sum
    return dydt

# initial condition
y0 = 1

# time points
t_odeint = np.linspace(0,3)

# solve ODE
%time x_odeint = odeint(model,y0,t_odeint)
%time x_odeint_s = odeint(model_s,y0,t_odeint)

# plot results
euler_constant = 0.57721566490153286060 # Euler Mascheroni Constant

def psy_analytic(x):
    '''
        Profile of the exact solution
    '''
    return 1 - (euler_constant * x + torch.lgamma(1 + x)) / 2.

x0 = torch.unsqueeze(torch.linspace(0, 3, 20), dim=1)  # x data (tensor), shape=(100, 1)
#x = x0.clone().detach(requires_grad=True)
x=x0.clone().detach().requires_grad_(True)
ya = psy_analytic(x)

# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_odeint,x_odeint_s, "rx", label = "ODEint_s")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Euler method: one-step
tmax = 3
dt = 0.15
nt = int(tmax/dt) 
#t  = np.linspace(0., tmax, 20)
t_euler = np.linspace(0., nt * dt, nt)


nb = 20
bmax = np.pi
bmin = np.pi / 2 + 10**-2 # collinear divergent around np.pi / 2
b0 = [1 for num in range(nb)] # discretization along b direction
#bi = np.linspace(bmin, bmax, nb)
bi = np.linspace(bmin, bmax, nb)
print(bi)
X0 = np.array(b0, dtype="float32")
X  = np.zeros([nt, len(X0)]) #nt * nb
print(X0)
print(X)


def model2(b,t):
    a= 0
    #b=np.pi
    phia=0.
    phib=np.pi
    return int2(a,b,phia,phib,t)/4/np.pi
#f = lambda b,t: model2(b,t)
#f_v = np.vectorize(model2)
#print([model2(np.pi / 2 +10**-1, t) for t in t_euler.tolist()])

def model2_s(b,t):
    return int_sum(b, t) #integration by sum

%time fl = [[model2_s(b,t) for b in bi.tolist()] for t in t_euler.tolist()] #nested list comprehension
fl_np = np.array(fl)
#print(fl_np[: -1])
#print(int2(0,np.pi,0,np.pi,2)/4/np.pi)

def Euler(func, X0, t):
    #dt = t[1] - t[0]
    #nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        X[i+1] = X[i] + func[i] * dt
    return X

%time X_euler = Euler(fl_np, X0, t_euler)
x_euler = X_euler[:,-1]
#print(X_euler)



# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_euler, x_euler, "or", label = "Euler")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Modify Euler's method: Adams–Bashforth 2-4 Step
t_ab = np.linspace(0., nt * dt, nt)

def AB2(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + func[0] * dt
    for i in range(nt-2):
        X[i+2] = X[i+1] + dt * (3. * func[i+1] - func[i]) / 2.
    return X

def AB3(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + func[0] * dt
    #X[2] = X[1] + func[1] * dt
    X[2] = X[1] + dt * (3. * func[1]  - func[0] ) / 2.
    for i in range(nt-3):
        X[i+3] = X[i+2] + dt * ( 23 * func[i+2] - 16 * func[i+1] + 5 * func[i] ) / 12
    return X

def AB4(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + func[0] * dt
    X[2] = X[1] + dt * (3. * func[1]  - func[0] ) / 2.
    X[3] = X[2] + dt * ( 23 * func[2] - 16 * func[1] + 5 * func[0] ) / 12
    for i in range(nt-4):
        X[i+4] = X[i+3] + dt * ( 55 * func[i+3] - 59 * func[i+2] + 37 * func[i+1] - 9 * func[i] ) / 24
    return X

%time X_ab2 = AB2(fl_np, X0, t_ab)
x_ab2 = X_ab2[:,-1]
%time X_ab3 = AB3(fl_np, X0, t_ab)
x_ab3 = X_ab3[:,-1]
%time X_ab4 = AB4(fl_np, X0, t_ab)
x_ab4 = X_ab4[:,-1]
#print(X_ab2)


# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_euler, x_euler, "or", label = "Euler")
plt.plot(t_ab, x_ab2, "gs", label = "AB2")
plt.plot(t_ab, x_ab3, "bX", label = "AB3")
plt.plot(t_ab, x_ab4, "yd", label = "AB4")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()

plt.figure(figsize=(6,4),dpi=100)
#plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
#plt.plot(t_odeint,, "mv", label = "ODEint")
plt.plot(t_euler, np.subtract(x_euler, ya.data.flatten().numpy()), "or", label = "Euler")
plt.plot(t_ab, np.subtract(x_ab2, ya.data.flatten().numpy()), "gs", label = "AB2")
plt.plot(t_ab, np.subtract(x_ab3, ya.data.flatten().numpy()), "bX", label = "AB3")
plt.plot(t_ab, np.subtract(x_ab4, ya.data.flatten().numpy()), "yd", label = "AB4")
#plt.plot(t_trap, np.subtract(x_trap, ya.data.flatten().numpy()), "cP", label = "TRAP")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Trapezoidal rule w/ Adams–Bashforth Method
t_trap = np.linspace(0., nt * dt, nt)

def TRAP(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    #X[1] = X[0] + func[0] * dt
    for i in range(nt-1):
        X[i+1] = X[i] + dt * (func[i+1] + func[i]) / 2.
    return X

def TRAP2(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + dt * (func[1] + func[0]) / 2.
    for i in range(nt-2):
        X[i+2] = X[i+1] + dt * (3. * func[i+1] - func[i]) / 2.
    return X

def TRAP3(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + dt * (func[1] + func[0]) / 2.
    #X[2] = X[1] + dt * (3. * func[1] - func[0]) / 2.
    X[2] = X[1] + dt * (func[2] + func[1]) / 2.
    for i in range(nt-3):
        X[i+3] = X[i+2] + dt * ( 23 * func[i+2] - 16 * func[i+1] + 5 * func[i] ) / 12
    return X

%time X_trap = TRAP(fl_np, X0, t_trap)
x_trap = X_trap[:,-1]
%time X_trap2 = TRAP2(fl_np, X0, t_trap)
x_trap2 = X_trap2[:,-1]
#%time X_trap3 = TRAP3(fl_np, X0, t_trap)
#x_trap3 = X_trap3[:,-1]
#print(X_ab4)


# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_euler, x_euler, "or", label = "Euler")
plt.plot(t_ab, x_ab2, "gs", label = "AB2")
plt.plot(t_trap, x_trap, "cP", label = "TRAP")
plt.plot(t_trap, x_trap2, "bX", label = "TRAP2")
#plt.plot(t_trap, x_trap3, "yd", label = "TRAP3")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()

plt.figure(figsize=(6,4),dpi=100)
#plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
#plt.plot(t_odeint,, "mv", label = "ODEint")
plt.plot(t_euler, np.subtract(x_euler, ya.data.flatten().numpy()), "or", label = "Euler")
plt.plot(t_ab, np.subtract(x_ab2, ya.data.flatten().numpy()), "gs", label = "AB2")
plt.plot(t_trap, np.subtract(x_trap, ya.data.flatten().numpy()), "cP", label = "TRAP")
plt.plot(t_trap, np.subtract(x_trap2, ya.data.flatten().numpy()), "bX", label = "TRAP2")
#plt.plot(t_trap, np.subtract(x_trap3, ya.data.flatten().numpy()), "yd", label = "TRAP3")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Backwards Euler (Implicit) Method
#More Methods:
#https://github.com/devkapupara/ODE-Methods
#http://math.oit.edu/~paulr/Upper/Math_45x/Math_452/multistep.pdf
#https://en.wikipedia.org/wiki/Linear_multistep_method
#https://hplgit.github.io/num-methods-for-PDEs/doc/pub/nonlin/pdf/nonlin-4print.pdf
#ODEint

#print(integrand(0,np.pi,np.pi/2,0,np.pi,np.pi,2))
#print(int1(0,np.pi,np.pi/2,0,np.pi,2)/4/np.pi)
#print(int2(0,np.pi,0,np.pi,2)/4/np.pi)

# function that returns dy/dt
def model_r(y, t):
    a=0.
    b=np.pi
    phia=0.
    phib=np.pi
    dydt = y * int2(a,b,phia,phib,t)/4/np.pi #integration from scipy
    return dydt

def model_rs(y, t, b=np.pi):
    dydt = y * int_sum(b, t) #integration by sum
    return dydt

# initial condition
y0 = 1

# time points
t_odeint = np.linspace(0,5)

# solve ODE
%time x_odeint = odeint(model_r,y0,t_odeint)
%time x_odeint_s = odeint(model_rs,y0,t_odeint)

# plot results
euler_constant = 0.57721566490153286060 # Euler Mascheroni Constant

def psy_analytic_r(x):
    '''
        Profile of the exact solution
    '''
    return torch.exp(- (euler_constant * x + torch.lgamma(1 + x)) / 2.)
    #return np.exp(- (euler_constant * x + gammaln(1 + x)) / 2.)


x0 = torch.unsqueeze(torch.linspace(0, 5, 50), dim=1)  # x data (tensor), shape=(100, 1)
#x = x0.clone().detach(requires_grad=True)
x=x0.clone().detach().requires_grad_(True)
yb = psy_analytic_r(x)

# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), yb.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_odeint,x_odeint_s, "rx", label = "ODEint_s")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Euler method: one-step
tmax = 5
dt = 0.1
nt = int(tmax/dt) 
#t  = np.linspace(0., tmax, 20)
t_euler = np.linspace(0., nt * dt, nt)


nb = 50
bmax = np.pi
bmin = np.pi / 2 + 10**-2 # collinear divergent around np.pi / 2
b0 = [1 for num in range(nb)] # discretization along b direction
#bi = np.linspace(bmin, bmax, nb)
bi = np.linspace(bmin, bmax, nb)
print(bi)
X0 = np.array(b0, dtype="float32")
X  = np.zeros([nt, len(X0)]) #nt * nb
print(X0)
print(X)


def model2(b,t):
    a= 0
    #b=np.pi
    phia=0.
    phib=np.pi
    return int2(a,b,phia,phib,t)/4/np.pi
#f = lambda b,t: model2(b,t)
#f_v = np.vectorize(model2)
#print([model2(np.pi / 2 +10**-1, t) for t in t_euler.tolist()])

def model2_s(b,t):
    return int_sum(b, t) #integration by sum

%time fl = [[model2_s(b,t) for b in bi.tolist()] for t in t_euler.tolist()] #nested list comprehension
fl_np = np.array(fl)
#print(fl_np[: -1])
#print(int2(0,np.pi,0,np.pi,2)/4/np.pi)

def Euler(func, X0, t):
    #dt = t[1] - t[0]
    #nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        X[i+1] = X[i] + X[i] * func[i] * dt
    return X

%time X_euler = Euler(fl_np, X0, t_euler)
x_euler = X_euler[:,-1]
#print(X_euler)



# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), yb.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_euler, x_euler, "or", label = "Euler")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Modify Euler's method: Adams–Bashforth 2-4 Step
t_ab = np.linspace(0., nt * dt, nt)

def AB2(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + X[0] * func[0] * dt
    for i in range(nt-2):
        X[i+2] = X[i+1] + X[i+1] * dt * (3. * func[i+1] - func[i]) / 2.
    return X

def AB3(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + X[0] * func[0] * dt
    #X[2] = X[1] + func[1] * dt
    X[2] = X[1] + X[1] * dt * (3. * func[1]  - func[0] ) / 2.
    for i in range(nt-3):
        X[i+3] = X[i+2] + X[i+2] * dt * ( 23 * func[i+2] - 16 * func[i+1] + 5 * func[i] ) / 12
    return X

def AB4(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + X[0] * func[0] * dt
    X[2] = X[1] + X[1] * dt * (3. * func[1]  - func[0] ) / 2.
    X[3] = X[2] + X[2] * dt * ( 23 * func[2] - 16 * func[1] + 5 * func[0] ) / 12
    for i in range(nt-4):
        X[i+4] = X[i+3] + X[i+3] * dt * ( 55 * func[i+3] - 59 * func[i+2] + 37 * func[i+1] - 9 * func[i] ) / 24
    return X

%time X_ab2 = AB2(fl_np, X0, t_ab)
x_ab2 = X_ab2[:,-1]
%time X_ab3 = AB3(fl_np, X0, t_ab)
x_ab3 = X_ab3[:,-1]
%time X_ab4 = AB4(fl_np, X0, t_ab)
x_ab4 = X_ab4[:,-1]
#print(X_ab2)


# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), yb.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_euler, x_euler, "or", label = "Euler")
plt.plot(t_ab, x_ab2, "gs", label = "AB2")
plt.plot(t_ab, x_ab3, "bX", label = "AB3")
plt.plot(t_ab, x_ab4, "yd", label = "AB4")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()

plt.figure(figsize=(6,4),dpi=100)
#plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
#plt.plot(t_odeint,, "mv", label = "ODEint")
plt.plot(t_euler, np.subtract(x_euler, yb.data.flatten().numpy()), "or", label = "Euler")
plt.plot(t_ab, np.subtract(x_ab2, yb.data.flatten().numpy()), "gs", label = "AB2")
plt.plot(t_ab, np.subtract(x_ab3, yb.data.flatten().numpy()), "bX", label = "AB3")
plt.plot(t_ab, np.subtract(x_ab4, yb.data.flatten().numpy()), "yd", label = "AB4")
#plt.plot(t_trap, np.subtract(x_trap, ya.data.flatten().numpy()), "cP", label = "TRAP")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
print(x_euler)
print(x_ab2)
print(yb.data.numpy().flatten())
#Trapezoidal rule w/ Adams–Bashforth Method
t_trap = np.linspace(0., nt * dt, nt)

def TRAP(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    #X[1] = X[0] + func[0] * dt
    for i in range(nt-1):
        X[i+1] = X[i] + X[i] * dt * (func[i+1] + func[i]) / 2.
    return X

def TRAP2(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + X[0] * dt * (func[1] + func[0]) / 2.
    for i in range(nt-2):
        X[i+2] = X[i+1] + X[i+1] * dt * (3. * func[i+1] - func[i]) / 2.
    return X

def TRAP3(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    X[1] = X[0] + X[0] * dt * (func[1] + func[0]) / 2.
    #X[2] = X[1] + dt * (3. * func[1] - func[0]) / 2.
    X[2] = X[1] + X[1] * dt * (func[2] + func[1]) / 2.
    for i in range(nt-3):
        X[i+3] = X[i+2] + X[i+2] * dt * ( 23 * func[i+2] - 16 * func[i+1] + 5 * func[i] ) / 12
    return X

%time X_trap = TRAP(fl_np, X0, t_trap)
x_trap = X_trap[:,-1]
%time X_trap2 = TRAP2(fl_np, X0, t_trap)
x_trap2 = X_trap2[:,-1]
#%time X_trap3 = TRAP3(fl_np, X0, t_trap)
#x_trap3 = X_trap3[:,-1]
#print(X_ab4)


# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), yb.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_ab, x_euler, "or", label = "Euler")
plt.plot(t_ab, x_ab2, "gs", label = "AB2")
plt.plot(t_trap, x_trap, "cP", label = "TRAP")
plt.plot(t_trap, x_trap2, "b*", label = "TRAP2")
#plt.plot(t_trap, x_trap3, "yd", label = "TRAP3")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()

plt.figure(figsize=(6,4),dpi=100)
#plt.plot(x.data.numpy(), ya.data.numpy(), color = "orange", label = "Exact solution")
#plt.plot(t_odeint,, "mv", label = "ODEint")
plt.plot(t_euler, np.subtract(x_euler, yb.data.flatten().numpy()), "or", label = "Euler")
plt.plot(t_ab, np.subtract(x_ab2, yb.data.flatten().numpy()), "gs", label = "AB2")
plt.plot(t_trap, np.subtract(x_trap, yb.data.flatten().numpy()), "cP", label = "TRAP")
plt.plot(t_trap, np.subtract(x_trap2, yb.data.flatten().numpy()), "b*", label = "TRAP2")
#plt.plot(t_trap, np.subtract(x_trap3, ya.data.flatten().numpy()), "yd", label = "TRAP3")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
#Runge–Kutta method
t_rk2 = np.linspace(0., nt * dt, nt)

%time fl_rk2 = [[model2_s(b, t + dt / 2) for b in bi.tolist()] for t in t_euler.tolist()] #nested list comprehension
%time fl_rk4 = [[model2_s(b, t + dt) for b in bi.tolist()] for t in t_euler.tolist()] #nested list comprehension
fl_rk2_np = np.array(fl_rk2)
fl_rk4_np = np.array(fl_rk4)

def RK2(func, func2, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        K1 = X[i] * func[i]
        K2 = (X[i] + K1 * dt / 2) * func2[i]
        X[i+1] = X[i] +  dt * (K1  + K2) / 2
    return X

def RK4(func, func2, func3, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        K1 = X[i] * func[i]
        K2 = (X[i] + K1 * dt / 2) * func2[i]
        K3 = (X[i] + K2 * dt / 2) * func2[i]
        K4 = (X[i] + K3 * dt) * func3[i]
        X[i+1] = X[i] +  dt * (K1  + 2 * K2 + 2* K3 + K4) / 6
    return X

#RK4 w/ Adams-Bashforth 3 Step
def RK4_AB3(func, X0, t):
    X  = np.zeros([nt, len(X0)])
    X[0:3] = X_rk4[0:3]
    for i in range(nt-3):
        X[i+3] = X[i+2] + X[i+2] * dt * ( 23 * func[i+2] - 16 * func[i+1] + 5 * func[i] ) / 12
    return X

%time X_rk2 = RK2(fl_np , fl_rk2_np, X0, t_rk2)
x_rk2 = X_rk2[:,-1]
%time X_rk4 = RK4(fl_np , fl_rk2_np , fl_rk4_np, X0, t_rk2)
x_rk4 = X_rk4[:,-1]
#print(X_rk4)
X_rk4_ab3 = RK4_AB3(fl_np, X0, t_rk2)
x_rk4_ab3 = X_rk4_ab3[:,-1]

# view data
plt.figure(figsize=(6,4),dpi=100)
plt.plot(x.data.numpy(), yb.data.numpy(), color = "orange", label = "Exact solution")
plt.plot(t_odeint,x_odeint, "mv", label = "ODEint")
plt.plot(t_euler, x_euler, "or", label = "Euler")
plt.plot(t_rk2, x_rk2, "gs", label = "RK2")
plt.plot(t_rk2, x_rk4, "cP", label = "RK4")
plt.plot(t_rk2, x_rk4_ab3, "b*", label = "RK4_AB3")
#plt.grid()
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()

plt.figure(figsize=(6,4),dpi=100)
plt.plot(t_euler, np.subtract(x_euler, yb.data.flatten().numpy()), "or", label = "Euler")
plt.plot(t_rk2, np.subtract(x_rk2, yb.data.flatten().numpy()), "gs", label = "RK2")
plt.plot(t_rk2, np.subtract(x_rk4, yb.data.flatten().numpy()), "cP", label = "RK4")
plt.plot(t_rk2, np.subtract(x_rk4_ab3, yb.data.flatten().numpy()), "b*", label = "RK4_AB3")
plt.xlabel('L')
plt.ylabel('g(L)')
plt.legend()
plt.show()
