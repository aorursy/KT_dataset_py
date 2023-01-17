# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def der1(f,x,dx) :

    return (f(x+dx)-f(x))/dx
def bissection(f,a,b,N,realSol) :

    x = (a+b)/2

    dx = 0.00001

    i=0

    E = []

    while der1(f,x,dx) != 0 and i < N :

        x = (a+b)/2

        der = der1(f,x,dx)

        if der > 0 :

            b=x

        elif der < 0: 

            a=x

        else : i = N

        E.append(realSol-x)

        i+=1

    return x,E
f = lambda x: x**2 + 2*x

a=-3

b=6

N=20
x= np.linspace(a,b,1000)

y=f(x)

plt.plot(x,y)
sol, E = bissection(f,a,b,N,-1)

sol
plt.plot(E)
def der2(f,x,dx) :

    return (f(x+dx)+f(x-dx)-2*f(x))/dx**2
def Newton(f,x,epsilon,N) :

    i=0

    dx = 0.001

    x_k = x

    x_kp1 = x_k - der1(f,x,dx)/der2(f,x,dx)

    E=[x_k]

    while np.abs((x_kp1-x_k)/x_kp1) >= epsilon and i<N :

        x_k = x_kp1

        x_kp1 = x_k - der1(f,x_k,dx)/der2(f,x_k,dx)

        E.append(x_kp1)

        i+=1

    return x_kp1 , E
def f(x) : 

    if x>=0 :

        return 4*x**3 - 3*x**4

    else :

        return 4*x**3 + 3*x**4      

f(-1)
X = np.linspace(-6,5,1000)

Y=[]

for x in X :

    Y.append(f(x))

plt.plot(X,Y)
N=300

epsilon = 0.000001

x = 0.4

sol, E = Newton(f,x,epsilon,N)

sol
plt.plot(E)
N=300

epsilon = 0.0001

x = 0.6

sol, E = Newton(f,x,epsilon,N)

sol
plt.plot(E)
N=300

epsilon = 0.0001

x = 0.7

sol, E = Newton(f,x,epsilon,N)

sol
plt.plot(E)
N=300

epsilon = 0.0001

x = -0.7

sol, E = Newton(f,x,epsilon,N)

sol
plt.plot(E)
def Secante(f,x,epsilon,N) :

    i=0

    dx = 0.001

    x_km1 = x

    x_k = x_km1 - dx

    x_kp1=1

    E = [x]

    while i < N and np.abs(x_kp1-x_k/x_kp1)>=epsilon and x_k!= x_km1 :

        

        #Calcul des dérivées en next et en prev

        derx_k = der1(f,x_k,dx)

        derx_km1 = der1(f,x_km1,dx)

        

        x_kp1 = x_k - (derx_k*(x_k-x_km1))/(derx_k-derx_km1)

        

        x_k , x_km1 = x_kp1, x_k

        E.append(x_kp1)

        i+=1

    return x_kp1,E
N=100

epsilon = 0.001

x = 0.4

sol, E = Secante(f,x,epsilon,N)

sol
plt.plot(E)
N=100

epsilon = 0.001

x = 0.6

sol, E = Secante(f,x,epsilon,N)

sol
plt.plot(E)
N=100

epsilon = 0.001

x = 0.7

sol, E = Secante(f,x,epsilon,N)

sol
plt.plot(E)
N=100

epsilon = 0.001

x = -0.7

sol, E = Secante(f,x,epsilon,N)

sol
plt.plot(E)
f = lambda X : (1-X[0])**2 + p*(X[1]-X[0]**2)**2

p=1

f([0,1])

falpha = lambda alpha,X,G: f(X-alpha*G)
def gradf2(f,X,dx) :

    X = np.array(X)

    return np.array([(f(X+[dx,0])-f(X))/dx,(f(X+[0,dx])-f(X))/dx])
gradf2(f,[0,1],0.00001)
def alphaNewton(falpha,alpha,epsilon,X,G,N) :

    dx = 0.001

    x_k = alpha

    x_kp1 = 2*x_k

    i=0

    while np.abs((x_kp1-x_k)/x_kp1) >= epsilon and i < N:

        x_k = x_kp1

        der_1 = (falpha(x_k+dx,X,G)-falpha(x_k,X,G))/dx

        der_2 = (falpha(x_k+dx,X,G)+falpha(x_k-dx,X,G)-2*falpha(x_k,X,G))/dx**2

        x_kp1 = x_k - der_1/der_2

        i+=1

    return x_kp1
def gradientAlg(f,falpha,X,N,epsilon) :

    stop = False

    k=0

    dx=0.001

    E = [X]

    A = []

    X = np.array(X)

    epsilon = 0.001

    Na = 100

    alpha = 0.1

    while not stop :

        alpha= alphaNewton(falpha,alpha,epsilon,X,gradf2(f,X,dx),200)



        X = X - alpha*gradf2(f,X,dx)

        

        if k>N or gradf2(f,X,dx).all() < epsilon :

            stop = True

        else :

            k+=1

        A.append(alpha)

        E.append(X)

    return X,E,A
X = [0,1]

p=1

dx = 0.01

epsilon = 0.001

sol,E,A = gradientAlg(f,falpha,X,100,epsilon)

print("Solution : " ,sol)

print("f(X*)=",f(sol))

gradf2(f,sol,0.000001)
plt.plot(A)
plt.plot(np.array(E)[:,0])
plt.plot(np.array(E)[:,1])
P = np.linspace(1,105,105)

SOLS = []

ATABLO = []

ETABLO = []

for p in P :

    sol,E,A = gradientAlg(f,falpha,X,100,epsilon)

    SOLS.append(list(sol))

    ETABLO.append(E)

    ATABLO.append(A)
plt.plot(np.array(SOLS)[:,0])
plt.plot(np.array(SOLS)[:,1])
n = 50

plt.figure(figsize=(25,20))

for i in range(1,n+1) :

    plt.subplot((n//8)+1,8,i)

    plt.plot(ETABLO[(i-1)*105//(n)])

    plt.title("p="+str((i-1)*105//(n)+1))

plt.show()
f = lambda x,y : (1-x)**2 + p*(y-x**2)**2
def gradf2(f,X,dx) :

    x,y = X[0],X[1]

    return np.array([(f(x+dx,y)-f(x,y))/dx,(f(x,y+dx)-f(x,y))/dx])
def gradf2exact(f,X,dx) :

    x,y = X[0],X[1]

    x1 = 4*p*x**3 - 2*x*(2*p*y-1)-2

    x2 = 2*p*(y-x**2)

    return np.array([x1,x2])
def hess(f,X,dx) :

    x,y = X[0],X[1]

    x11 = (f(x+dx,y)-f(x-dx,y)-2*f(x,y))/dx**2

    x12 = (f(x+dx,y+dx)-f(x+dx,y+dx)-f(x,y+dx)+f(x,y))/dx**2

    x22 = (f(x,y+dx)-f(x,y-dx)-2*f(x,y))/dx**2

    return np.array([[x11, x12],

                     [x12, x22]])
def hessexact(f,X,dx) :

    x,y = X[0],X[1]

    x11 = 2*p*x-4*p*y+2

    x12 = -4*p*x

    x22 = 2*p

    return np.array([[x11, x12],

                     [x12, x22]])
X = [0,1]

p=1

dx = 0.001

print("Approximate Hessian: \n",hess(f,X,dx))

print("Exact derivative Hessian: \n",hessexact(f,X,dx))
def NewtonAlg(f,X,N,epsilon) :

    stop = False

    k=0

    dx=0.001

    X = np.array(X)

    E = [X]

    while not stop :

        D = hessexact(f,X,dx)

        Dinv = np.linalg.inv(D)

        grad = gradf2exact(f,X,dx)

        X = X - np.dot(Dinv,grad)

        if k>N or gradf2(f,X,dx).all()<epsilon:

            stop = True

        else :

            k+=1

        E.append(X)

    return X,E
X=[0,1]

p=1

epsilon = 0.001

sol, E = NewtonAlg(f,X,100,epsilon)

sol ,f(sol[0],sol[1])
plt.plot(E)
fnewt = lambda x: f(x,x) 
def Newton(f,x,epsilon,N) :

    i=0

    dx = 0.001

    x_k = x

    x_kp1 = x_k - der1(f,x,dx)/der2(f,x,dx)

    E=[x_k]

    while np.abs((x_kp1-x_k)/x_kp1) >= epsilon and i<N :

        x_k = x_kp1

        x_kp1 = x_k - der1(f,x_k,dx)/der2(f,x_k,dx)

        E.append(x_kp1)

        i+=1

    return x_kp1 , E
N=300

epsilon = 0.0001

x = 0

sol, E = Newton(fnewt,x,epsilon,N)

sol
plt.plot(E)
f = lambda X : (1-X[0])**2 + p*(X[1]-X[0]**2)**2
def gradf2(f,X,dx) :

    X = np.array(X)

    return np.array([(f(X+[dx,0])-f(X))/dx,(f(X+[0,dx])-f(X))/dx])
def amijo_goldstein(f,X,d,alpha,beta,lam) :

    if (beta >=1 or beta<=0 or lam >=1 or lam<=0 ) :

        print("Les paramètres alpha et beta doivent être compris dans l'intervalle 0,1 bornes non comprises")

        return 0

    

    while f(X+alpha*d) >= f(X)+alpha*beta*np.dot(gradf2(f,X,0.001).T,d) :

        alpha = lam * alpha

    return alpha
def gradientAlg(f,falpha,X,N,epsilon) :

    stop = False

    k=0

    dx=0.001

    E = [X]

    A = []

    X = np.array(X)

    epsilon = 0.001

    Na = 100

    alpha = 10

    beta = 0.5

    lam = 0.5

    while not stop :

        alpha = 10

        d = -gradf2(f,X,dx)

        alpha= amijo_goldstein(f,X,d,alpha,beta,lam)



        X = X - alpha*gradf2(f,X,dx)

        

        if k>N or gradf2(f,X,dx).all()<epsilon :

            stop = True

        else :

            k+=1

        A.append(alpha)

        E.append(X)

    return X,E,A
X = [0,1]

p=1

dx = 0.01

epsilon = 0.1

sol,E,A = gradientAlg(f,falpha,X,100,epsilon)

print("Solution : " ,sol)

print("f(X*)=",f(sol))
plt.plot(A)
plt.plot(np.array(E)[:,0])
plt.plot(np.array(E)[:,1])