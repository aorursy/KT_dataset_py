import numpy as np

import holoviews as hv



hv.extension('matplotlib')
x0 = -5; x1 = +5

x = np.linspace(x0,x1,100)
def f(x):

    return np.exp(-x/4)*np.cos(x)*np.cos(x)
curve = hv.Curve( (x,f(x)) )
curve
N = 5000 # number of random dots in x-y plane

x = np.random.uniform(low=x0, high=x1, size=N)

y = np.random.uniform(low=0, high=np.max(f(x)), size=N)
%opts Scatter (alpha=0.5 s=5)



dots = hv.Scatter( (x,y) )
dots*curve
X = x[y < f(x)]
Y = y[y < f(x)]
selected_dots = hv.Scatter( (X,Y) )
dots*selected_dots*curve
len(X)
print("The resulting efficiency (the yield of useful values) is %.1f %%" % (100*len(X)/N) )
efficiency = len(X)/N
Np = 10000
N = int(1.1*(Np/efficiency))
N
x = np.random.uniform(low=x0, high=x1, size=N)

y = np.random.uniform(low=0, high=np.max(f(x)), size=N)
X = x[y < f(x)]
len(X)
X = X[0:Np]
len(X)
Y = y[y < f(x)]

Y = Y[0:Np]
hv.Scatter( (X,Y) )