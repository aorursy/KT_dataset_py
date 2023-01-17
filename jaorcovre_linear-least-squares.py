import numpy as np

import pandas as pd

import plotnine as p9

import matplotlib.pyplot as plt

import sympy as sp

import re



from random import uniform as rand

from matplotlib import cm

from numpy import array, transpose, linspace, sum, exp, log

from numpy.linalg import inv

from pandas import DataFrame as df

%matplotlib inline



# Style definitions

plt.style.use('dark_background')

p9.theme_set(

    p9.theme_dark() + 

    p9.theme(rect=p9.element_rect(color='black', size=3, fill='black')) +

    p9.theme(text=p9.element_text(color='lightgray')))

COLORS = {'red': '#ff2626',

          'orange': '#f29524',

          'green': '#169340',

          'blue': '#377CB9',

          'white': '#EEEEEE',

          'black': '#222222'}



# Misc utils

def squaremesh(low,high,res):

    return np.meshgrid(np.linspace(low,high,res),np.linspace(low,high,res))

def randspace(low, high, res):

    return array([rand(low, high) for _ in range(res)])

def field(f, low, high, res):

    X, Y = squaremesh(low, high, res)

    return [X, Y, f(X, Y)]

def unbox(dictionary, *items):

    return [dictionary[item] for item in items]



# Plot functions

def ax3D():

    ax = plt.axes(projection='3d', facecolor=(0,0,0,0))

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_zlabel('z')

    return ax  

def ax2D():

    ax = plt.axes(facecolor=(0,0,0,0))

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    return ax  

def darkgraph():

    return plt.figure(figsize=(20,10), facecolor=(0.1,0.1,0.1))

def surface(X, Y, Z, fig, ax):

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

    fig.colorbar(surf, shrink=0.6, aspect=5)

def translucent(X, Y, Z, fig, ax, res=300, alpha=0.6):

    surf = ax.contourf(X, Y, Z, res, cmap=cm.coolwarm, linewidth=0, alpha=alpha)

    fig.colorbar(surf, shrink=0.5, aspect=5)

def fitplot2D(X, Y, f, res=50):

    if len(X)!=len(Y): raise

    d1 = df({'x':X,'y':Y,'Source':['Original']*len(X)})

    Xf = linspace(min(X),max(X),res)

    d2 = df({'x':Xf,'yf':f(Xf),'Source':['Fitted']*len(Xf)})

    return (p9.ggplot(pd.concat([d1,d2])) +

            p9.geom_point(p9.aes('x', 'y', color = 'Source')) +

            p9.geom_path(p9.aes('x', 'yf', color = 'Source')))

def fitplot3D(X, Y, Z, f, fig, ax, res=50, alpha=0.2, pcolor='black'):

    if not len(X)==len(Y)==len(Z): raise

    Xf,Yf = squaremesh(min(X),max(X),res)

    Zf = array([f(x,y) for x,y in zip(np.ravel(Xf), np.ravel(Yf))])

    Zf = Zf.reshape(Xf.shape)

    translucent(Xf,Yf,Zf,fig,ax,res=300,alpha=0.2)

    ax.scatter(X, Y, Z, s=9, color=pcolor)



# Ignore warnings (keep commented during editting)

from warnings import filterwarnings

filterwarnings('ignore')



# Clear this cell's output

from IPython.display import clear_output

clear_output()
def poly2D(X, P): 

    return sum([P[i] * X**i for i in range(len(P))], axis=0)



def straightfit(X, Y):

    if len(X)!=len(Y): raise

    At = np.array([[1]*len(X), X])

    A = np.transpose(At)

    return inv(At.dot(A)).dot(At.dot(Y))



X = array([-0.47, -0.26,  0.15,  0.82, -0.60])

Y = array([ 1.14,  1.21,  1.28,  1.47,  0.93])

P = straightfit(X, Y)

f = lambda X: poly2D(X, P)



fitplot2D(X, Y, f).draw();
def polyfit(X, Y, o):

    if len(X)!=len(Y): raise

    At = np.array([[x**n for x in X] for n in range(o+1)])

    A = np.transpose(At)

    return inv(At.dot(A)).dot(At.dot(Y))



X = array([-0.47, -0.26,  0.15,  0.82, -0.60])

Y = array([ 0.12,  0.25,  0.18,  0.26,  -0.11])

P = polyfit(X, Y, 3)

f = lambda X: poly2D(X, P)



fitplot2D(X, Y, f).draw();
nf = lambda X, a, b: 1 - 0.5*X + 2*X**2 + rand(a, b) 



X = linspace(-1,1,100)

Y = array([nf(x, -0.3, 0.3) for x in X]) 

a,b,c = polyfit(X, Y, 2)

f = lambda X: a + b*X + c*X**2



fitplot2D(X, Y, f).draw();
f = lambda X: 0.5 * exp(2.5 * X)



X = randspace(0,1,5)

Y = f(X)

c, b = polyfit(X, log(Y), 1)

c = exp(c)

f = lambda X: c * exp(b * X)



fitplot2D(X, Y, f).draw();
def opoly(v, o):

    if len(v) == 1:

        p = []

        for n in range(o + 1):

            p.append(v[0]**n)

        return p

    p = []

    q = opoly(v[1:],o)

    for n in range(o + 1):

        for t in q:

            p.append(v[0]**n * t)

    return p



def opolyfit(data, order, vec_out):

    A = np.array([opoly(v, order) for v in data])

    At = np.transpose(A)

    return np.linalg.inv(At.dot(A)).dot(At.dot(vec_out))



def opolynomial(var, coef):

    p = opoly(var, int(len(coef)**(1. / len(var)) - 1))

    return sum([coef[i] * p[i] for i in range(len(coef))])



P = randspace(-1,1,9)

X = randspace(-1,1,50)

Y = randspace(-1,1,50)

Z = array([opolynomial(v, P) + rand(-0.1,0.1) for v in np.stack((X, Y), axis=1)])



Pf = opolyfit(transpose([X,Y]),2,Z)

f = lambda X, Y: opolynomial([X,Y],Pf)



fitplot3D(X,Y,Z,f,darkgraph(),ax3D())
data = pd.read_csv("../input/microwave250gwatter/microwave.csv")



t, Ti, Tf = unbox(data, 't', 'Ti', 'Tf')



dT = Tf - Ti



X = array(t)

Y = array(dT)

P = straightfit(X, Y)

f = lambda X: poly2D(X, P)



(

    fitplot2D(X, Y, f) +

    p9.labels.ggtitle("$250ml$ of Watter on My Microwave") +

    p9.labels.ylab("Temperature Change (°C)") +

    p9.labels.xlab("Time inside Microwave (s)")

).draw();



formula = '$t \\approx \\frac{T_f - T_0 - %g}{%g}$' % (P[0],P[1])

plt.plot()

plt.text(83,55,formula,fontsize=14)

plt.show()
data = pd.read_csv('../input/english-word-frequency/unigram_freq.csv').head(1000)

print(data)



X = array(data.index)+1

Y = array(data['count'])

a, b = polyfit(log(X), log(Y), 1)

c = exp(a) ; 

f = lambda x: c * x**b



(

    fitplot2D(X, Y, f, res=50000) +

    p9.scales.scale_y_log10() +

    p9.scales.scale_x_log10() +

    p9.labels.ggtitle("Google's English Internet 1000 Most Used Words") +

    p9.labels.ylab("Number of Occurences") +

    p9.labels.xlab("Ranking (from most to least common)")

).draw();



formula = '$frequency \\approx (%.2g) \\times ranking^{%.2g}$ \n' % (c,b)

plt.plot()

plt.text(0.05,7.9,formula,fontsize=14)

plt.show()
data = pd.read_table('../input/galton-height-data/galton-stata11.csv')



X = F = data['father']

Y = M = data['mother']

Z = H = data['height']



Pf = opolyfit(transpose([X,Y]),1,Z)

f = lambda X, Y: opolynomial([X,Y],Pf)

fig=darkgraph() ; ax=ax3D()

fitplot3D(X,Y,Z,f,fig,ax)

ax.set_xlabel("Father's Height")

ax.set_ylabel("Mother's Height")

ax.set_zlabel("Kid's Height");
Mean = np.mean([F, M], axis=0)

D = H - Mean

X = Mean ; Y = D



P = straightfit(X, Y)

f = lambda X: poly2D(X, P)

mean = pd.DataFrame({'x':[np.mean(Mean)],'y':[0],'Source':['Mean']})

(

    fitplot2D(X, Y, f) +

    p9.geom_point(p9.aes('x', 'y', color='Source'), data=mean, size=3) +

    p9.labels.ggtitle("Height difference between children and their parents") +

    p9.labels.ylab("Height Difference") +

    p9.labels.xlab("Mid-Parental Height") +

    p9.scale_color_manual(unbox(COLORS,"red", "blue", "orange"))

).draw();