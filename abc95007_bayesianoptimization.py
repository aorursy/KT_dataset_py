# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def Gamma(x):
    return x ** 2.2 / 255 ** 2.2
X = np.linspace(0,255,255)
Y = [Gamma(x) for x in X]
plt.plot(X, Y)
num = 5

# 目標曲線
X = np.arange(0, 256, 1)
Y = [Gamma(x) for x in X]

# 綁點位置
xp = np.random.randint(1, 254, num)
xp = sorted(np.append(xp, [0, 255]))
yp = [Gamma(i) for i in xp]

# 內插曲線
x = np.arange(0, 256, 1)
y = [np.interp(i, xp, yp) for i in x]

# 計算 loss
loss = -(abs(np.asarray(Y) - np.asarray(y))).sum()

plt.figure(figsize=(16,8))
plt.plot(x, y, color = "r", alpha=0.6)
plt.scatter(xp, yp, color = "r",  alpha=0.6)
plt.plot(X, Y, color= "b", alpha=0.6)
print("loss", loss)
def lineCurve(x1, x2, x3, x4, x5, isShow=False):
    # 目標曲線
    X = np.arange(0, 256, 1)
    Y = [Gamma(x) for x in X]
    
    # 綁點位置
    #xp = np.random.randint(1, 254, num)
    xp = sorted(np.append([x1, x2, x3, x4, x5], [0, 255]))
    yp = [Gamma(i) for i in xp]
    
    # 內插曲線
    x = np.arange(0, 256, 1)
    y = [np.interp(i, xp, yp) for i in x]
    
    # 計算 loss
    loss = -(abs(np.asarray(Y) - np.asarray(y))).sum()
    
    if(isShow):
        print("loss=", loss)
        plt.figure(figsize=(16,8))
        plt.plot(x, y, color = "r", alpha=0.6)
        plt.scatter(xp, yp, color = "r",  alpha=0.6)
        plt.plot(X, Y, color= "b", alpha=0.6)
    return loss
pbounds = {'x1': (1, 255), 'x2': (1, 255), 'x3': (1, 255), 'x4': (1, 255), 'x5': (1, 255)}
optimizer = BayesianOptimization(f=lineCurve, pbounds=pbounds)
optimizer.maximize(init_points=5,n_iter=100)
print(optimizer.max)

ans = list(optimizer.max["params"].values())
lineCurve(ans[0], ans[1], ans[2], ans[3], ans[4], True)

def fitCurve(y1, y2, y3, y4, y5, y6, y7):
    x1, x2, x3, x4, x5, x6, x7 = 30, 60, 120, 150, 180 , 210, 255
    X = [x1, x2, x3, x4, x5, x6, x7]
    Y = [y1, y2, y3, y4, y5, y6, y7]
    loss = 0
    for i in range(len(X)):
        loss = loss-abs(Gamma(X[i])-Y[i])
    return 30*loss
print("各點 y 值 " + str([Gamma(x) for x in [30, 60, 120, 150, 180 , 210, 255]]))
pbounds = {'y1': (0, 1), 'y2': (0, 1), 'y3': (0, 1), 'y4': (0, 1), 'y5': (0, 1), 'y6': (0, 1), 'y7': (0, 1)}
optimizer = BayesianOptimization(f=fitCurve, pbounds=pbounds)
optimizer.maximize(init_points=5,n_iter=50)
optimizer.max
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    #return -(x-2) ** 2 - (y - 1) ** 2 + 1
    #return -np.sqrt(x**2+y**2 +x*y)
    return 10*(1-x+(x)**4+y**3)*np.exp(-x**2-y**2)

X = np.linspace(-3,3,50)
Y = np.linspace(-3,3,50)
Z = [black_box_function(x,y) for x in X for y in Y]
Z = np.asarray(Z).reshape(50,50)

plt.contourf(X,Y,Z,8,alpha=0.75,cmap=plt.cm.hot)
C = plt.contour(X,Y,Z,8,color='black',lw=0.5)
plt.clabel(C,inline=True,fontsize=10, fmt="%d")
Z.max()
# Bounded region of parameter space
pbounds = {'x': (-3, 3), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
)
optimizer.maximize(
    init_points=3,
    n_iter=20,
)
optimizer.max
plt.figure(figsize=(8,8))
plt.contourf(X,Y,Z,8,alpha=0.75,cmap=plt.cm.hot)
C = plt.contour(X,Y,Z,8,color='black',lw=0.5)
plt.clabel(C,inline=True,fontsize=10, fmt="%d")
for i in range(len(optimizer.res)):
    plt.scatter(optimizer.res[i]["params"]["y"],  optimizer.res[i]["params"]["x"])
def black_box_function2(x, y):
    return 10*(1-x/2+x**5+y**5) * np.exp(-x**2-y**2)

X = np.linspace(-3,3,50)
Y = np.linspace(-3,3,50)
Z = [black_box_function2(x,y) for x in X for y in Y]
Z = np.asarray(Z).reshape(50,50)

plt.figure(figsize=(8,8))
plt.contourf(X,Y,Z,8,alpha=0.75,cmap=plt.cm.hot)
C = plt.contour(X,Y,Z,8,color='black',lw=0.5)
plt.clabel(C,inline=True,fontsize=10, fmt="%d")
print("Z max =", Z.max())
# fit
pbounds = {'x': (-3, 3), 'y': (-3, 3)}
optimizer = BayesianOptimization(f=black_box_function2,pbounds=pbounds,)
optimizer.maximize(init_points=5,n_iter=30)
print("optimizer max=", optimizer.max)

# plot
plt.figure(figsize=(8,8))
plt.contourf(X,Y,Z,8,alpha=0.75,cmap=plt.cm.hot)
C = plt.contour(X,Y,Z,8,color='black',lw=0.5)
plt.clabel(C,inline=True,fontsize=10, fmt="%d")
for i in range(len(optimizer.res)):
    plt.scatter(optimizer.res[i]["params"]["y"],  optimizer.res[i]["params"]["x"])
from hyperopt import fmin, tpe, hp
best = fmin(
    fn=lambda x: (x-1)**2,
    space=hp.uniform('x', -2, 2),
    algo=tpe.suggest,
    max_evals=100)
print(best)
from hyperopt import  hp,fmin,rand, tpe, space_eval
from hyperopt import Trials

trials = Trials()
result = []
def black_box_function3(space):
    x = space["x"]
    y = space["y"]
    loss = -10*(1-x/2+x**5+y**5) * np.exp(-x**2-y**2)
    result.append(loss)
    return loss

space = {
    "x": hp.uniform('x', -3, 3),
    "y": hp.uniform('y', -3, 3),
}
best = fmin(fn=black_box_function3, space=space, algo=tpe.suggest, max_evals=200)
print(best)
print(black_box_function3(best))
len([0]*10)