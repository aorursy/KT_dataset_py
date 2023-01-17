import numpy as np

np.random.seed(40643)

import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import Matern #,DotProduct, WhiteKernel

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from scipy.interpolate import interp2d, interp1d
def branin(x,y,a=1,b=5.1/(4*np.pi**2),c=5/np.pi,r=6,s=10,t=1/(8*np.pi)):

    return a*(y-b*x**2+c*x-r)**2+s*(1-t)*np.cos(x)+s
X=np.linspace(-5,10,1500)

Y=np.linspace(0,15,1500)

points=[(x,y) for y in Y for x in X]
Z=[branin(x,y) for (x,y) in points]

m=max(Z)

colors=cm.rainbow([z/m for z in Z])
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter([p[0] for p in points],[p[1] for p in points],Z, zdir='z', s=0.1, c=colors, depthshade=True)
coordinates=[(x,y,z) for x,y,z in zip([p[0] for p in points],[p[1] for p in points],Z)]

init_points=np.random.choice(len(coordinates),100)

X,y=np.asarray([points[i] for i in init_points]),[Z[i] for i in init_points]

X_test,y_test=np.asarray([points[i] for i in range(len(coordinates)) if i not in init_points]),[Z[i] for i in range(len(coordinates)) if i not in init_points]

# small=np.random.choice([i for i in range(len(coordinates)) if i not in init_points],100)

# X_test_small,y_test_small=np.asarray([points[i] for i in small]),[Z[i] for i in small]
def mse(true,estim):

    loss=0

    for t,e in zip(true,estim):

        loss+=(t-e)**2

    loss=loss/len(true)

    return loss
dic_errors={'branin':0}
def interpol_2d(kind,X=X,y=y):

    fun=interp2d([x[0]for x in X],[x[1] for x in X],y,kind=kind)

    return fun
linear=interpol_2d('linear')

Z_linear=[linear(x[0],x[1]) for x in X_test[::2]]

Z_linear=np.concatenate(Z_linear)
dic_errors['linear']=mse(Z_linear,y_test[::2])

dic_errors
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_linear])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_linear, zdir='z', s=0.1, c=colors, depthshade=True)
cubic=interpol_2d('cubic')

Z_cubic=[cubic(x[0],x[1]) for x in X_test[::2]]

Z_cubic=np.concatenate(Z_cubic)

dic_errors['cubic_splines']=mse(Z_cubic,y_test[::2])

dic_errors
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_cubic])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_cubic, zdir='z', s=0.1, c=colors, depthshade=True)
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_cubic])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_cubic, zdir='z', s=0.1, c=colors, depthshade=True)

ax.set_zlim(0, 300)
neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(X, y)

Z_knn=neigh.predict(X_test[::2])

dic_errors['knn'] = mse(Z_knn,y_test[::2])

dic_errors
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_knn])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_knn, zdir='z', s=0.1, c=colors, depthshade=True)
regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=100, random_state=0)

regr.fit(X, y)

Z_randforest=regr.predict(X_test[::2])

dic_errors['random_forest'] = mse(Z_randforest,y_test[::2])

dic_errors
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_randforest])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_randforest, zdir='z', s=0.1, c=colors, depthshade=True)
kernel=Matern(length_scale=1,nu=2.5)

gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X, y)

print(gpr.score(X, y))
Z_krig=gpr.predict(X_test[::2], return_std=False)

dic_errors['kriging'] = mse(Z_krig,y_test[::2])

dic_errors
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_krig])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_krig, zdir='z', s=0.1, c=colors, depthshade=True)
Z_error=gpr.predict(X_test[::2], return_std=True)
Z_error=Z_error[1]
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

mm=max(Z_error)

colors=cm.gist_earth([z/mm for z in Z_error])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_error, zdir='z', s=0.1, c=colors, depthshade=True)
dic_errors_montecarlo=dic_errors

dic_errors_grid={'branin' : 0}
def ackley(x,y=0):

    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.exp(1) + 20
ackley_x=np.linspace(-5,5,100000)

ackley_y=[ackley(x) for x in ackley_x]

ackley_m=max(ackley_y)

colors=cm.rainbow([y/ackley_m for y in ackley_y])

plt.scatter(ackley_x,ackley_y,s=0.1,c=colors)
ackley_grid_x=np.linspace(-5,5,11)

ackley_grid_y=[ackley(x) for x in ackley_grid_x]

colors_2=cm.rainbow([y/ackley_m for y in ackley_grid_y])

plt.scatter(ackley_x,ackley_y,s=0.1,c=colors)

for x,y,c in zip(ackley_grid_x,ackley_grid_y,colors_2):

    plt.axvline(x=x,linestyle='--',linewidth=0.3,c=c)

#     plt.axhline(y=y,linestyle='--',linewidth=0.3,c=c)

    plt.plot(x,y,'o',c=c)
ackley_interp_linear=[interp1d(ackley_grid_x,ackley_grid_y,kind='linear')(x) for x in ackley_x]

ackley_interp_quadratic=[interp1d(ackley_grid_x,ackley_grid_y,kind='quadratic')(x) for x in ackley_x]

labels=['Linear Interpolation','Quadratic Interpolation','Original Ackley Function']

plt.figure(figsize=(8,8))

plt.scatter(ackley_x,ackley_y,s=0.1,c=colors)

plt.plot(ackley_x,ackley_interp_linear,'k',linewidth=0.4)

plt.plot(ackley_x,ackley_interp_quadratic,'slategray',linewidth=0.4)

for x,y,c in zip(ackley_grid_x,ackley_grid_y,colors_2):

    plt.axvline(x=x,linestyle='--',linewidth=0.3,c=c)

    plt.plot(x,y,'o',c=c)

plt.legend(labels)
grid_x=np.linspace(-5,10,10)

grid_y=np.linspace(0,15,10)

points_grid = [(x,y) for x in grid_x for y in grid_y]

branin_grid=[branin(x,y) for x in grid_x for y in grid_y]
linear=interpol_2d('linear',X=points_grid,y=branin_grid)

Z_linear=[linear(x[0],x[1]) for x in X_test[::2]]

Z_linear=np.concatenate(Z_linear)

dic_errors_grid['linear'] = mse(Z_linear,y_test[::2])

dic_errors_grid
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_linear])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_linear, zdir='z', s=0.1, c=colors, depthshade=True)
kernel=Matern(length_scale=1,nu=2.5)

gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(points_grid, branin_grid)

krig=gpr.predict(X_test[::2], return_std=True)

Z_krig,Z_error=krig[0],krig[1]

dic_errors_grid['kriging'] = mse(Z_krig,y_test[::2])

dic_errors_grid
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

colors=cm.rainbow([z/m for z in Z_krig])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_krig, zdir='z', s=0.1, c=colors, depthshade=True)
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

max_error=max(Z_error)

colors=cm.gist_earth([z/max_error for z in Z_error])

ax.scatter([x[0] for x in X_test[::2]],[x[1] for x in X_test[::2]],Z_error, zdir='z', s=0.1, c=colors, depthshade=True)