import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('../input/eps-test/EPS.xlsx')
dfp=df.rename(columns={"E": "E_dy", "stiffness": "E_static", "re": "Re"}, errors="raise")

cor=dfp.corr()
plt.subplots(figsize=(7,7))         
sns.heatmap(cor,square=True,annot=True)

g = sns.PairGrid(dfp, y_vars=["E_dy"], x_vars=["density", "yield",'E_static','Re','f'], height=4.5,  aspect=1.1)
ax = g.map(plt.scatter, alpha=0.6)
ax = g.add_legend()


sns.pairplot(df,palette="husl",diag_kind="kde");
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(X,a, n1,n2,n3):
    X= re,dens,f 
    return a*(dens/981)**n1/((re)**n2)*f**n3
dens=df['density'].to_numpy()
re=df['re'].to_numpy()
E=df['E'].to_numpy()
f=df['f'].to_numpy()

plt.scatter(dens, E,  label='data')
p0 = 10, 5,.5,.5
popt, pcov = curve_fit(func, (re,dens,f), E, p0,)
popt
plt.plot(dens, func((re,dens), *popt), 'g--',label='fit: a=%5.3f, n1=%5.3f, n2=%5.3f, n3=%5.3f' % tuple(popt))

plt.xlabel('dens')
plt.ylabel('E')
plt.legend()
plt.show()

#calculate R2
residuals = E- func((re,dens,f), *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((E-np.mean(E))**2)
r_squared = 1 - (ss_res / ss_tot)
print('R2=',r_squared)
print('parameters',popt)

import seaborn as sns
from sklearn.metrics import r2_score
df=pd.read_excel('../input/eps-test/EPS.xlsx')
df['Var']=3685*(df['density']/981)**1.7322/df['re']**0.540*df['f']**.05
dfp=df.rename(columns={"E": "E_dy", "stiffness": "E_static", "re": "Re"}, errors="raise")
print('R2 =',r2_score(df['Var'],df['E']))
sns.regplot(x="E", y="Var", data=df);
df
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import axes3d



# Make data.
x= np.array([10,15,20,25,32])
y = np.array([0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.4])
z = df['E']
X,Y = np.meshgrid(x,y)
Z = 3685*(X/981)**1.7322/(Y**0.540)*(1)**.05

mycmap = plt.get_cmap('gist_earth')
fig = plt.figure(figsize=(13,10))
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel('Re', fontsize=10, rotation=0)
ax1.set_ylabel('Density (kg/cu.m)')
ax1.set_zlabel('Edyn (Mpa)', fontsize=10, rotation=60)
ax1.set_title('f = 1Hz',fontsize=12,)



# Plot a 3D surface
#ax.plot_surface(X, Y, Z)
surf1 = ax1.plot_surface(Y, X, Z, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=8)

plt.show()
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import axes3d



# Make data.
x= np.array([1,2,3,4,5])
y = np.array([0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.4])
z = df['E']
X,Y = np.meshgrid(x,y)
Z = 3685*(20/981)**1.7322/(Y**0.540)*(X)**.05

mycmap = plt.get_cmap('gist_earth')
fig = plt.figure(figsize=(13,10))
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel('Re', fontsize=10, rotation=0)
ax1.set_ylabel('f (Hz)')
ax1.set_zlabel('Edyn (MPa)', fontsize=10, rotation=60)
ax1.set_title('density= 20 kg/cu.m',fontsize=12,)



# Plot a 3D surface
#ax.plot_surface(X, Y, Z)
surf1 = ax1.plot_surface(Y, X, Z, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=8)

plt.show()
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import axes3d



# Make data.
x= np.array([10,15,20,25,32])
y = np.array([1,2,3,4,5])
z = df['E']
X,Y = np.meshgrid(x,y)
Z = 3685*(X/981)**1.7322/(0.8**0.540)*(Y)**.05

mycmap = plt.get_cmap('gist_earth')
fig = plt.figure(figsize=(13,10))
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel('f', fontsize=10, rotation=0)
ax1.set_ylabel('Density (kg/cu.m)')
ax1.set_zlabel('Edyn (Mpa)', fontsize=10, rotation=60)
ax1.set_title('Re = 0.8',fontsize=12,)



# Plot a 3D surface
#ax.plot_surface(X, Y, Z)
surf1 = ax1.plot_surface(Y, X, Z, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=8)

plt.show()
Z
