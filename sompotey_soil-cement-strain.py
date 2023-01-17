import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('../input/soil-cement/response.xlsx')
#dfp=df.rename(columns={"E": "E_dy", "stiffness": "E_static", "re": "Re"}, errors="raise")

cor=dfp.corr()
plt.subplots(figsize=(7,7))         
sns.heatmap(cor,square=True,annot=True)

g = sns.PairGrid(dfp, y_vars=["q"], x_vars=["p", "f"], height=4.5,  aspect=1.1)
ax = g.map(plt.scatter, alpha=0.6)
ax = g.add_legend()


sns.pairplot(df,palette="husl",diag_kind="kde");
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(X,a, n1):
    X= p,f 
    return -a*(p)**n1*(f)
q=df['q'].to_numpy()
p=df['p'].to_numpy()
f=df['f'].to_numpy()

plt.scatter(q, p,  label='data')
p0 = 1,1
popt, pcov = curve_fit(func, (p,f), q, p0,)
popt
plt.plot(p, func((p,f), *popt), 'g--',label='fit: a=%5.3f, n1=%5.3f' % tuple(popt))

plt.xlabel('p')
plt.ylabel('q')
plt.legend()
plt.show()
#calculate R2
residuals=q-func((p,f), *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((q-np.mean(q))**2)
r_squared = 1 - (ss_res / ss_tot)
print('R2=',r_squared)
print('parameters',popt)


ss_res
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import axes3d



# Make data.
x= np.array([50,100,150,200,250,300])
y = np.array([0,-0.5,-1,-1.5,-2,-2.5,-3,-3.5,-4,-4.5,-5])
z = df['q']
X,Y = np.meshgrid(x,y)
Z = -0.93*(X)**0.776*Y
mycmap = plt.get_cmap('gist_earth')
fig = plt.figure(figsize=(13,10))
ax1 = fig.add_subplot(111, projection='3d')

ax1.set_xlabel('state parameter', fontsize=10, rotation=0)
ax1.set_ylabel('p')
ax1.set_zlabel('q', fontsize=10, rotation=60)
ax1.set_title('strain=0.1%',fontsize=12,)



# Plot a 3D surface
#ax.plot_surface(X, Y, Z)
surf1 = ax1.plot_surface(Y, X, Z, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=8)

plt.show()
df
Z
