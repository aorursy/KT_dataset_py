import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('../input/eps-creep/EPSset.xlsx')
df['p2']=df['p2']/1.2e3
df['p3']=df['p3']/1.2e3
#Elastic strain
ee=28.8/7700
print(ee)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(X,A,n,B):
    X= t
    return 0.003+A*(28.8/121.)**n*(t**(-B*np.log10(1-(28.8/121.))))

df['t_hr']=df['day']*24
p2=-df['p2'].to_numpy()
t=df['t_hr'].to_numpy()


plt.scatter(t, p2,  label='data')
p0 = .00209, 2.47,0.5
popt, pcov = curve_fit(func, t, p2, p0,)
popt
plt.plot(t, func((t), *popt), 'g--',label='fit: a=%f, n=%f,B=%f' % tuple(popt))

plt.xlabel('t')
plt.ylabel('p2')
plt.legend()
plt.show()
print('coeficient',popt)

#modelprevious
A_p=0.0029
n_p=2.47
B_p=0.9
#modelfit
A=0.00540921
n=1.15381866
B=3.03032334
func_p=0.003+A_p*(28.8/121.)**n_p*(t**(-B_p*np.log10(1-(28.8/121.))))
func1=0.003+A*(28.8/121.)**n*(t**(-B*np.log10(1-(28.8/121.)))) #P2
func_p3=0.003+A*(22.8/121.)**n*(t**(-B*np.log10(1-(22.8/121.))))
#plotting graph
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-5,5,100)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.scatter(df['t_hr'], df['p2'], label='p2')
plt.plot(df['t_hr'], df['p3'], '-r', label='p3',color='blue',linestyle='dashed')
plt.plot(df['t_hr'], -func_p, '-r', label='predict_p2_p',color='green')
plt.plot(df['t_hr'], -func1, '-r', label='predict_p2',color='red')
plt.plot(df['t_hr'], -func_p3, '-r', label='predict_p3',color='blue')
plt.legend(loc='upper right')
plt.show()
popt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(X,A,B,n):
    X= t
    return 0.003+A*np.sinh(28.8/54.2)+B*np.sinh(28.8/33)*t**n

df['t_hr']=df['day']*24
p2=-df['p2'].to_numpy()
t=df['t_hr'].to_numpy()


plt.scatter(t, p2,  label='data')
p0 = .00209, 2.47,0.5
popt, pcov = curve_fit(func, t, p2, p0,)
popt
plt.plot(t, func((t), *popt), 'g--',label='fit: a=%f, b=%f,n=%f' % tuple(popt))

plt.xlabel('t')
plt.ylabel('p2')
plt.legend()
plt.show()
print('coeficient',popt)
#this research
A=-0.00611159
B=0.00169727
n=0.31842576
#previous model
A_p=1.1
B_p=0.0305
n_p=0.2
stress_i=28.8
func=0.003+A*np.sinh(28.8/54.2)+B*np.sinh(28.8/33)*t**n
func_p3=0.003+A*np.sinh(22.8/54.2)+B*np.sinh(22.8/33)*t**n
#plotting graph
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-5,5,100)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(df['t_hr'], df['p2'], '-r', label='p2',linestyle='dashed')
plt.plot(df['t_hr'], df['p3'], '-r', label='p3',color='blue',linestyle='dashed')
plt.plot(df['t_hr'], -func, '-r', label='predict_p2',color='red')
plt.plot(df['t_hr'], -func_p3, '-r', label='predict_p3',color='blue')
plt.legend(loc='upper right')
plt.show()

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func(X,B,n):
    X= t
    return B*np.sinh(28.8/33)*t**n

df['t_hr']=df['day']*24
p2=-df['p2'].to_numpy()
t=df['t_hr'].to_numpy()


plt.scatter(t, p2,  label='data')
p0 = 2.47,0.5
popt, pcov = curve_fit(func, t, p2, p0,)
popt
plt.plot(t, func((t), *popt), 'g--',label='fit: b=%f,n=%f' % tuple(popt))

plt.xlabel('t')
plt.ylabel('p2')
plt.legend()
plt.show()
print('coeficient',popt)
#this research

B=0.00161039
n=0.32264529
#previous model
A_p=1.1
B_p=0.0305
n_p=0.2
stress_i=28.8
func=B*np.sinh(28.8/33)*t**n
func_p3=B*np.sinh(22.8/33)*t**n
#plotting graph
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-5,5,100)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(df['t_hr'], df['p2'], '-r', label='p2',linestyle='dashed')
plt.plot(df['t_hr'], df['p3'], '-r', label='p3',color='blue',linestyle='dashed')
plt.plot(df['t_hr'], -func, '-r', label='predict_p2',color='red')
plt.plot(df['t_hr'], -func_p3, '-r', label='predict_p3',color='blue')
plt.legend(loc='upper right')
plt.show()


df['day']
