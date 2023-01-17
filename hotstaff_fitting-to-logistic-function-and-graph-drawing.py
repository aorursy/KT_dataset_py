import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as pl



from scipy.optimize import curve_fit
# csv read

df1 = pd.read_csv("/kaggle/input/west-african-ebola-virus-epidemic-timeline/cases_and_deaths.csv", delimiter=',')

df1.dataframeName = 'cases_and_deaths.csv'



# df -> list

x_values = df1['Days'].values.tolist() 

y_values = df1['Case'].values.tolist()
# fitting functions

def f(t, K, P0, r):

    return  (K / (1 + ((K-P0)/P0)*np.exp(-r*t)))



# fitting

popt, pcov = curve_fit(f, x_values, y_values, p0=[1, 1, 0.5], maxfev=300000)

print(f"Fitting parameters")

print(f"K: {popt[0]}, P0: {popt[1]}, r: {popt[2]}")
# init main graph

fig = pl.figure(figsize=(16, 9))

ax = pl.axes()



# main graph captions

pl.suptitle("2014 Ebola epidemic in West Africa ", fontweight="bold")

pl.ylabel('Cases')

pl.xlabel('Days')



# main fitting plot

xx = np.linspace(0, x_values[-1], 100)

yy = f(xx, popt[0], popt[1], popt[2])

pl.xlim(x_values[0], x_values[-1])

pl.ylim(y_values[0], y_values[-1])



pl.plot(x_values, y_values,'o', label='Cases')

pl.plot(xx, yy, label="Logistic Function")

pl.legend(loc='lower right')



# Any results you write to the current directory are saved as output.

pl.savefig("graph.png")
# init main graph

fig = pl.figure(figsize=(16, 9))

ax = pl.axes()



# main graph captions

pl.suptitle("2014 Ebola epidemic in West Africa (log scale)", fontweight="bold")

pl.ylabel('Cases')

pl.xlabel('Days')



pl.yscale('Log')

pl.locator_params(axis='x',tight=True, nbins=5)

pl.plot(x_values, y_values,'o', label='Cases')

pl.plot(xx, yy, label="Logistic Function")

pl.legend(loc='lower right')



# Any results you write to the current directory are saved as output.

pl.savefig("graph_log.png")