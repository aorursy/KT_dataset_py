#All Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb
SuicideByState = pd.read_csv("../input/unscaleddata/SBS.csv") #Data before scaled values

SuicideByState .head(3)
conv = pd.read_csv("../input/statedata/SBS.csv") #Data after I scaled values

conv.head(3)
colormap = plt.cm.RdBu #heatmap made to show correlation of variables

plt.figure(figsize = (14,12))

plt.title('Correlation Suicide', y=1.2, size = 15)

sb.heatmap(conv.corr(),linewidths=0.1,vmax=1.0, 

square=True, cmap=colormap, linecolor='white', annot=True)
sb.pairplot(conv)
sb.pairplot(conv, vars= ["SuicideRate", "lackOfFirearmReg"])
Polynomial = np.polynomial.Polynomial

X  = conv["SuicideRate"]

Y = conv["lackOfFirearmReg"]

pfit, stats = Polynomial.fit(X, Y, 1, full=True)



plt.plot(X,Y, 'o')

plt.title("How lax gun laws impact suicide rate")

plt.xlabel("Suicide Rate")

plt.ylabel("Lack of Gun Regulation")

plt.plot(X, pfit(X))

np.corrcoef(X, Y)[0, 1]
X  = conv["SuicideRate"]

Y = conv["PopDensity "]

pfit, stats = Polynomial.fit(X, Y, 1, full=True)



plt.plot(X,Y, 'o')

plt.title("S vs PD with outliers")

plt.xlabel("Suicide Rate")

plt.ylabel("Population Density")

plt.plot(X, pfit(X))

plt.ylim(-.1,1)

np.corrcoef(X, Y)[0, 1]
Q1= Y.quantile(.25)

Q3 = Y.quantile(.75)

IQR = Q3-Q1

filter = (Y >= Q1 - 1.5 * IQR) & (Y <= Q3 + 1.5 *IQR)

X = X.loc[filter]

Y = Y.loc[filter]

Y.size
pfit, stats = Polynomial.fit(X, Y, 1, full=True)

plt.plot(X,Y, 'o')

plt.title("S vs PD removal of outliers")

plt.xlabel("Suicide Rate")

plt.ylabel("Population Density")

plt.plot(X, pfit(X))

plt.ylim(-.1,1)

np.corrcoef(X, Y)[0, 1]
#Maybe It follows more of an exponential curve (*note* I switched the x and y axis, looked better with fit)

X  = conv["PopDensity "]

Y = conv["SuicideRate"]

pfit = np.poly1d(np.polyfit(X, Y, 3)) #degree 3



plt.plot(X,Y, 'o')

plt.title("SuicideRate vs Population Density")

plt.xlabel("Population Density")

plt.ylabel("SuicideRate")

t = np.linspace(0, .9, 80)

plt.plot(X, Y, 'o', t, pfit(t), '-')
X  = conv["SuicideHotlineCalls "]

Y = conv["Population"]

pfit, stats = Polynomial.fit(X, Y, 1, full=True)



plt.plot(X,Y, 'o')

plt.title("Population vs Suicide Hotline Calls")

plt.xlabel("Suicide Hotline Calls")

plt.ylabel("Population")

plt.plot(X, pfit(X))

np.corrcoef(X, Y)[0, 1]
X  = conv["CrisisCenters"]

Y = conv["Population"]

pfit, stats = Polynomial.fit(X, Y, 1, full=True)



plt.plot(X,Y, 'o')

plt.title("Population vs Crisis Centers")

plt.xlabel("Crisis Centers")

plt.ylabel("Population")

plt.plot(X, pfit(X))

np.corrcoef(X, Y)[0, 1]
#Thanks For Checking This Out!