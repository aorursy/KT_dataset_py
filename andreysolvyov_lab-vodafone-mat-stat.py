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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib


import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
Target = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')['pH']
Target
d = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
abs(d[['residual sugar', 'pH']].corr().iloc[1][0])
t = Target
varr = pd.DataFrame()
varr['2.74-2.86'] = [len(t[(2.74 <= t) & (t <= 2.86)])]
varr['2.87-2.93'] = [len(t[(2.87 <= t) & (t <= 2.93)])]
varr['2.94-3.08'] = [len(t[(2.94 <= t) & (t <= 3.08)])]
varr['3.09-3.19'] = [len(t[(3.09 <= t) & (t <= 3.19)])]
varr['3.2-3.3'] = [len(t[(3.2 <= t) & (t <= 3.3)])]
varr['3.31-3.42'] = [len(t[(3.31 <= t) & (t <= 3.42)])]
varr['3.43-3.56'] = [len(t[(3.43 <= t) & (t <= 3.56)])]
varr['3.57-3.7'] = [len(t[(3.57 <= t) & (t <= 3.7)])]
varr['3.71-3.8'] = [len(t[(3.71 <= t) & (t <= 3.8)])]
varr['3.81-3.89'] = [len(t[(3.81 <= t) & (t <= 3.89)])]
varr
plt.figure(figsize = (25, 5))
plt.bar(height = varr.iloc[0], x = varr.columns);
plt.figure(figsize = (15, 5))
plt.plot(varr.iloc[0]);
n = len(Target)
Srednee = sum(Target)/n
Srednee
Sigma = (sum((Target - Srednee)**2)/n)**(1/2)
Sigma
Assimetria = sum((Target - Srednee)**3)/(n*Sigma**3)
Assimetria
Ekscess = sum((Target - Srednee)**4)/(len(Target)*Sigma**4)-3
Ekscess
Dispersia_assim = 6*(n-1)/((n+1)*(n+3))
print(Dispersia_assim)
Dispersia_eks = 24*n*(n-2)*(n-3)/((n+3)*(n+5)*(n+1)**2)
print(Dispersia_eks)
print('Асимметрия меньше либо равна трех корней из диспресии асимметрии - это: ', abs(Assimetria)<=3*Dispersia_assim**(1/2))
print('Эксцесс меньше либо равен пяти корней из диспресии ексцесса - это: ', abs(Ekscess)<=5*Dispersia_eks**(1/2))
S = ((Sigma**2)*n/(n+1))**(1/2)
S
print(Srednee - (S/(n**(1/2)))*0.81524, ' ; ', (Srednee + (S/(n**(1/2)))*0.81524))
print(S*(1-0.111),' ; ', S*(1+0.111))
t = varr.T.rename(columns = {0:'N_i'})
t['X_i'] = [round((float(list(varr.columns)[i].split('-')[0]) + float(list(varr.columns)[i].split('-')[1]))/2, 3) for i in range(len(varr.columns))]
t['X_i*N_i'] = t['N_i']*t['X_i']
t['(X_i^2)*N_i'] = t['N_i']*t['X_i']*t['X_i']
t
import math
t = varr.T.rename(columns = {0:'N_i'})
t['X_i'] = [round((float(list(varr.columns)[i].split('-')[0]) + float(list(varr.columns)[i].split('-')[1]))/2, 3) for i in range(len(varr.columns))]
t['Z_i'] = (t['X_i'] - Srednee)/S
t['f(Z_i)'] = (math.e**(-(t['Z_i']**2)/2))/((2*math.pi)**(1/2))
Ni = []
for i in range(len(t)):
    Ni.append(t['f(Z_i)'][i]*(sum(t['N_i']))*1/S*(float(t.index[0].split('-')[1]) - float(t.index[0].split('-')[0])))
t["N_i'"] = Ni
t
t['Хи^2'] = ((t['N_i'] - t["N_i'"])**2)/t["N_i'"]
t
sum(t['Хи^2'])
t  = d['residual sugar'][(d['residual sugar'] < 5) & (1.25 < d['residual sugar'])]
varr = pd.DataFrame({str(round(min(t) + (max(t) - min(t))/20*(i), 2)) + '-' + str(round(min(t) + (max(t) - min(t))/20*(i+1), 2)) : [len(t[(min(t)+(max(t) - min(t))/20*(i) <= t) & (t <= min(t) + (max(t) - min(t))/20*(i+1))])] for i in range(20)})
varr = varr[varr.columns[:10]]
varr
varr = pd.DataFrame()
varr['1.3-1.55'] = [len(t[(1.3 <= t) & (t <= 1.55)])]
varr['1.56-1.7'] = [len(t[(1.56 <= t) & (t <= 1.7)])]
varr['1.71-1.89'] = [len(t[(1.71 <= t) & (t <= 1.89)])]
varr['1.9-2.08'] = [len(t[(1.9 <= t) & (t <= 2.08)])]
varr['2.09-2.21'] = [len(t[(2.09 <= t) & (t <= 2.21)])]
varr['2.22-2.4'] = [len(t[(2.22 <= t) & (t <= 2.4)])]
varr['2.41-2.64'] = [len(t[(2.41 <= t) & (t <= 2.64)])]
varr['2.65-3.11'] = [len(t[(2.65 <= t) & (t <= 3.11)])]
varr
Target  = d['residual sugar'][(d['residual sugar'] < 5) & (1.25 < d['residual sugar'])]
plt.figure(figsize = (25, 5))
plt.bar(height = varr.iloc[0], x = varr.columns);
plt.figure(figsize = (15, 5))
plt.plot(varr.iloc[0]);
n = len(Target)
Srednee = sum(Target)/n
Srednee
Sigma = (sum((Target - Srednee)**2)/n)**(1/2)
Sigma
Assimetria = sum((Target - Srednee)**3)/(n*Sigma**3)
Assimetria
Ekscess = sum((Target - Srednee)**4)/(len(Target)*Sigma**4)-3
Ekscess
Dispersia_assim = 6*(n-1)/((n+1)*(n+3))
print(Dispersia_assim)
Dispersia_eks = 24*n*(n-2)*(n-3)/((n+3)*(n+5)*(n+1)**2)
print(Dispersia_eks)
print('Асимметрия меньше либо равна трех корней из диспресии асимметрии - это: ', abs(Assimetria)<=3*Dispersia_assim**(1/2))
print('Эксцесс меньше либо равен пяти корней из диспресии ексцесса - это: ', abs(Ekscess)<=5*Dispersia_eks**(1/2))
S = ((Sigma**2)*n/(n+1))**(1/2)
S
print(Srednee - (S/(n**(1/2)))*0.81524, ' ; ', (Srednee + (S/(n**(1/2)))*0.81524))
print(S*(1-0.111),' ; ', S*(1+0.111))
t = varr.T.rename(columns = {0:'N_i'})
t['X_i'] = [round((float(list(varr.columns)[i].split('-')[0]) + float(list(varr.columns)[i].split('-')[1]))/2, 3) for i in range(len(varr.columns))]
t['X_i*N_i'] = t['N_i']*t['X_i']
t['(X_i^2)*N_i'] = t['N_i']*t['X_i']*t['X_i']
t
t = varr.T.rename(columns = {0:'N_i'})
t['X_i'] = [round((float(list(varr.columns)[i].split('-')[0]) + float(list(varr.columns)[i].split('-')[1]))/2, 3) for i in range(len(varr.columns))]
t['Z_i'] = (t['X_i'] - Srednee)/S
t['f(Z_i)'] = (math.e**(-(t['Z_i']**2)/2))/((2*math.pi)**(1/2))
Ni = []
for i in range(len(t)):
    Ni.append(t['f(Z_i)'][i]*(sum(t['N_i']))*1/S*(float(t.index[0].split('-')[1]) - float(t.index[0].split('-')[0])))
t["N_i'"] = Ni
t
t['Хи^2'] = ((t['N_i'] - t["N_i'"])**2)/t["N_i'"]
t
sum(t['Хи^2'])
r = abs(d[['fixed acidity', 'citric acid']].corr().iloc[1][0])
r
sig_r = (1-r**2)/10
sig_r
from sklearn.linear_model import LinearRegression
c = LinearRegression().fit(d[['fixed acidity']], d['citric acid']).coef_
i = LinearRegression().fit(d[['fixed acidity']], d['citric acid']).intercept_
print('a*' + str(c[0])+str(i))