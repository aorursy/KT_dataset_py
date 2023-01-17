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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

from os import listdir 

ozone = pd.read_csv('/kaggle/input/ozone.txt', sep=";", decimal=',')

#print(os.getcwd())

#print(os.listdir('/kaggle/input/ozone.txt'))
ozone.head()
df = ozone.plot()
#df=pyplot.scatter(T9, T12, c = 'red')
sns.set()

ax = sns.scatterplot(x="T12", y="maxO3", data=ozone)

ax.set(xlabel='T12', ylabel='MaxO3')

ax.xaxis.set_major_locator(plt.MaxNLocator(5))

#sns.lmplot(x="T12", y="maxO3", data=ozone);

reg_simp = smf.ols('maxO3 ~ T12', data=ozone).fit()
print(reg_simp.summary())
ax = sns.lmplot(x="T12", y="maxO3", data=ozone, ci=None, line_kws={'color':'black'})

ax.set(xlabel='T12', ylabel='MaxO3')

plt.show()
print(reg_simp.summary())
ozone['maxO3_ajust_s'] = reg_simp.predict() 





X_plot = [ozone['maxO3'].min(), ozone['maxO3'].max()]



ax = sns.scatterplot(x="maxO3", y="maxO3_ajust_s", data=ozone)

ax.set(xlabel='MaxO3', ylabel='MaxO3 ajusté')

plt.plot(X_plot, X_plot, color='r')

plt.show()


ozone['residu_s'] = reg_simp.resid
plt.hist(ozone['residu_s'], density=True)

plt.xlabel('Résidus')

plt.title('Histogramme des résidus')

plt.show()




a_prevoir = pd.DataFrame({'T12':[15]})

maxO3_prev = reg_simp.predict(a_prevoir)

print(round(maxO3_prev[0], 2))



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

from scipy.stats import t, shapiro

from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels






ozone = pd.read_csv('/kaggle/input/ozone.txt', sep=";", decimal=',')

#print(os.getcwd())

#print(os.listdir('/kaggle/input/ozone.txt'))



reg_multi = smf.ols('maxO3~T9+T12+T15+Ne9+Ne12+Ne15+maxO3v', data=ozone).fit()

print(reg_multi.summary())
reg_multi = smf.ols('maxO3~T9+T12+T15+Ne9+Ne12+maxO3v', data=ozone).fit()

print(reg_multi.summary())
reg_multi = smf.ols('maxO3~T9+T12+T15+Ne9+maxO3v', data=ozone).fit()

print(reg_multi.summary())
reg_multi = smf.ols('maxO3~T12+T15+Ne9+maxO3v', data=ozone).fit()

print(reg_multi.summary())
reg_multi = smf.ols('maxO3~T12+Ne9+maxO3v', data=ozone).fit()

print(reg_multi.summary())
a_prevoir = pd.DataFrame({'T12': 26, 'Ne9': 2, 'maxO3v': 70}, index=[0])

maxO3_prev = reg_multi.predict(a_prevoir)

print(round(maxO3_prev[0], 2))
alpha = 0.05
n = ozone.shape[0]

p = 4
analyses = pd.DataFrame({'obs':np.arange(1, n+1)})

analyses['levier'] = reg_multi.get_influence().hat_matrix_diag



seuil_levier = 2*p/n
plt.figure(figsize=(10,6))

plt.bar(analyses['obs'], analyses['levier'])

plt.xticks(np.arange(0, 115, step=5))

plt.xlabel('Observation')

plt.ylabel('Leviers')

plt.plot([0, 115], [seuil_levier, seuil_levier], color='r')

plt.show()
influence = reg_multi.get_influence().summary_frame()
analyses['dcooks'] = influence['cooks_d']

seuil_dcook = 4/(n-p)
plt.figure(figsize=(10,6))

plt.bar(analyses['obs'], analyses['dcooks'])

plt.xticks(np.arange(0, 115, step=5))

plt.xlabel('Observation')

plt.ylabel('Leviers')

plt.plot([0, 115], [seuil_dcook, seuil_dcook], color='r')

plt.show()
variables = reg_multi.model.exog

[variance_inflation_factor(variables, i) for i in np.arange(1,variables.shape[1])]