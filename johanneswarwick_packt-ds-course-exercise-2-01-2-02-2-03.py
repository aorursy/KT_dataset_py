%matplotlib inline
# It is an IPython magic function that renders the figure in a notebook

import matplotlib as mpl
# Matplotlib is a package for creating static, animated, and interactive data visualizations

import seaborn as sns
# Seaborn is a Python data visualization package, based on matplotlib
# It provides a high-level interface for drawing attractive and informative statistical graphics

import matplotlib.pyplot as plt
# Gives an unfamiliar reader a hint that PYPLOT IS A MODULE, rather than a function

import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from statsmodels.graphics.correlation import plot_corr
# StatsModels is a Python package for the estimation of many statistical models,
# as well as for conducting statistical tests, and statistical data exploration

import pandas as pd
# Pandas is an package providing data structures and data analysis tools

import numpy as np
# NumPy is a package for scientific computing with Python
# It contains among other things a powerful N-dimensional array object,
# useful linear algebra, Fourier transform, and random number capabilities

import patsy
# Patsy is a package for describing statistical models (especially linear models
# or models that have a linear component) and building design matrices

from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets

plt.style.use('seaborn')
# Customizing Matplotlib with style sheets
rawBostonData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')
rawBostonData.head()
rawBostonData = rawBostonData.dropna()
rawBostonData = rawBostonData.drop_duplicates()
list(rawBostonData.columns)
renamedBostonData = rawBostonData.rename(columns = {'CRIM':'crimeRatePerCapita',
' ZN ':'landOver25K_sqft',
'INDUS ':'non-retailLandProptn',
'CHAS':'riverDummy',
'NOX':'nitrixOxide_pp10m',
'RM':'AvgNo.RoomsPerDwelling',
'AGE':'ProptnOwnerOccupied',
'DIS':'weightedDist',
'RAD':'radialHighwaysAccess',
'TAX':'propTaxRate_per10K',
'PTRATIO':'pupilTeacherRatio',
'LSTAT':'pctLowerStatus',
'MEDV':'medianValue_Ks'})

renamedBostonData.head()
renamedBostonData.info()
renamedBostonData.describe(include=[np.number]).T
X = renamedBostonData.drop('crimeRatePerCapita', axis = 1)
y = renamedBostonData[['crimeRatePerCapita']]
X.head()
y.head()
seed = 10
test_data_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state = seed)
train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)
train_data.info()
test_data.info()
corrMatrix = train_data.corr(method = 'pearson')

corrMatrix.head()
corrMatrix_2 = corrMatrix.mask(np.logical_and(corrMatrix > -0.6, corrMatrix < 0.6) ,0)
for i in range(len(corrMatrix_2)):
  corrMatrix_2.iat[i,i] = 0

corrMatrix_2.head()
xnames=list(train_data.columns)
ynames=list(train_data.columns)

plot_corr(corrMatrix_2, xnames=xnames, ynames=ynames, title='Matriz de Correlação', normcolor=False, cmap='RdYlBu_r')
fig_1, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x='medianValue_Ks', y='crimeRatePerCapita', ci=None, data=train_data, ax=ax, color='k', scatter_kws={"s":22, "color":"royalblue", "alpha":0.8})
ax.set_ylabel('Crime rate per Capita', fontsize=15, fontname='DejaVu Sans')
ax.set_xlabel("Median value of owner-occupied homes in $1000's", fontsize=15, fontname='DejaVu Sans')
ax.set_xlim(left=4, right=51)
ax.set_ylim(bottom=-5, top=20)
ax.tick_params(axis='both', which='major', labelsize=12)
fig_1.tight_layout()
fig_2, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x='medianValue_Ks', y=np.log(train_data['crimeRatePerCapita']), ci=90, data=train_data, ax=ax, color='k', scatter_kws={"s":22, "color":"royalblue", "alpha":0.8})
ax.set_ylabel(r"$Crime~rate~per~Capita~_{[log]}$", fontsize=15, fontname='DejaVu Sans', multialignment='center')
ax.set_xlabel(r"$Median~value~of~owner{\_}occupied~homes~in~{\$}1000's$", fontsize=15, fontname='DejaVu Sans', multialignment='center')
ax.set_xlim(left=None, right=None)
ax.set_ylim(bottom=None, top=None)
ax.tick_params(axis='both', which='major', labelsize=12)
fig_2.tight_layout()