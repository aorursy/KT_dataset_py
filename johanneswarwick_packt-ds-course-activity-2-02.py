# Gráficos
%matplotlib inline
import matplotlib as mpl
import seaborn as sns
mpl.pyplot.style.use('seaborn')
# Manipulação de Dados
import pandas as pd
import numpy as np
# Estatística
from statsmodels.graphics.correlation import plot_corr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
# Machine Learning
from sklearn.model_selection import train_test_split
rawBostonData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')
rawBostonData = rawBostonData.dropna()
rawBostonData = rawBostonData.drop_duplicates()
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
X = renamedBostonData.drop('crimeRatePerCapita', axis = 1)
y = renamedBostonData[['crimeRatePerCapita']]
seed = 10
test_data_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state = seed)
train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)
corrMatrix = train_data.corr(method = 'pearson')

corrMatrix_2 = corrMatrix.mask(np.logical_and(corrMatrix > -0.4, corrMatrix < 0.4) ,0)
for i in range(len(corrMatrix_2)):
  corrMatrix_2.iat[i,i] = 0

plot_corr(corrMatrix_2, xnames=list(train_data.columns), ynames=list(train_data.columns), \
title='Correlação Para |r| > 0.4', normcolor=False, cmap='RdYlBu_r')
logmultiLinearModel = smf.ols(formula=\
'np.log(crimeRatePerCapita) ~ (pctLowerStatus + radialHighwaysAccess + medianValue_Ks + nitrixOxide_pp10m)**2', data=train_data)

logmultiLinearModelResult = logmultiLinearModel.fit()
print(logmultiLinearModelResult.summary())
fig_1 = mpl.pyplot.figure(figsize=(14,36))
fig_1 = sm.graphics.plot_partregress_grid(logmultiLinearModelResult, fig=fig_1)
fig_2, ax = mpl.pyplot.subplots(figsize=(17,6))
sns.regplot(x='medianValue_Ks', y=np.log(train_data['crimeRatePerCapita']), ci=95, data=train_data, ax=ax, color='k', scatter_kws={"s":22, "color":"royalblue", "alpha":0.8})
ax.set_ylabel(r"$Crime~rate~per~Capita~_{[log]}$", fontsize=15, fontname='DejaVu Sans', multialignment='center')
ax.set_xlabel(r"$Median~value~of~owner{\_}occupied~homes~in~{\$}1000's$", fontsize=15, fontname='DejaVu Sans', multialignment='center')
ax.set_xlim(left=None, right=None)
ax.set_ylim(bottom=None, top=None)
ax.tick_params(axis='both', which='major', labelsize=12)
fig_2.tight_layout()
fig_3, ax = mpl.pyplot.subplots(figsize=(20,8))
sm.graphics.plot_fit(logmultiLinearModelResult, "medianValue_Ks", ax=ax)
ax.set_ylabel(r"$Crime~rate~per~Capita~_{[log]}$", fontsize=15, fontname='DejaVu Sans', multialignment='center')
ax.set_xlabel(r"$Median~value~of~owner{\_}occupied~homes~in~{\$}1000's$", fontsize=15, fontname='DejaVu Sans', multialignment='center')
fig_3.suptitle('Valores Reais vs. Valores do Modelo', fontsize=15, fontname='DejaVu Sans', multialignment='center')
ax.set_xlim(left=None, right=None)
ax.set_ylim(bottom=None, top=None)
ax.tick_params(axis='both', which='major', labelsize=14)
fig_3.tight_layout()
#fig_3, ax = mpl.pyplot.subplots(figsize=(20,8))
#fig_3 = sm.graphics.plot_fit(logmultiLinearModelResult, "medianValue_Ks", ax=ax)