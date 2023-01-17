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
import pandas as pd

import numpy as np

import seaborn as sns

import plotly.express as px

import matplotlib

import time

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')

import statsmodels.api as sm

matplotlib.rcParams['axes.labelsize'] = 20

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
from scipy import stats

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 

from sklearn.model_selection import GridSearchCV
filepath = '../input/walmart-recruiting-store-sales-forecasting/'
dados = pd.read_csv(f'{filepath}train.csv.zip', parse_dates=['Date'], compression='zip')

dados.info()
lojas = pd.read_csv(f'{filepath}stores.csv')

lojas.info()
features = pd.read_csv(f'{filepath}features.csv.zip', parse_dates=['Date'], compression='zip')

features.info()
predizer = pd.read_csv(f'{filepath}test.csv.zip', parse_dates=['Date'], compression='zip')

predizer.info()
plt.figure(figsize=(10,8))

sns.distplot(dados.Weekly_Sales)
plt.figure(figsize=(10,8))

sns.boxplot(x="Type", y="Size", data=lojas)
valores_por_loja_tamanho = dados.merge(lojas, on='Store', how='left')

lojas_completas = valores_por_loja_tamanho.merge(features, on=['Store','Date'], how='left')

lojas_completas.info()
plt.figure(figsize=(20,6))

sns.boxplot(x="Store", y="Weekly_Sales",hue='IsHoliday_x', data=lojas_completas)
vendas_no_tempo = lojas_completas.groupby(['Date','Type']).mean()['Weekly_Sales'].reset_index()
plt.figure(figsize=(30,8))

sns.lineplot(x="Date", y="Weekly_Sales",hue='Type', data=vendas_no_tempo)
vendas_agrupadas = lojas_completas.groupby(['Store','Date','Type','IsHoliday_x']).sum()['Weekly_Sales'].reset_index()
plt.figure(figsize=(20,6))

sns.boxplot(x="Store", y="Weekly_Sales",hue='IsHoliday_x', data=vendas_agrupadas)
feriados = {

    'Date' : (pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',

                             '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06',

                            '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29',

                            '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'])),

    'TypeHoliday' :(['SB', 'SB', 'SB', 'SB',

                    'Labor', 'Labor', 'Labor', 'Labor',

                    'Thanksgiving', 'Thanksgiving', 'Thanksgiving', 'Thanksgiving',

                    'Christmas', 'Christmas', 'Christmas', 'Christmas']) 



}
lojas_feriados = lojas_completas.merge(pd.DataFrame(feriados), on='Date', how='left')

lojas_feriados['TypeHoliday'].fillna(0,inplace=True)

lojas_feriados
lojas_feriados['Month'] = pd.to_datetime(lojas_feriados.Date).dt.month

lojas_feriados['Year'] = pd.to_datetime(lojas_feriados.Date).dt.year
plot_feriados = lojas_feriados.groupby(['Store','Year','TypeHoliday','Type']).sum()['Weekly_Sales'].reset_index()

plt.figure(figsize=(15,8))

sns.boxplot(x="Year", y="Weekly_Sales",hue='TypeHoliday', data=plot_feriados.loc[plot_feriados['TypeHoliday']!=0])
plt.figure(figsize=(15,8))

sns.boxplot(x="Type", y="Weekly_Sales",hue='TypeHoliday', data=plot_feriados.loc[plot_feriados['TypeHoliday']!=0])
plt.figure(figsize=(30,8))

sns.boxplot(x="Store", y="MarkDown1",hue='IsHoliday_x', data=lojas_feriados)
plt.figure(figsize=(30,8))

sns.boxplot(x="Store", y="MarkDown2",hue='IsHoliday_x', data=lojas_feriados)
plt.figure(figsize=(30,8))

sns.boxplot(x="Store", y="MarkDown3",hue='IsHoliday_x', data=lojas_feriados)
plt.figure(figsize=(30,8))

sns.boxplot(x="Store", y="MarkDown4",hue='IsHoliday_x', data=lojas_feriados)
plt.figure(figsize=(30,8))

sns.boxplot(x="Store", y="MarkDown5",hue='IsHoliday_x', data=lojas_feriados)
filling = lojas_feriados.groupby(['Store','IsHoliday_x']).median()[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].reset_index()

filling.rename(columns={'MarkDown1':'FMD1','MarkDown2':'FMD2','MarkDown3':'FMD3','MarkDown4':'FMD4','MarkDown5':'FMD5'}, inplace=True)
lojas_preenchidas = lojas_feriados.merge(filling, on=['Store','IsHoliday_x'], how='inner')
lojas_preenchidas.MarkDown1.fillna(lojas_preenchidas['FMD1'],inplace=True)

lojas_preenchidas.MarkDown2.fillna(lojas_preenchidas['FMD2'],inplace=True)

lojas_preenchidas.MarkDown3.fillna(lojas_preenchidas['FMD3'],inplace=True)

lojas_preenchidas.MarkDown4.fillna(lojas_preenchidas['FMD4'],inplace=True)

lojas_preenchidas.MarkDown5.fillna(lojas_preenchidas['FMD5'],inplace=True)

lojas_preenchidas.drop(['FMD1','FMD2','FMD3','FMD4','FMD5'], axis=1, inplace=True)

lojas_preenchidas.info()
lojas_preenchidas['IsHoliday'] = pd.get_dummies(lojas_preenchidas.IsHoliday_x)[1]

lojas_preenchidas['SB'] =pd.get_dummies(lojas_preenchidas.TypeHoliday)['SB']

lojas_preenchidas['Labor'] =pd.get_dummies(lojas_preenchidas.TypeHoliday)['Labor']

lojas_preenchidas['Thanksgiving'] =pd.get_dummies(lojas_preenchidas.TypeHoliday)['Thanksgiving']

lojas_preenchidas['Christmas'] =pd.get_dummies(lojas_preenchidas.TypeHoliday)['Christmas']

lojas_preenchidas['TypeA'] = pd.get_dummies(lojas_preenchidas.Type)['A']

lojas_preenchidas['TypeB'] = pd.get_dummies(lojas_preenchidas.Type)['B']

lojas_preenchidas['TypeC'] = pd.get_dummies(lojas_preenchidas.Type)['C']

lojas_preenchidas.head()
plt.figure(figsize=(25,20))

sns.heatmap(lojas_preenchidas.fillna(0).corr(), annot=True)
plt.figure(figsize=(20,10))

lojas_preenchidas['Weekly_Sales_Size'] = lojas_preenchidas['Weekly_Sales']/lojas_preenchidas['Size']

sns.lineplot(x="Date", y="Weekly_Sales_Size",hue='Type', data=lojas_preenchidas)
lojas_limpas = lojas_preenchidas.drop(['MarkDown4','MarkDown5','IsHoliday_y','IsHoliday_x','Type','TypeHoliday','TypeC','Weekly_Sales_Size'], axis=1)
OfT = lojas_limpas.loc[lojas_limpas['Date'] >= '2012-10-05']

lojas_predict = lojas_limpas.loc[lojas_limpas['Date'] < '2012-10-05']
# Conjunto OfT

X_OfT = OfT.drop(['Weekly_Sales','Date'], axis=1)

y_OfT = OfT['Weekly_Sales']

# Conjunto OfS

X = lojas_predict.drop(['Weekly_Sales','Date'], axis=1)

y = lojas_predict['Weekly_Sales']
X_OfT.head()
def WMAE(y_obs, y_pred, flag):

    peso = flag*5+1*(1-flag)

    

    indice = (1/sum(peso))*sum(peso*abs(y_obs-y_pred))



    return indice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(f'Formato do conjunto de treino OfS: {X_train.shape}')

print(f'Formato do conjunto de teste: {X_test.shape}')
parameters = {'normalize': [False,True], 'fit_intercept': [False,True]} 

model = LinearRegression()



SEED = 1988

np.random.seed(SEED)

cv = KFold(10, shuffle=True)







clf = GridSearchCV(model, parameters, cv=cv, verbose=5, n_jobs=8)

clf.fit(X_train, y_train)

clf.best_params_

clf.predict(X_test)

wmae = WMAE(y_test, clf.predict(X_test), X_test.IsHoliday)

r2 = r2_score(y_test, clf.predict(X_test))

results = cross_validate(model, X, y, cv=cv, return_train_score = False)

media = results['test_score'].mean()

desvio = results['test_score'].std()



print("LinearRegression")

print("------------------------------")

print(f'Parametros ótimos = {clf.best_params_}')

print(f'Mean: {media*100}')

print(f'Accuracy: [{(media-2*desvio)*100} , {(media+2*desvio)*100}]')

print(f'WMAE = {wmae} and R-square = {r2}')

print("------------------------------")
SEED = 1988

np.random.seed(SEED)

ln = LinearRegression(fit_intercept = True, normalize = False)

rf = RandomForestRegressor(max_depth = 30, max_features =  18, min_samples_leaf = 2 ,n_estimators = 200)

gbm = GradientBoostingRegressor(max_depth = 5, max_features = 12, min_samples_leaf = 5, n_estimators = 200)

np.random.seed(SEED)



ln.fit(X_train, y_train)

ln_y_pred = ln.predict(X_test)

wmae_ln_pred = WMAE(y_test, ln_y_pred, X_test.IsHoliday)

r2_ln_pred = r2_score(y_test, ln_y_pred)



rf.fit(X_train, y_train)

rf_y_pred = rf.predict(X_test)

wmae_rf_pred = WMAE(y_test, rf_y_pred, X_test.IsHoliday)

r2_rf_pred = r2_score(y_test, rf_y_pred)



gbm.fit(X_train, y_train)

gbm_y_pred = gbm.predict(X_test)

wmae_gbm_pred = WMAE(y_test, gbm_y_pred, X_test.IsHoliday)

r2_gbm_pred = r2_score(y_test, gbm_y_pred)



print("LinearRegression")

print("------------------------------")

print(f'WMAE = {wmae_ln_pred} and R-square = {r2_ln_pred}')

print("------------------------------")

print("RandomForest")

print("------------------------------")

print(f'WMAE = {wmae_rf_pred} and R-square = {r2_rf_pred}')

print("------------------------------")

print("GradientBoosting")

print("------------------------------")

print(f'WMAE = {wmae_gbm_pred} and R-square = {r2_gbm_pred}')

print("------------------------------")
plt.figure(figsize = (20,12))

plt.scatter(y_test,ln_y_pred,label='LR',marker = 'o',color='r')

plt.scatter(y_test,rf_y_pred,label='RF',marker = 'o',color='b')

plt.scatter(y_test,gbm_y_pred,label='GBR',marker = 'o',color='y')

plt.title('Modelos',fontsize = 25)

plt.legend(fontsize = 20)

plt.show()
ln_y_OfT = ln.predict(X_OfT)

wmae_ln_OfT = WMAE(y_OfT, ln_y_OfT, X_OfT.IsHoliday)

r2_ln_OfT = r2_score(y_OfT, ln_y_OfT)





rf_y_OfT = rf.predict(X_OfT)

wmae_rf_OfT = WMAE(y_OfT, rf_y_OfT, X_OfT.IsHoliday)

r2_rf_OfT = r2_score(y_OfT, rf_y_OfT)





gbm_y_OfT = gbm.predict(X_OfT)

wmae_gbm_OfT = WMAE(y_OfT, gbm_y_OfT, X_OfT.IsHoliday)

r2_gbm_OfT = r2_score(y_OfT, gbm_y_OfT)



print("LinearRegression")

print("------------------------------")

print(f'WMAE = {wmae_ln_OfT} and R-square = {r2_ln_OfT}')

print("------------------------------")

print("RandomForest")

print("------------------------------")

print(f'WMAE = {wmae_rf_OfT} and R-square = {r2_rf_OfT}')

print("------------------------------")

print("GradientBoosting")

print("------------------------------")

print(f'WMAE = {wmae_gbm_OfT} and R-square = {r2_gbm_OfT}')

print("------------------------------")
plt.figure(figsize = (20,12))

plt.scatter(y_OfT,ln_y_OfT,label='LR',marker = 'o',color='r')

plt.scatter(y_OfT,rf_y_OfT,label='RF',marker = 'o',color='b')

plt.scatter(y_OfT,gbm_y_OfT,label='GBR',marker = 'o',color='y')

plt.title('Modelos',fontsize = 25)

plt.legend(fontsize = 20)

plt.show()
predicoes_loja = predizer.merge(features, on=['Store','Date'], how='inner')
lojas_p_features = predicoes_loja.merge(lojas, on='Store', how='inner')
filling_p = lojas_p_features.groupby(['Store','IsHoliday_x','Type']).median()[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']].reset_index()

filling_p.rename(columns={'MarkDown1':'FMD1','MarkDown2':'FMD2','MarkDown3':'FMD3','MarkDown4':'FMD4','MarkDown5':'FMD5','CPI': 'C','Unemployment':'UN'}, inplace=True)
lojas_preenchidas_p = lojas_p_features.merge(filling_p, on=['Store','IsHoliday_x','Type'], how='inner')
lojas_preenchidas_p.MarkDown1.fillna(lojas_preenchidas_p['FMD1'],inplace=True)

lojas_preenchidas_p.MarkDown2.fillna(lojas_preenchidas_p['FMD2'],inplace=True)

lojas_preenchidas_p.MarkDown3.fillna(lojas_preenchidas_p['FMD3'],inplace=True)

lojas_preenchidas_p.MarkDown4.fillna(lojas_preenchidas_p['FMD4'],inplace=True)

lojas_preenchidas_p.MarkDown5.fillna(lojas_preenchidas_p['FMD5'],inplace=True)

lojas_preenchidas_p.CPI.fillna(lojas_preenchidas_p['C'],inplace=True)

lojas_preenchidas_p.Unemployment.fillna(lojas_preenchidas_p['UN'],inplace=True)

lojas_preenchidas_p.drop(['FMD1','FMD2','FMD3','FMD4','FMD5','C','UN'], axis=1, inplace=True)

lojas_preenchidas_p.info()
lojas_feriados_p = lojas_preenchidas_p.merge(pd.DataFrame(feriados), on='Date', how='left')

lojas_feriados_p['TypeHoliday'].fillna(0,inplace=True)

lojas_feriados_p
lojas_feriados_p['IsHoliday'] = pd.get_dummies(lojas_feriados_p.IsHoliday_x)[1]

lojas_feriados_p['SB'] = pd.get_dummies(lojas_feriados_p.TypeHoliday)['SB']

lojas_feriados_p['Labor'] = 0

lojas_feriados_p['Thanksgiving'] = pd.get_dummies(lojas_feriados_p.TypeHoliday)['Thanksgiving']

lojas_feriados_p['Christmas'] = pd.get_dummies(lojas_feriados_p.TypeHoliday)['Christmas']



lojas_feriados_p
lojas_feriados_p['TypeA'] = pd.get_dummies(lojas_feriados_p.Type)['A']

lojas_feriados_p['TypeB'] = pd.get_dummies(lojas_feriados_p.Type)['B']

lojas_feriados_p['TypeC'] = pd.get_dummies(lojas_feriados_p.Type)['C']

lojas_feriados_p['Month'] = pd.to_datetime(lojas_feriados_p.Date).dt.month

lojas_feriados_p['Year'] =  pd.to_datetime(lojas_feriados_p.Date).dt.year
colunas = X_train.columns
lojas_predizer = lojas_feriados_p[colunas]

lojas_predizer.fillna(0, inplace=True)
lojas_preenchidas_p['Weekly_Sales'] = rf.predict(lojas_predizer).round(2)
predicoes_finais = lojas_preenchidas_p[['Store','Dept','Date','Weekly_Sales']]
predicoes_finais['Id'] = (predicoes_finais[['Store','Dept','Date']].astype('str').apply('_'.join, axis=1))
sample_submission = (f'{filepath}sampleSubmission.csv.zip')
submission = predicoes_finais[['Id','Weekly_Sales']]
submission.info()
submission.to_csv('submission.csv', index=False)