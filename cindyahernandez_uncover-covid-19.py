import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy.linalg as la

from  zipfile import ZipFile

import warnings

warnings.filterwarnings("ignore")
import os

print(os.listdir("../input/uncoverzip"))
paises_a = ['Colombia', 'Peru', 'Sweden', 'South Africa', 'Chile', 'Panama', 'United States of America', 'Ecuador', 'Spain', 'Italy', 'Argentina', 'Bolivia', 'Canada', 'Costa Rica', 'United Kingdom']

paises_b = ['Colombia', 'Peru', 'Sweden', 'South Africa', 'Chile', 'Panama', 'United States', 'Ecuador', 'Spain', 'Italy', 'Argentina', 'Bolivia', 'Canada', 'Costa Rica', 'United Kingdom']
#Numero de casos

our_world_in_data_stad = pd.read_csv('../input/uncoverzip/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research (1).csv')

our_world_in_data_stad = our_world_in_data_stad.loc[our_world_in_data_stad['location'].isin(paises_b)]

our_world_in_data_stad = our_world_in_data_stad.drop(['tests_units'],1)

our_world_in_data_stad.date = pd.to_datetime(our_world_in_data_stad.date)



our_world_in_data_stad.head()

primeros_casos = []

datos_pais = pd.DataFrame()

for i in paises_b:

  datos_pais = our_world_in_data_stad.loc[our_world_in_data_stad['location'].isin([i])]

  info = datos_pais.loc[datos_pais['total_cases_per_million']>10]

  min = info.min()

  primeros_casos.append(min['date'])

casos = {'country':paises_b,'date50':primeros_casos}



casos_50 = pd.DataFrame (casos, columns = ['country','date50'])

casos_50.date50 = pd.to_datetime(casos_50.date50) 

casos_50.head(20)
# Medidas implementadas

HDE_1 = pd.read_csv('../input/uncoverzip/HDE/acaps-covid-19-government-measures-dataset.csv')



HDE_1 = HDE_1.loc[HDE_1['country'].isin(paises_a)]

sub_grupo =  pd.get_dummies(HDE_1['measure'])

columnas_HDE_1 = sub_grupo.columns

sub_grupo = pd.concat([sub_grupo, HDE_1.date_implemented.reindex(sub_grupo.index)], axis=1)

sub_grupo = pd.concat([sub_grupo, HDE_1.country.reindex(sub_grupo.index)], axis=1)

sub_grupo = pd.concat([sub_grupo, HDE_1.iso.reindex(sub_grupo.index)], axis=1)

sub_grupo = sub_grupo.dropna()

sub_grupo = sub_grupo.sort_values(by=['country','date_implemented'])

sub_HDE_1 = pd.DataFrame()



for i in paises_a:

    fechas = sub_grupo.loc[sub_grupo['country'].isin([i])].date_implemented.unique()



    for j in range(len(fechas)):

        a = sub_grupo.loc[sub_grupo['country'].isin([i])].loc[ sub_grupo['date_implemented'].isin([fechas[j]]) ]



        if np.shape(a)[0] >0 :

            b = np.sum(a.iloc[:,:-3])

            b = b.to_frame().T

            b['country'] = i

            b['date'] = fechas[j]

            b['iso_code'] = a.iso.iloc[0]

            sub_HDE_1 = sub_HDE_1.append(b, ignore_index=True)

                

            

columnas_HDE_1 = sub_grupo.columns



sub_HDE_1.date = pd.to_datetime(sub_HDE_1.date)

sub_HDE_1 = sub_HDE_1.drop(['country'],1)



sub_HDE_1.head()
#Movilidad

google_mobility = pd.read_csv('../input/uncoverzip/google_mobility/regional-mobility.csv')

google_mobility = google_mobility.loc[google_mobility['country'].isin(paises_b)]

google_mobility = google_mobility.loc[google_mobility['region'].isin(['Total'])]

google_mobility = google_mobility.dropna()

google_mobility = google_mobility.drop(['region'],1)

google_mobility.date = pd.to_datetime(google_mobility.date)



google_mobility.head()
our_world_in_data_stad = our_world_in_data_stad.merge(sub_HDE_1, on=['date', 'iso_code'],how='left')

our_world_in_data_stad[columnas_HDE_1[:-3]] = our_world_in_data_stad[columnas_HDE_1[:-3]].fillna(0)

our_world_in_data_stad = our_world_in_data_stad.sort_values(by=['location','date'])

datos = pd.DataFrame()
for i in paises_b:

    paises = our_world_in_data_stad.loc[our_world_in_data_stad['location'].isin([i])]

    fechas = paises.date.unique()

    for j in range(len(fechas)):

        if j > 0 :

            c = np.logical_or(paises.iloc[j-1,15:],paises.iloc[j,15:])

            paises.iloc[j,15:]= c   

    datos = datos.append(paises, ignore_index = True)

 
datos = datos.rename(columns={'location': 'country'})

datos = datos.merge(google_mobility, on=['date', 'country'],how='left')

datos = datos.merge(casos_50, on=[ 'country'],how='left')
datos['days since cases per millon are over ten'] = (datos['date'] -   datos['date50']).dt.days

datos.new_cases_per_million.loc[datos['new_cases']==0] = 0

datos.new_deaths_per_million.loc[datos['new_deaths']==0]=0

datos = datos.drop(['date','date50','total_cases','total_deaths','new_cases', 'new_deaths', 'total_tests',	'new_tests',	'total_tests_per_thousand','new_tests_per_thousand'],1)

datos = datos.dropna()

datos.head(10)
plt.figure(figsize=(15,5))



plt.subplot(121)

cases = sns.lineplot(x='days since cases per millon are over ten', y="new_cases_per_million", hue="country",data=datos)

#cases.set_yscale("log")

plt.subplot(122)

deaths = sns.lineplot(x='days since cases per millon are over ten', y="new_deaths_per_million", hue="country", data=datos)

#deaths.set_yscale('log')
HDE_1.date_implemented = pd.to_datetime(HDE_1.date_implemented)

HDE_1 = HDE_1.replace({'country': {'United States of America': 'United States'}})

HDE_1 = HDE_1.merge(casos_50, on=[ 'country'],how='left')

HDE_1['days since cases per millon are over ten'] = (HDE_1['date_implemented'] -   HDE_1['date50']).dt.days
sns.catplot(x='days since cases per millon are over ten', y="measure", hue="country", kind="swarm", data=HDE_1, height=10, aspect=1)
col = google_mobility.columns

col = col[2:]

plt.figure(figsize=(15,20))

for i in range(len(col)):

  plt.subplot(3,2,i+1)

  sns.lineplot(x='days since cases per millon are over ten', y=col[i], hue="country",data=datos)

paises_b.remove('Colombia')
X_train = datos.loc[datos['country'].isin(paises_b)]

X_train = X_train.loc[X_train['days since cases per millon are over ten']>0]

Y_train_cases = X_train.new_cases_per_million.to_numpy()

Y_train_deaths = X_train.new_deaths_per_million.to_numpy()

X_train = X_train.drop(['iso_code',	'country',	'total_cases_per_million',	'new_cases_per_million',	'total_deaths_per_million',	'new_deaths_per_million'],axis = 1)

predictor = X_train.columns

X_train =X_train.to_numpy()

X_test = datos.loc[datos['country'].isin(['Colombia'])]

X_test = X_test.loc[X_test['days since cases per millon are over ten']>0]

Y_test_cases = X_test.new_cases_per_million.to_numpy()

Y_test_deaths = X_test.new_deaths_per_million.to_numpy()

X_test = X_test.drop(['iso_code',	'country',	'total_cases_per_million',	'new_cases_per_million',	'total_deaths_per_million',	'new_deaths_per_million'],axis = 1).to_numpy()
n_trees = np.logspace(0.5,3,10).astype(int)

av_cases = np.zeros([np.shape(predictor)[0], 10])

av_deaths = np.zeros([np.shape(predictor)[0], 10])

error_cases = []

error_deaths = []

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



for i in range(10):



  regressor_cases = RandomForestRegressor(n_estimators = n_trees[i], random_state = 0)

  regressor_deaths = RandomForestRegressor(n_estimators = n_trees[i], random_state = 0)



  regressor_cases.fit(X_train, Y_train_cases)

  regressor_deaths.fit(X_train, Y_train_deaths)



  Y_predict_cases = regressor_cases.predict(X_test)

  Y_predict_deaths = regressor_deaths.predict(X_test)



  av_cases[:,i] =  regressor_cases.feature_importances_

  av_deaths[:,i] = regressor_deaths.feature_importances_



  error_deaths.append(metrics.mean_squared_error((Y_test_deaths), (Y_predict_deaths)))

  error_cases.append(metrics.mean_squared_error((Y_test_cases), (Y_predict_cases)))

  plt.figure(figsize=(15,5))



  plt.subplot(121)

  plt.plot(X_test[:,-1], (Y_test_cases), label='Real data')

  plt.plot(X_test[:,-1], (Y_predict_cases), label = 'Predicted data')

  plt.xlabel('days since cases per millon are over ten')

  plt.ylabel('new cases per million')

  plt.legend()

  plt.title('n. trees: %i. Mean Squared Error: %f '%(n_trees[i], metrics.mean_squared_error((Y_test_cases), (Y_predict_cases))))

  plt.subplot(122)

  plt.plot(X_test[:,-1], (Y_test_deaths), label='Real data')

  plt.plot(X_test[:,-1], (Y_predict_deaths), label = 'Predicted data')

  plt.xlabel('total since cases per millon are over ten')

  plt.ylabel('new deaths per million')

  plt.legend()

  plt.title('Mean Squared Error %f '%(metrics.mean_squared_error((Y_test_deaths), (Y_predict_deaths))))

plt.figure(figsize= (15,5))

plt.subplot(121)

plt.plot(n_trees, error_cases, 'r',label = 'Cases')

plt.legend()

plt.xlabel('number of trees')

plt.ylabel('mean squared error')

plt.subplot(122)

plt.plot(n_trees, error_deaths, 'g',label = 'Deaths')

plt.legend()

plt.xlabel('number of trees')

plt.ylabel('mean squared error')
plt.figure(figsize = (10,2))



a = pd.Series(np.mean(av_cases, axis = 1), index=predictor)

a.nlargest().plot(kind='barh')

plt.xlabel('Feature Importance')



plt.figure(figsize = (10,2))



a = pd.Series(np.mean(av_deaths, axis = 1), index=predictor)

a.nlargest().plot(kind='barh')

plt.xlabel('Feature Importance')