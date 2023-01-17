import pandas as pd

import numpy as np

from dateutil import parser

import matplotlib.pyplot as plt

import seaborn as sns

import random

import csv
!ls
dataTransactions = pd.read_csv('transactions.csv')

dataHolidays = pd.read_csv('holidays_events.csv')

#dataTrain = pd.read_csv('train.csv')

dataTrain = pd.read_csv('test.csv')

dataTest = pd.read_csv('test.csv')

dataStores = pd.read_csv('stores.csv')

dataItems = pd.read_csv('items.csv')
#dataTrain['unit_sales'] = int(random.randint(1,6))

dataTest['unit_sales'] = 0

dataTransactions['year'] = dataTransactions['date'].apply(lambda row: int(row.split('-')[0]))

dataTransactions['month'] = dataTransactions['date'].apply(lambda row: int(row.split('-')[1]))
dataTransactions.head()
transactionsYear = pd.DataFrame(

    {'transactions' : dataTransactions.groupby( ['year'] )['transactions'].sum()}

).reset_index()
transactionsYear
sns.set_style("darkgrid")

ax = sns.factorplot(x="year", y="transactions",data=transactionsYear, size=5, aspect=1.5, kind="bar")

plt.show()
transactionsMont = pd.DataFrame(

    {'transactions' : dataTransactions.groupby( ['year', 'month'] )['transactions'].sum()}

).reset_index()

transactionsMont['period'] = range(1, len(transactionsMont) +1)
# numero de periodos o meses disponibles para el calculo de ña regresion 

transactionsMont['period'].max()
#regresion lineal de ventas segun el numero de periodos

g = sns.lmplot(x="period", y="transactions", data=transactionsMont, size=7)

plt.show()
#regresion lineal de ventas segun por anio

g = sns.lmplot(x="period", y="transactions", data=transactionsMont, size=7, hue="year")

plt.show()
# join de los datos

dataTrainState = dataTransactions.join(dataStores.set_index('store_nbr'), on='store_nbr').reset_index()

dataTrainState.head() 
dataTrainStateCount = pd.DataFrame(

    {'transactions' : dataTrainState.groupby( ['year', 'state'] )['transactions'].sum()}

).reset_index()

dataTrainStateCount.head()
g = sns.factorplot(x="year", y="transactions", col = 'state', data=dataTrainStateCount, kind="bar", size=2.7, col_wrap=3);

plt.show()
g = sns.factorplot(x="year", y="transactions", hue='state', data=dataTrainStateCount, size=6);

plt.show()
g = sns.lmplot(x="month", y="transactions", data=dataTrainState, size=7, hue="state")

plt.show()
dataTrainStorade = pd.DataFrame(

    {'transactions' : dataTransactions.groupby( ['store_nbr'] )['transactions'].sum()}

).reset_index()

dataTrainStorade = dataTrainStorade.sort_values(['transactions'], ascending=[False])

dataTrainStorade.head()
g = sns.factorplot(x="transactions", y="store_nbr", data=dataTrainStorade, size=10,  kind="bar", orient="h", order=dataTrainStorade.store_nbr)

plt.show()
# join de los datos

dataTrainHoliday = dataTransactions.join(dataHolidays.set_index('date'), on='date').reset_index()

dataTrainHoliday = pd.DataFrame(

    {'transactions' : dataTrainHoliday.groupby( ['description'])['transactions'].sum()}

).reset_index()

dataTrainHoliday['transactions'].min()
g = sns.factorplot(x="transactions", y="description", data=dataTrainHoliday, kind="bar", size=20);

plt.show()
dataItemFamily = dataTrain.join(dataItems.set_index('item_nbr'), on='item_nbr').reset_index()

dataItemFamily.head()
dataItemFamily['id'].max()
dataItemFamilyStore = pd.DataFrame(

    {'transactions' : dataItemFamily.groupby( ['family', 'store_nbr']).size()}

).reset_index()

dataItemFamilyStore = dataItemFamilyStore.join(dataStores.set_index('store_nbr'), on='store_nbr').reset_index()

dataItemFamilyStore.head()
len(dataItemFamilyStore)
### se relaciona tanto la tienda y su influencia y transacionabilidad de los productos
g = sns.factorplot(x="transactions", y="family", col="store_nbr", data=dataItemFamilyStore, kind="bar", size=5, col_wrap=3);

plt.show()
dataItemFamilyCity = pd.DataFrame(

    {'transactions' : dataItemFamilyStore.groupby( ['state', 'family'])['transactions'].sum()}

).reset_index()

dataItemFamilyCity.head()
g = sns.factorplot(x="transactions", y="state", col="family", data=dataItemFamilyCity, kind="bar", size=5, col_wrap=3);

plt.show()
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

dataTrain = pd.read_csv('../input/train.csv', dtype=dtypes)

dataTest = pd.read_csv('../input/test.csv', dtype=dtypes)

dataTest['unit_sales'] = 0

#dataTrain['unit_sales'] = int(random.randint(1,6))
# graficos embebidos

%matplotlib inline

dataTrain['day'] = dataTrain['date'].apply(lambda row: int(row.split('-')[2]))

dataTrain['date'] = dataTrain['date'].apply(lambda row: int(row.split('-')[1]))

dataTrain['onpromotion'] = dataTrain['onpromotion'].apply(lambda row: 1 if row == True else 0)

dataTrain.head()
feature_cols = ['date', 'store_nbr', 'item_nbr', 'day']
from sklearn.linear_model import LinearRegression



rl = LinearRegression() # Creando el modelo.

rl.fit(np.array(dataTrain[feature_cols]), np.array(dataTrain['unit_sales'])) # ajustando el modelo x, y
dataTest['onpromotion'] = dataTest['onpromotion'].apply(lambda row: 1 if row == True else 0)

dataTest['day'] = dataTest['date'].apply(lambda row: int(row.split('-')[2]))

dataTest['date'] = dataTest['date'].apply(lambda row: int(row.split('-')[1]))
predicciones = rl.predict(np.array(dataTest[feature_cols]))

predicciones_df = pd.DataFrame({'id': dataTest['id'], 'unit_sales': predicciones.round()})

predicciones_df.head() # predicciones de las primeras 5 lineas
print(rl.intercept_)

print(rl.coef_)
# filas requeridas 3370464

len(dataTrain['day'])
predicciones_df.to_csv('predicciones.csv', index=False, float_format='%.2f')