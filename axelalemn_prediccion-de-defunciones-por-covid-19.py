import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import imblearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import BorderlineSMOTE



covid_mexico = pd.read_csv('/kaggle/input/mexico-covid19-clinical-data/mexico_covid19.csv')

covid_mexico.head()
covid_mexico.shape
columnas_conservar = ['RESULTADO', 'DELAY', 'SEXO', 'TIPO_PACIENTE', 'FECHA_DEF', 'INTUBADO', 'NEUMONIA', 'EDAD',

                     'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',

                     'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'OTRO_CASO', 'UCI']

covid_mexico_limpio = covid_mexico[columnas_conservar]

covid_mexico_limpio.dropna()

covid_mexico_limpio.shape
covid_mexico_limpio.head(10)
covid_mexico_limpio.loc[:,'RESULTADO']= covid_mexico_limpio['RESULTADO'].replace(2, 0)

covid_mexico_limpio.loc[:,'DEFUNCION']= covid_mexico_limpio['FECHA_DEF'] != '9999-99-99'

covid_mexico_limpio.loc[:,'DEFUNCION'] = covid_mexico_limpio['DEFUNCION'].map({True: 1, False: 0})

covid_mexico_limpio
covid_mexico_limpio.info()
resultados = covid_mexico_limpio['RESULTADO'].value_counts()

resultados.plot(kind = 'bar')
defunciones_en_positivos = covid_mexico_limpio[covid_mexico_limpio['RESULTADO'] == 1]['DEFUNCION'].value_counts()

defunciones_en_positivos.plot(kind = 'bar')
columnas_conservar = ['RESULTADO', 'DELAY', 'SEXO', 'TIPO_PACIENTE', 'DEFUNCION', 'INTUBADO', 'NEUMONIA', 'EDAD',

                     'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',

                     'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'OTRO_CASO', 'UCI']

covid_mexico_raw = covid_mexico_limpio[columnas_conservar]

covid_mexico_raw = covid_mexico_raw[covid_mexico_raw['RESULTADO'] == 1]

covid_mexico_raw = covid_mexico_raw[['DELAY', 'SEXO', 'TIPO_PACIENTE', 'DEFUNCION', 'INTUBADO', 'NEUMONIA', 'EDAD',

                     'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',

                     'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'OTRO_CASO', 'UCI']]



X = covid_mexico_raw.drop('DEFUNCION',axis=1)

y = covid_mexico_raw.DEFUNCION



scaler = MinMaxScaler()



X[['EDAD']] = scaler.fit_transform(X[['EDAD']])



oversample = BorderlineSMOTE()

X, y = oversample.fit_resample(X, y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
n_neighbors = 5

 

knn = KNeighborsClassifier(n_neighbors)

knn.fit(X_train, y_train)



prediccionKNN = knn.predict(X_test)

print(confusion_matrix(y_test, prediccionKNN))

print(classification_report(y_test, prediccionKNN))



fpr, tpr, _ = roc_curve(y_test,  prediccionKNN)

auc = roc_auc_score(y_test, prediccionKNN)

plt.plot(fpr,tpr,label='KNN K=5 Precision: '+str(auc))

plt.legend(loc=4)

plt.show()