# Importación de las librerías de análisis y carga del archivo

import numpy as np

import pandas as pd

import seaborn as sns

import scipy as stats

import matplotlib.pyplot as plt

import matplotlib as matplot

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

filename = '../input/HR_comma_sep.csv'

df = pd.read_csv(filename)
print(df)
# Número de filas

print(len(df))
#Columnas

print(df.columns)

len(df.columns)



#Informacion de las columnas

df.info()
# Ya que hay dos columnas categóricas, observamos los valores que pueden tomar.

print(df.sales.unique())

print(df.salary.unique())
#Encabezados

df.head()
# Comprobar si hay algún valor vacío en la tabla.

df.isnull().any()
# Poner un nombre más apropiado a la columna 'sales':

df = df.rename(columns={'sales': 'Department'})
# Convertimos las columnas "salary" y "Department" en categorías.

df.salary = df.salary.astype('category')

df.Department = df.Department.astype('category')

df.info()
# Estadísticos descriptivos de los datos

df.describe()
# Cantidad de personas que abandonaron la empresa y porcentaje sobre el total de empleados

print(df.left.value_counts())

print(int(df[df['left']==1].size)/df.size *100, '% de empleados abandonaron la empresa')
#Comparación de medias entre empleados que abandonaron la empresa y los que siguen

df.groupby('left').mean()
#Matriz de correlación

corr = df.corr()

corr
#Mapa de calor

sns.heatmap(corr)

plt.show()
# Vamos a comprobar como afecta la banda salarial

by_salary = df.groupby('salary').mean()

print(df['salary'].value_counts())

plt.figure(figsize=(6, 6))

sns.countplot(data=df, x="salary", hue="left")

by_salary
sns.barplot(data=df, x="salary", y="left")
# Comprobamos las diferencias entre departamentos

by_department = df.groupby('Department').mean()

print(df['Department'].value_counts())

plt.figure(figsize=(8, 8))

sns.countplot(data=df, x="Department", hue="left")

plt.xticks(rotation= -45)

by_department
plt.xticks(rotation= -45)

sns.barplot(data=df, x="Department", y="left")
# Primera columna: distribución de los empleados en función de su nivel de satisfacción, última evaluación y media de horas mensuales.

# Segunda columna: Comparación entre los que abandonan la empresa y los que no.

f, ax = plt.subplots(3, 2, figsize=(30,15))

sns.distplot(df["satisfaction_level"], ax = ax[0,0])

sns.kdeplot(df.loc[(df['left'] == 1),'satisfaction_level'], shade=True, label='turnover', ax = ax[0,1])

sns.kdeplot(df.loc[(df['left'] == 0),'satisfaction_level'], shade=True, label='no turnover', ax = ax[0,1])

sns.distplot(df["last_evaluation"], color ='green', ax = ax[1,0])

sns.kdeplot(df.loc[(df['left'] == 1),'last_evaluation'], shade=True, label='turnover', ax = ax[1,1])

sns.kdeplot(df.loc[(df['left'] == 0),'last_evaluation'], shade=True, label='no turnover', ax = ax[1,1])

sns.distplot(df["average_montly_hours"], color ='red', ax = ax[2,0])

sns.kdeplot(df.loc[(df['left'] == 1),'average_montly_hours'], shade=True, label='turnover', ax = ax[2,1])

sns.kdeplot(df.loc[(df['left'] == 0),'average_montly_hours'], shade=True, label='no turnover', ax = ax[2,1])