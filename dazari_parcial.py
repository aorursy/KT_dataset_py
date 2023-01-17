# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import decomposition

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, precision_score

from  sklearn.feature_selection import  SelectKBest, f_classif

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
datos = pd.read_csv("../input/RR_HH.csv")

Y = datos["Renuncia"]

datos.drop(["Renuncia","IdEmpleado","Renuncia_Disc","UltimaEvaluacion_Disc","SatisfaccionLaboral_Disc","AccidentesTrabajo_Disc","Ascendido_Disc","ProyectosRealizados_Disc","HorasMensuales_Disc","Antiguedad_Disc"], axis=1, inplace=True)
datos.head()
correlacion = datos.corr()

sns.heatmap(correlacion, 

        xticklabels=correlacion.columns,)
#scatterplot

sns.set()

cols = ['UltimaEvaluacion', 'ProyectosRealizados', 'HorasMensuales', 'Antiguedad',"NivelSatisfaccion"]

sns.pairplot(datos[cols], size = 2.5)

plt.show();
sns.distplot(datos['NivelSatisfaccion']);
sns.distplot(datos['UltimaEvaluacion']);
sns.distplot(datos['ProyectosRealizados']);
one_hot_salario = pd.get_dummies(datos['NivelSalarial'])

one_hot_area = pd.get_dummies(datos['AreaTrabajo'])

datos.drop(['NivelSalarial','AreaTrabajo'],axis = 1, inplace=True)
datos = datos.join(one_hot_salario)

datos = datos.join(one_hot_area)
datos.head()
datos = preprocessing.StandardScaler().fit_transform(datos)
datos = pd.DataFrame(datos)
datos.head()
datos  = SelectKBest(f_classif, k=10).fit_transform(datos, Y)
modelo = RandomForestClassifier(n_estimators=5)

modelo.fit(datos, Y) 

y_pred = modelo.predict(datos)

precision = precision_score(Y, y_pred)

print(precision)
tn, fp, fn, tp = confusion_matrix(y_pred, Y).ravel()
tn, fp, fn, tp