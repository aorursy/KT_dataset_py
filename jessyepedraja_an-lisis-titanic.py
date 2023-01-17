# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train_titanic = pd.read_csv("../input/titanic/train.csv")
df_test_titanic = pd.read_csv("../input/titanic/test.csv")
df_train_titanic
df_train_titanic.isnull().sum()
edad=df_train_titanic["Age"].mean()
df_train_titanic= df_train_titanic.fillna(edad)
df_train_titanic
moda_cabin=df_train_titanic["Cabin"].mode()
df_train_titanic= df_train_titanic.fillna(moda_cabin)
df_train_titanic
df_train_titanic.isnull().sum()
df_train_titanic.info()
df_train_titanic[["Name","Sex","Ticket"]] = df_train_titanic[["Name","Sex","Ticket"]].astype("category")
df_train_titanic.dtypes
df_solo_hombres=df_train_titanic[df_train_titanic["Sex"]=="male"]
df_solo_hombres
pasajeros_clase_1_2= df_train_titanic[df_train_titanic["Pclass"].between(1,2)]
pasajeros_clase_1_2
#pasajeros clase 1 y 3 menores de 30 años
menores_30= df_train_titanic[(df_train_titanic["Pclass"].isin([1,3])) & (df_train_titanic["Age"]<30)]
menores_30
# pasajeros entre 20 y 30 años 
pasajeros_20_30= df_train_titanic[(df_train_titanic["Age"].isin([20,30]))]
pasajeros_20_30
# pasajeros menores de 25 años y mayores a 50 años
pasajeros_menores_mayores= df_train_titanic[~df_train_titanic["Age"].between(25,50)]
pasajeros_menores_mayores
# remplazar male, female por M, F 
df_train_titanic["Sex"] = df_train_titanic["Sex"].replace("male","M")
df_train_titanic["Sex"] = df_train_titanic["Sex"].replace("female","F")
df_train_titanic
# Fare menor a 33 respecto a la columna Pclass igual a 2
comparativo= df_train_titanic[(df_train_titanic["Fare"]<33) & (df_train_titanic["Pclass"]==2)].head()
comparativo
# costo total de los tickets de por las 3 clases, y el numero de pasajeros de cada clase
print("Clase 1:\nCosto total: ",df_train_titanic[df_train_titanic["Pclass"]==1]["Fare"].sum(), "No. Pasajeros",len(df_train_titanic[df_train_titanic["Pclass"]==1]["Fare"]))
print("Clase 2:\nCosto total: ",df_train_titanic[df_train_titanic["Pclass"]==2]["Fare"].sum(), "No. Pasajeros",len(df_train_titanic[df_train_titanic["Pclass"]==2]["Fare"]))
print("Clase 3:\nCosto total: ",df_train_titanic[df_train_titanic["Pclass"]==3]["Fare"].sum(), "No. Pasajeros",len(df_train_titanic[df_train_titanic["Pclass"]==3]["Fare"]))
#1000 pasajeros que no sobrevivieron mas jovenes y mujeres 
df_train_titanic[df_train_titanic["Survived"]==0].sort_values("Age").head(1000)["Sex"].value_counts()
# mostrando a 5 pasajeros que sean hombres y 5 que sena mujeres, sobrevivientes para cada clase
listdf=[]
for i in ["male","female"]:
    for j in [1,2,3]:
        new_df = df_train_titanic[(df_train_titanic["Sex"]==i) & (df_train_titanic["Survived"]==1) & (df_train_titanic["Pclass"]==j)]
        listdf.append(new_df)
pd.concat(listdf,axis=0)

# randgo de edad niño, joven, adulto, mayor
df_train_titanic.loc[df_train_titanic['Age'] < 11, 'Rango edad'] ='Niño'
df_train_titanic.loc[df_train_titanic['Age'].between(11,17), 'Rango edad'] = 'Joven'
df_train_titanic.loc[df_train_titanic['Age'].between(18, 49), 'Rango edad'] = 'Adulto'
df_train_titanic.loc[df_train_titanic['Age']>=50, 'Rango edad']='Mayor'
df_train_titanic
# pasajeros jovenes y mayores de la clase 2 y 3 sobrevivientes
df_train_titanic[(df_train_titanic["Rango edad"].isin(["Joven","Mayor"])) & df_train_titanic["Pclass"].isin([2,3]) & (df_train_titanic["Survived"]==1)]
# agrupando por sobrevivientes por sexo
sobrevive_F= df_train_titanic[(df_train_titanic["Sex"]=="F") & (df_train_titanic["Survived"]==1)]
sobrevive_F
sobrevive_M= df_train_titanic[(df_train_titanic["Sex"]=="M") & (df_train_titanic["Survived"]==1)]
sobrevive_M
