import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt #trabalhar com datas 

import matplotlib.pyplot as plt # graficos

import seaborn as sn



#Modelo

from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.ensemble import RandomForestClassifier



base = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv') 

base.head()
base.info()

base.describe()
#Valores nulos: valores nulos no salary indicam que a pessoa em questão não foi contratada, logo essas linhas não podem ser removidas 

#base.isnull().sum()   

#base.dropna(inplace = True)
#Dados categóricos 

base.nunique()

base.head()

base['status'].value_counts()    # muitas observações a mais na class 1 em comparação a classe 0 

#base['status'].value_counts()['Placed']

#utilizando o dicionário e a função "replace" para transformar em numericos

mapeamento_classes = {"workex": {"No": 0, "Yes": 1},

                "status": {"Placed": 1,"Not Placed": 0 }} 



base.replace(mapeamento_classes, inplace=True)

#base['status'] = base['status'].astype('category')

#base['workex'] = base['workex'].astype('category')





plotar = base.groupby(['status','workex'])['status'].count().unstack('workex')#.fillna(0)

plotar[[0, 1]].plot(kind='bar', stacked=True) 

plt.title('Relação entre experiência (workex) de trabalho e ser contratado (status) ') 

plt.xlabel('Status') 

plt.ylabel('Frequência')

#Preparando dados 

x = base['workex'].values

x_reshaped = x.reshape(-1,1)

y = base['status'].values



# Separando dados em treino e  teste 

x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y, random_state = 1)
x_train
#Random Forest 

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)
#previsão 

rfc_pred = rfc.predict(x_test) 



#metricas 

print(classification_report(y_test, rfc_pred))
#Matrix de confusão 

cm = confusion_matrix(y_test, rfc_pred) #gera a matriz de confusão

df_cm = pd.DataFrame(cm, index = [i for i in "01"],columns = [i for i in "01"]) #cria o df com as classes

plt.figure(figsize = (10,7)) #indica o tamanho da figura 

sn.heatmap(df_cm, annot=True) #plota a figura