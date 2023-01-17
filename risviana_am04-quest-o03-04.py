# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import spatial
%matplotlib inline
import  statistics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importando dataset
data_wine= pd.read_csv("/kaggle/input/wine-daset/wineDATA.CSV")
x=data_wine.iloc[:,1:]
y=data_wine.iloc[:,:1]
def comparar_classificadores(num,x,y,kk):
    lista=[]
    r=40
    for k in range(num):
        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.5,random_state=r)
        neigh = KNeighborsClassifier(n_neighbors=kk,metric='euclidean')
        knn=neigh.fit(x_train, y_train['Class'])
        y_pred=knn.predict(x_test)
        acc=accuracy_score(y_test,y_pred)*100
        r+=2
        lista.append(acc)
    return lista
#lista=comparar_classificadores(30,x,y)

def houldout(num,x,y):
    #data=pd.DataFrame(index=range(15))
    frame=[]
    for i in range(1,num):
        lista=comparar_classificadores(30,x,y,i)
        frame.append(lista)
    return pd.DataFrame(frame)
dt=houldout(16,x,y)
dt=dt.T
#Calculando intervalo de confiança
#calculando a média de acerto
def calcular_intervalo_confiança(data):
    lista_conf=[]
    for k in range(len(data.columns)):
        media_acerto=statistics.mean(data[k].values)
        #calculando o desvio padrão
        desvio_padrao=statistics.stdev(data[k].values)
        #intervalo de confiança
        lista_conf.append(["%.2f" %(media_acerto-1.96*desvio_padrao),"%.2f" %(media_acerto+1.96*desvio_padrao)])
    return lista_conf
calcular_intervalo_confiança(dt)
#importando dataset
data_wine= pd.read_csv("/kaggle/input/wine-daset/wineDATA.CSV")
x=data_wine.iloc[:,1:13]
y=data_wine.iloc[:,:1]
dt=houldout(16,x,y)
dt=dt.T
calcular_intervalo_confiança(dt)
