# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import spatial
from sklearn.metrics import recall_score
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import LeaveOneOut
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
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1,random_state=42)
neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn=neigh.fit(x_train, y_train['Class'])
y_pred=knn.predict(x_test)
acc=accuracy_score(y_test,y_pred)*100
print("Taxa de acerto do holdout completamente aleatório com 90% dos dados para treino e o restante para teste=",acc)
def holdout_aleatorio_com_repeticao(n,frac,x,y):
    
    dict_taxa={}
    lista_acerto=[]
    r=40
    for k in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=frac,random_state=r)
        neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
        knn=neigh.fit(x_train, y_train['Class'])
        y_pred=knn.predict(x_test)
        acc=accuracy_score(y_test,y_pred)*100
        lista_acerto.append(acc)
        r+=2
    print("Média=",pd.DataFrame(lista_acerto)[0].mean())
    
    return lista_acerto
lista_acerto=holdout_aleatorio_com_repeticao(10,0.1,x,y)
lista_acerto


def holdout_estraticicado(n,frac,x,y):
    
    dict_taxa={}
    lista_acerto=[]
    r=40
    for k in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y,test_size=frac,random_state=r)
        neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
        knn=neigh.fit(x_train, y_train['Class'])
        y_pred=knn.predict(x_test)
        acc=accuracy_score(y_test,y_pred)*100
        lista_acerto.append(acc)
        r+=2
    print("Média=",pd.DataFrame(lista_acerto)[0].mean())
    
    return lista_acerto
lista_acerto=holdout_estraticicado(10,0.1,x,y)
lista_acerto

def k_fold_cross_validation(num_split,nun_repeat,x,y):
    
    kf=RepeatedStratifiedKFold(n_splits=num_split, n_repeats=nun_repeat,random_state=0)
    kf.get_n_splits(x,y)
    lista_acerto=[]
    
    for train_index, test_index in kf.split(x,y):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
        knn=neigh.fit(x_train, y_train['Class'])
        y_pred=knn.predict(x_test)
        acc=accuracy_score(y_test,y_pred)*100
        lista_acerto.append(acc)
        
    print("Média=",pd.DataFrame(lista_acerto)[0].mean())
    print("Soma das taxas de acertos=",pd.DataFrame(lista_acerto)[0].sum()/10)
    return lista_acerto
              
k_fold_cross_validation(10,1,x,y)
def Leave_one_out(n,frac,x,y):
     
    lista_acerto=[]
    loo = LeaveOneOut()
    loo.get_n_splits(x,y)
    for train_index, test_index in loo.split(x,y):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
        knn=neigh.fit(x_train, y_train['Class'])
        y_pred=knn.predict(x_test)
        acc=accuracy_score(y_test,y_pred)*100
        lista_acerto.append(acc)
    
    return lista_acerto
lista_acerto=Leave_one_out(10,0.1,x,y)
print("Média da taxa de acerto do leave one out:",pd.DataFrame(lista_acerto)[0].mean())
x=data_wine.iloc[:,1:13]
y=data_wine.iloc[:,:1]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1,random_state=42)
neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn=neigh.fit(x_train, y_train['Class'])
y_pred=knn.predict(x_test)
acc=accuracy_score(y_test,y_pred)*100
print("Taxa de acerto do holdout completamente aleatório com 90% dos dados para treino e o restante para teste=",acc)
lista_acerto=holdout_aleatorio_com_repeticao(10,0.1,x,y)
lista_acerto
lista_acerto=holdout_estraticicado(10,0.1,x,y)
lista_acerto
k_fold_cross_validation(10,1,x,y)
lista_acerto=Leave_one_out(10,0.1,x,y)
print("Média da taxa de acerto do leave one out:",pd.DataFrame(lista_acerto)[0].mean())