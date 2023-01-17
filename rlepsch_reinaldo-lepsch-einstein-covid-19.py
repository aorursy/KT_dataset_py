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
ds_red = pd.read_excel('/kaggle/input/newfile/dataset_modificado.xlsx', index_col=0)
ds_red
ds_red.shape
ds_red.describe
import numpy as np

from sklearn.tree import DecisionTreeClassifier

def dividir_treino_teste(data, perc_treinamento):

  np.random.seed(0)

  #np.random.shuffle(data)

  divisao = int(data.shape[0] * perc_treinamento)

  conj_treinamento = data[:divisao]

  conj_teste = data[divisao:]

  

  return conj_treinamento,conj_teste
def gerar_atributos_labels(data):

  

  labels = data['SARS-Cov-2 exam result']



  atributos = data.iloc[:,5:]



  atributos= preprocessing.StandardScaler().fit(atributos).transform(atributos)



  return atributos, labels
def decision_tree_previsto_real(data):

    

  # dividir o conjunto de dados em treinamento (70%) e teste

  pac_treinamento, pac_teste = dividir_treino_teste(data, 0.7)



  # gerar os conjuntos de atributos e labels para treinamento e testes

  atributos_treinamento, labels_treinamento = gerar_atributos_labels(pac_treinamento)

  atributos_teste, labels_teste = gerar_atributos_labels(pac_teste)



  # cria uma instância de classificador por árvore de decisão

  dtr = DecisionTreeClassifier()



  # treina o classificador com os pacientes de treinamento

  dtr.fit(atributos_treinamento,labels_treinamento)



  # obtém previsões para os atributos de teste

  previsoes = dtr.predict(atributos_teste)



  # retorna as previsões e os labels teste (reais)

  return previsoes,labels_teste





classe_prevista, classe_real = decision_tree_previsto_real(ds_red)



# mostra os 10 primeiros resultados

print("Alguns resultados iniciais...\n   previsto,  real")

for i in range(10):

    print("{}. {}, {}".format(i, classe_prevista[i], classe_real[i]))

 
import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.tree import DecisionTreeClassifier

#from support_functions import plot_confusion_matrix





# Implement the following function

def calcula_precisao(previsto, real):

  previsoes = 0

  previsoes_corretas = 0

  for i in range(len(previsto)):

    previsoes += 1

    if(previsto[i] == real[i]):

      previsoes_corretas += 1

  return (previsoes_corretas / previsoes)

    

# split the data

atributos, labels = gerar_atributos_labels(ds_red)



# train the model to get predicted and actual classes

dtc = DecisionTreeClassifier()

previsto = cross_val_predict(dtc, atributos, labels, cv=10)



# calculate the model score using your function

model_score = calcula_precisao(previsto, labels)

print("Score de precisão:", model_score)



# calculate the models confusion matrix using sklearns confusion_matrix function

class_labels = list(set(labels))

model_cm = confusion_matrix(y_true=labels, y_pred=previsto, labels=class_labels)



model_cm
import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestClassifier





def random_forest_previsto_real(data, n_estimators):

  

  # gerar atributos e labels

  atributos, labels = gerar_atributos_labels(data)



  # instanciar um classificador por floresta aleatória

  rfc = RandomForestClassifier(n_estimators=n_estimators)

  

  # obtem previsoes usando 10-fold cross validation

  previsto = cross_val_predict(rfc,atributos,labels,cv=10)



  # retorna valores previstos e reais

  return previsto,labels





# get the predicted and actual classes

num_arvores = 50              

previsto, real = random_forest_previsto_real(ds_red, num_arvores)



# calculate the model score using your function

precisao = calcula_precisao(previsto, real)

print("Score de precisão:", precisao)



# calculate the models confusion matrix using sklearns confusion_matrix function

class_labels = list(set(real))

model_cm = confusion_matrix(y_true=real, y_pred=previsto, labels=class_labels)



model_cm

import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
ds_enc = pd.read_excel('/kaggle/input/classif/dataset_classif.xlsx', index_col=0)
ds_enc
ds_enc.shape
ds_enc['encaminhamento'].value_counts()
import seaborn as sns



bins = np.linspace(ds_enc.Hematocrit.min(), ds_enc.Hematocrit.max(), 10)

g = sns.FacetGrid(ds_enc, col="Patient age quantile", hue="encaminhamento", palette="Set1", col_wrap=2)

g.map(plt.hist, 'Hematocrit', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
atributos = ds_enc.iloc[:,3:]



X = atributos

y = ds_enc['encaminhamento'].values
X
y
X= preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4) #mantendo proporção de treinamento em 70%

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 3 

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
yhat = neigh.predict(X_test)

yhat[0:5]
lim=25

avg=np.zeros((lim-1))

stdd=np.zeros((lim-1))

cm=[];

maxacc = 0;

bestk = 0;

for k in range(1,lim):

    

    #Train Model and Predict  

    kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

    yhat = kNN_model.predict(X_test)

    avg[k-1]=np.mean(yhat==y_test);

    stdd[k-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

    

    print("k = ",k,"*** accuracy = ",avg[k-1] )

    

    if avg[k-1] > maxacc:

        bestk = k 

        maxacc = avg[k-1]

   

print("The best value of k (best accuracy) is: ",bestk)
# Building the model again, using the k=7 with best accuracy

from sklearn.neighbors import KNeighborsClassifier

k = 4

#Train Model and Predict  

kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

kNN_model
yhat = kNN_model.predict(X_test)

yhat
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4) #profundidade arbitrária por enquanto

dt
dt.fit(X_train,y_train)
yhat1 = dt.predict(X_test)

yhat1
avg=np.mean(yhat1==y_test);

print("*** accuracy = ",avg )
from sklearn import svm

modsvm = svm.SVC()

modsvm.fit(X_train, y_train) 
yhat2 = modsvm.predict(X_test)

yhat2
avg=np.mean(yhat2==y_test);

print("*** accuracy = ",avg )
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.01).fit(X_train,y_train)

LR
yhat3 = LR.predict(X_test)

yhat3
avg=np.mean(yhat3==y_test);

print("*** accuracy = ",avg )