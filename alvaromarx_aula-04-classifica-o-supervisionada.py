# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import make_blobs # Não se preocupe com essa função, ela apenas gera um conjunto de dados com labels



X, y = make_blobs(1000) # Criando 1000 pontos num conjunto de 2 dimensões, cada um com sua label. 

df_sample = pd.DataFrame(data=np.hstack([X,y.reshape(y.shape[0],1)])) # Criando um dataFrame com base nos dados X, y



sns.scatterplot(x=X[:,0],y=X[:,1],hue=y) # Mostrando o dataset de uma forma visual

plt.plot()



df_sample.head() 
y_example = np.array(["banana"]*10 + ["maca"]* 10 + ["laranja"] * 10) # Criando vetor com as strings banana, laranja e maca

np.random.shuffle(y_example) # Deixando esse vetor de forma aleatória

print(f"Antes da fatorização: {y_example}")

y_example, labels =  pd.factorize(y_example) # Fatorizando vetor

print(f"Depois da fatorização: {y_example}")

for i in range(len(labels)):

    print(f"O número {i} representa a label {labels[i]}") # Descobrindo qual label está ligada a cada número
from sklearn.model_selection import train_test_split
X_, y_ = make_blobs(1000,random_state=42) # Criando 100 pontos separados em 3 labels

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.1, random_state=42)

_, c = np.unique(y_,return_counts=True) # Np.unique -> retorna uma tupla (valores únicos da lista, contagem de cada valor)

_, c_Train = np.unique(y_train_,return_counts=True)

_, c_Test  = np.unique(y_test_,return_counts=True)

print(100*c/c.sum()) # Porcentagem da frequência de cada label no dataset 

print(100*c_Train/c_Train.sum()) # Porcentagem da frequência de cada label no dataset de treinamento

print(100*c_Test/c_Test.sum()) # Porcentagem da frequência de cada label no dataset de teste

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()

dummy
# dummy.fit(X_train, y_train)
# y_pred = dummy.predict(X_test) # conseguindo predição com base no X_test

# print(f"valor real: {y_test}")

# print(f"valor previsto: {y_pred}")





# plt.imshow(np.vstack([y_test.reshape(1,y_test.shape[0]),y_pred.reshape(1,y_pred.shape[0])])) # representando test x pred em forma de imagem

# plt.title("labels reais (em cima) e labels previstas (em baixo)")

# plt.show()

# dummy.score(X_test, y_test)