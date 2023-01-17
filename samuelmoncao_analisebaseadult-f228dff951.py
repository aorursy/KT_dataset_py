# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/db-adult"))

# Any results you write to the current directory are saved as output.
adult = pd.read_csv("../input/db-adult/train_data.csv",
        sep=',',
        engine='python',
        na_values="?")
adult.shape
nadult = adult.dropna()
nadult.describe()
import matplotlib.pyplot as plt
adult["age"].value_counts().plot(kind="bar")
#adult["fnlwgt"].value_counts().plot(kind="bar")
#Desabilitado para deixar o codigo mais leve
#Aqui podemos ver como a relação com o fnlwgt é de chave única
adult["native.country"].value_counts()
adult["hours.per.week"].value_counts().plot(kind="bar")
plt.axis([0,10, 0, 16000])
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="pie")
adult["race"].value_counts().plot(kind="pie")
adult["sex"].value_counts().plot(kind="pie")
adult["workclass"].value_counts().plot(kind="barh")
#Consideraremos inicialmente todos os valores numéricos
Xadult_v1 = nadult[["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]]
Yadult = nadult.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=26)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult_v1, Yadult, cv=10)
scores.mean()

#Aqui podemos suspeitar, baseado nas análises anteriores, que os valores de 'fnlwgt'
#na verdade não são bons exemplos de teste para o modelo. Se trata de um valor que é
#único na maioria dos casos, como mostrado nos gráficos.

#Agora, transformaremos todas as colunas em número afim de aproveitar melhor as possibilidades estudadas.

from sklearn import preprocessing

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

#Desconsideraremos os seguintes campos:
#native.country: Como podemos ver, os dados de teste são na maioria americanos
#race: Temos uma grande maioria de pessoas brancas nessa pesquisa
#fnlwgt: Como explicado anteriormente, se trata praticamente de uma chave única para cada pessoa

Xadult_v2 = numAdult[["age", "workclass", "education", "education.num", "occupation","sex", "capital.gain", "capital.loss","hours.per.week"]]
Yadult = numAdult.income
knn = KNeighborsClassifier(n_neighbors=26)
scores = cross_val_score(knn, Xadult_v2, Yadult, cv=10)

scores.mean()
Xadult_v3 = numAdult[["age", "workclass", "education", "education.num", "occupation","sex", "capital.gain", "capital.loss","hours.per.week"]]
Yadult = numAdult.income
knn = KNeighborsClassifier(n_neighbors=70)
scores = cross_val_score(knn, Xadult_v3, Yadult, cv=10)

#O valor de knn teve uma nova interação para que o valor ótimo fosse encontrado novamente

scores.mean()