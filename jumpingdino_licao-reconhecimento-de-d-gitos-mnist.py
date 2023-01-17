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
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

#A partir daqui temos os dados carregados
X_train = train.drop(['label'],axis=1)

y_train = train['label']
#Mostrar uma imagem de um número qualquer

plt.imshow(np.array(X_train.iloc[1,:]).reshape(28,28), cmap=plt.get_cmap('gray'))
X = X_train.copy()

y = y_train.copy()
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score



#Como só os dados de treino possuem as labels(classificação real)

#Realize um train_test_split a partir dos dados de treino

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



#Avaliar na particao de treino e avaliar os resultados através de classification report



#DESAFIO

#Treinar os dados com redes neurais e avaliar os hiperparametros com GridSearch 
from sklearn.metrics import classification_report



#abordagem Holdout - parametros padrao

mlp = MLPClassifier()

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)

print(classification_report(y_pred,y_test))



from sklearn.model_selection import GridSearchCV



mlp = MLPClassifier()

nn_params = {'hidden_layer_sizes':[(100,),(100,100),(85,50)],

            'learning_rate' : ['constant','adaptive'],

                                    'learning_rate_init' : [0.001,0.02,0.1]}

                                      

gs_nn = GridSearchCV(mlp,param_grid =nn_params,verbose = 1)

gs_nn.fit(X_train,y_train)

y_pred = gs_nn.predict(y_test)

print(classification_report(y_pred,y_test))