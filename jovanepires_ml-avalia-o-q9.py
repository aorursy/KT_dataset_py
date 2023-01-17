# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix



from warnings import filterwarnings

filterwarnings('ignore')



sns.set()

sns.set_palette("Set1")
# carregue os dados contidos no Dataset de Iris do scikit-learn

X, y = load_breast_cancer(return_X_y=True)
# dividir conjunto de dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
# aplicar pré-processamento

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# varie o parâmetro hidden_layer_sizes de [10,10], [25,50], [50,25] e [50,50] e mostre um gráfico contendo a diferença entre os scores

ind = []

trscore = []

tescore = []

confusion = []

params = [[10,10], [25,50], [50,25] , [50,50]]



for i in range(len(params)):

    ind.append(i)

    model = MLPClassifier(hidden_layer_sizes=params[i], random_state=42) # alpah=0.0001 by default

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trscore.append(model.score(X_train, y_train))

    tescore.append(model.score(X_test, y_test))

    confusion.append(confusion_matrix(y_test, y_pred))



plt.title("MLPClassifier: alpha=0.0001", fontsize=16)

plt.plot(ind,trscore,'r-', label='train score', marker='o')

plt.plot(ind,tescore,'b-', label='test score', marker='o')

plt.xticks(ind, ('[10,10]', '[25,50]', '[50,25]', '[50,50]'))

plt.legend()



plt.show()
# varie o parâmetro hidden_layer_sizes de [10,10], [25,50], [50,25] e [50,50] e mostre um gráfico contendo a diferença entre os scores

ind = []

trscore = []

tescore = []

confusion = []

params = [[10,10], [25,50], [50,25] , [50,50]]



for i in range(len(params)):

    ind.append(i)

    model = MLPClassifier(hidden_layer_sizes=params[i], alpha=0.001, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trscore.append(model.score(X_train, y_train))

    tescore.append(model.score(X_test, y_test))

    confusion.append(confusion_matrix(y_test, y_pred))



plt.title("MLPClassifier: alpha=0.001", fontsize=16)

plt.plot(ind,trscore,'r-', label='train score', marker='o')

plt.plot(ind,tescore,'b-', label='test score', marker='o')

plt.xticks(ind, ('[10,10]', '[25,50]', '[50,25]', '[50,50]'))

plt.legend()



plt.show()
# varie o parâmetro hidden_layer_sizes de [10,10], [25,50], [50,25] e [50,50] e mostre um gráfico contendo a diferença entre os scores

ind = []

trscore = []

tescore = []

confusion = []

params = [[10,10], [25,50], [50,25] , [50,50]]



for i in range(len(params)):

    ind.append(i)

    model = MLPClassifier(hidden_layer_sizes=params[i], alpha=0.01, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trscore.append(model.score(X_train, y_train))

    tescore.append(model.score(X_test, y_test))

    confusion.append(confusion_matrix(y_test, y_pred))



plt.title("MLPClassifier: alpha=0.01", fontsize=16)

plt.plot(ind,trscore,'r-', label='train score', marker='o')

plt.plot(ind,tescore,'b-', label='test score', marker='o')

plt.xticks(ind, ('[10,10]', '[25,50]', '[50,25]', '[50,50]'))

plt.legend()



plt.show()