# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix



from warnings import filterwarnings

filterwarnings('ignore')



sns.set()

sns.set_palette("Set1")
# carregue os dados contidos no Dataset de Iris do scikit-learn

X, y = load_iris(return_X_y=True)
# dividir conjunto de dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
# aplicar pré-processamento

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# varie o parâmetro hidden_layer_sizes de 10 a 100 (de 10 em 10) e mostre um gráfico contendo a diferença entre os scores

ind = []

trscore = []

tescore = []

confusion = []



for i in np.arange(10, 101, step=10): # interval [1:100]

    ind.append(i)

    model = MLPClassifier(hidden_layer_sizes=i, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trscore.append(model.score(X_train, y_train))

    tescore.append(model.score(X_test, y_test))

    confusion.append(confusion_matrix(y_test, y_pred))



plt.title("MLPClassifier - learning_rate = constant", fontsize=16)

plt.plot(ind,trscore,'r-', label='train score', marker='o')

plt.plot(ind,tescore,'b-', label='test score', marker='o')

plt.legend();



plt.show()
# apresente a matriz de confusão dos dados de teste.

data = load_iris()

targets = data.target_names

df_confusion = pd.DataFrame(np.around(np.mean(confusion, axis=0), 0), index=targets, columns=targets)



plt.title("Confusion Matrix", fontsize=16)

sns.heatmap(df_confusion, annot=True)

plt.show()
# varie o parâmetro hidden_layer_sizes de 10 a 100 (de 10 em 10) e mostre um gráfico contendo a diferença entre os scores

ind = []

trscore = []

tescore = []

confusion = []



for i in np.arange(10, 101, step=10): # interval [1:100]

    ind.append(i)

    model = MLPClassifier(hidden_layer_sizes=i, learning_rate='adaptive', random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trscore.append(model.score(X_train, y_train))

    tescore.append(model.score(X_test, y_test))

    confusion.append(confusion_matrix(y_test, y_pred))



plt.title("MLPClassifier - learning_rate = adaptive", fontsize=16)

plt.plot(ind,trscore,'r-', label='train score', marker='o')

plt.plot(ind,tescore,'b-', label='test score', marker='o')

plt.legend();



plt.show()
# apresente a matriz de confusão dos dados de teste.

data = load_iris()

targets = data.target_names

df_confusion = pd.DataFrame(np.around(np.mean(confusion, axis=0), 0), index=targets, columns=targets)



plt.title("Confusion Matrix", fontsize=16)

sns.heatmap(df_confusion, annot=True)

plt.show()