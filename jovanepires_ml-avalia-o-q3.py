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

from sklearn.svm import SVC



from warnings import filterwarnings

filterwarnings('ignore')



sns.set()

sns.set_palette("Set1")
# carregar dados do dataset

X, y = load_breast_cancer(return_X_y=True);
# divida o conjunto de dados entre treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
# normalize os dados de entrada.

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
#  varie o parâmetro C de 0.1 a 1 e mostre um gráfico contendo a diferença entre os scores obtidos

ind = []

trscore = []

tescore = []



for i in np.linspace(0.1, 1.0, num=10):

    ind.append(i)

    model = SVC(C=i)

    model.fit(X_train, y_train)

    trscore.append(model.score(X_train, y_train))

    tescore.append(model.score(X_test, y_test))

    

plt.title("SVC", fontsize=16)    

plt.plot(ind,trscore,'r-', label='train score', marker='o')

plt.plot(ind,tescore,'b-', label='test score', marker='o')

plt.legend();



plt.show()