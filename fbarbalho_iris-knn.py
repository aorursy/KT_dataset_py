# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Set style
sns.set_style("ticks")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importa o dataset 
data = pd.read_csv('../input/Iris.csv')
# Visualiza 10 linhas aleatorios do dataset
data.sample(10)
# SUbplot
#fig, ax =plt.subplots(1,2)

# Plota um scatterplot a partir da function lmplot do seaborn 
# O atributo fit_reg = false plota o gráfico sem a linha de regressão
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", data=data, fit_reg=False, legend=False,hue='Species')

# Posiciona a legenda para uma area vazia do gráfico 
plt.legend(loc='lower right')
# Plota um scatterplot a partir da function lmplot do seaborn 
# O atributo fit_reg = false plota o gráfico sem a linha de regressão
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=data, fit_reg=False, legend=False,hue='Species')

# Posiciona a legenda para uma area vazia do gráfico 
plt.legend(loc='lower right')
from sklearn.cross_validation import train_test_split
# Cria matrizes de dados para target e predictors  
X = np.array(data.drop(columns = ['Id','Species'])) 
y = np.array(data['Species']) 
# separa os dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Carrega biblioteca
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Instancia um modelo (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# Fit Model
knn.fit(X_train, y_train)

# prevê a specie
pred = knn.predict(X_test)

# Avalia acuracia
accuracy_score(y_test,pred)

# Matriz de confusão
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train) 

# prevê a specie
pred = clf.predict(X_test)

# Avalia acuracia
accuracy_score(y_test,pred)

# Matriz de confusão
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
