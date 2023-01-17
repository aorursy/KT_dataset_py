import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn import datasets

import matplotlib.pyplot as plt



digits = datasets.load_digits()

print(digits.keys())

print(digits.target_names)

print(digits.images)
digits.data.shape
plt.imshow(digits.images[1])
digits.images[100].shape
X = pd.DataFrame(digits.data)

y = pd.DataFrame(digits.target, columns=["target"])

dados = pd.concat([X,y], axis=1)

dados.head()



plt.hist(y['target'], bins=10, rwidth=0.9)

plt.title("Contagem de Imagens de digitos")

plt.xlabel("digito")

plt.ylabel("contagem")

plt.show()
dados.isnull().sum()
from sklearn.model_selection import train_test_split #importa a classe train_test_split

X = digits.data  #define quais são os atributos de entrada do modelo

y = digits.target #define quais são os rotúlos das instâncias

#divide os dados em 80% para treinamento e 20% para teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,   stratify=y) 



img = X_train[0].reshape(8,8)

resp = y_train[0]

plt.imshow(img)

resp
from sklearn.neighbors import KNeighborsClassifier #importa a classe da biblioteca

knn = KNeighborsClassifier(n_neighbors=7) # cria o objeto

knn.fit(X_train, y_train) # aplica o método fit para ajustar os dados ao modelo

previsao=knn.predict(X_test) #faz a predição

print(knn.score(X_test, y_test)) # Calcula a acurácia (precisão)
img = X_train[0].reshape(1,-1)

print(knn.predict(img))
compara = pd.DataFrame(y_test)

compara ['test'] = y_test

compara.head(100)