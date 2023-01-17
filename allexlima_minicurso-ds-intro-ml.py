%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np
df_jurassic = pd.read_csv("https://raw.githubusercontent.com/allexlima/MinicursoDataScience/master/modulo2/jurassic.csv")

df_jurassic
df_jurassic.drop(['Animal'], axis=1, inplace=True)

df_jurassic
def dist(a, b):

    # Esta função irá ser executada para cada exemplo da coleção, linha por linha

    summ = 0

    

    for i in range(len(a)): # Para cada atributo da linha, faça:

        if isinstance(a[i], str): # Se o atributo for string...

            val = 0 if a[i] == b[i] else 1 # `val` recebe 0 caso o atributo da coleção seja igual ao da instância ou 0, caso contrário.

        else: # Caso o atributo não seja string, então apenas calcula a diferença

            val = int(a[i]) - int(b[i])

        

        # Independente das condições acima, calcula o quadrado de `val`

        # e acrescenta `val` ao acumulador/somatório `summ`

        summ += val**2

        

    return np.sqrt(summ) # Retorna a raiz quadrada do somatório `summ`
instancia = np.array(['Cretaceous', 'V', 2, 'V']) # Era, Dentes, Número de Asas, Penas Simétricas
df_jurassic.iloc[0]
df_jurassic.iloc[0][:-1]
np.array(df_jurassic.iloc[0][:-1])
instancia
dist(df_jurassic.iloc[0][:-1], instancia)



# Perceba que passar o `pd.iloc` no formato Numpy é SUPEEEER opcional
eucledian_dist = df_jurassic.apply(lambda row: dist(row[:-1], instancia), axis=1)

eucledian_dist
df_jurassic['distance'] = eucledian_dist

df_jurassic
df_jurassic.sort_values('distance', ascending=True)
df_jurassic.sort_values('distance', ascending=True).head(3)
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
df_train = df_jurassic.iloc[:, :-1].copy()

df_train
le = {}



for column in df_train.columns: # Para cada coluna do DataFrame

    enc = LabelEncoder() # Instâncie um novo objeto LabelEncoder

    df_train[column] = enc.fit_transform(df_train[column].astype(str)) # Codifique os valores dessa coluna

    le[column] = enc # Salve o objeto numa tabela hash/dicionário cuja a chave seja o mesmo nome da coluna

    

df_train
le['Era'].classes_
X = df_train.iloc[:, :-1] # Obtém todos os atributos, exceto a coluna "Ave"

y = df_train['Ave'] # Seleciona apenas a coluna "Ave"
neigh = KNeighborsClassifier(n_neighbors=3, p=2)

neigh.fit(X, y)
instancia
test = np.full(len(instancia), np.nan, dtype=int)



test[0] = le['Era'].transform([instancia[0]])[0]

test[1] = le['Dentes'].transform([instancia[1]])[0]

test[2] = le['Número de Asas'].transform([instancia[2]])[0]

test[3] = le['Penas Simétricas'].transform([instancia[3]])[0]



test
y_pred = neigh.predict([test])

y_pred
le['Ave'].inverse_transform(y_pred)
dist, indexes = neigh.kneighbors([test])



print("Distâncias:", dist)

print("Índices", indexes)
X.loc[indexes[0]]
temp = X.loc[indexes[0]].copy()



temp['Ave'] = y[indexes[0]]



for col in temp.columns:

    temp[col] = le[col].inverse_transform(temp[col])

    

temp['dist'] = dist[0]



temp
df_titanic = pd.read_csv("https://raw.githubusercontent.com/allexlima/MinicursoDataScience/master/modulo2/titanic.csv")

df_titanic.head()
df_titanic.shape
df_titanic.isnull().sum()
# -- Vamos excluir algumas colunas e também as linhas que não contém valores



df_titanic.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)

df_titanic.dropna(inplace=True)

df_titanic.head()
df_titanic.shape
df_titanic.isnull().sum()
le = {}



for column in df_titanic.select_dtypes('object').columns:

    print("Codificando coluna `%s`..." % column)

    

    enc = LabelEncoder()

    

    df_titanic[column] = enc.fit_transform(df_titanic[column])

    le[column] = enc
df_titanic.head()
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)





ax[0].set_title("Passenger's sex")

df_titanic['Survived'].groupby(df_titanic['Sex']).sum().plot(kind='pie', explode=(0.15, 0), 

                                                             autopct="%.2f%%", ax=ax[0])



ax[1].set_title("Passenger's class")

df_titanic['Survived'].groupby(df_titanic['Pclass']).sum().plot(kind='pie', autopct="%.2f%%", 

                                                                labels=['1st', '2nd', '3rd'],

                                                                explode=(.05, .05, .05),

                                                                ax=ax[1])

ax[2].set_title("Where the passenger got on the ship")

df_titanic['Survived'].groupby(df_titanic['Embarked']).sum().plot(kind='pie', autopct="%.2f%%", 

                                                                labels=['Cherbourg', 'Southampton', 'Queenstown'],

                                                                explode=(.1, .1, .1), ax=ax[2])
from sklearn.model_selection import train_test_split





X = df_titanic.drop(["PassengerId", "Survived"], axis=1)

y = df_titanic["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
X.shape, X_train.shape, X_test.shape
from sklearn.neural_network import MLPClassifier





clf = MLPClassifier(hidden_layer_sizes=(15, 90), activation='relu', solver='adam', 

                    max_iter=300, random_state=11)



clf.fit(X_train, y_train)
import scikitplot as skplt
y_pred = clf.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred)
## Podemos também 'plotar' a CM de forma normalizada...



y_pred[:4]
y_probas = clf.predict_proba(X_test)



y_probas[:4]
skplt.metrics.plot_roc(y_test, y_probas)