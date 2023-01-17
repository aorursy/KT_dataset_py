# Imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV



# lendo o arquivo de treino

df_treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        na_values="Faltante")

df_treino.head()
# Lendo o arquivo de teste

df_teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        na_values="Faltante")

df_teste
# Separando valores numericos

X_treino = df_treino[["age","education.num","capital.gain","capital.loss","hours.per.week"]]

Y_treino = df_treino.income
# Pegando os valores nao numericos e transformando em dummies

dfDummies = pd.get_dummies(df_treino['native.country'], prefix = 'category')

X_treino = pd.concat([X_treino, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_treino['race'], prefix = 'category')

X_treino = pd.concat([X_treino, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_treino['sex'], prefix = 'category')

X_treino = pd.concat([X_treino, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_treino['relationship'], prefix = 'category')

X_treino = pd.concat([X_treino, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_treino['marital.status'], prefix = 'category')

X_treino = pd.concat([X_treino, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_treino['education'], prefix = 'category')

X_treino = pd.concat([X_treino, dfDummies], axis=1)



X_treino
# Primeiro teste simples com knn - 10

knn = KNeighborsClassifier(n_neighbors=10)

scores = cross_val_score(knn, X_treino, Y_treino, cv=10)

print("acuracia media foi de: " +str(scores.mean()) )
"""

for i in([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]):

    pca = PCA(n_components=i)

    pca.fit(X_treino)



    X_treino_pca = pca.transform(X_treino)

    knn = KNeighborsClassifier(n_neighbors=10)

    scores = cross_val_score(knn, X_treino_pca, Y_treino, cv=10)

    print("Pca com: " + str(i) + " deu: " + str(scores.mean())+ "de acuracia") 

"""
pca = PCA(n_components=60)

pca.fit(X_treino)

X_treino_pca = pca.transform(X_treino)
"""

modelo = GridSearchCV(KNeighborsClassifier(),cv = 10, n_jobs = -1, verbose = 2, param_grid={"n_neighbors": range(3,51,6),

                                                                                            "p":[1,3,5],

                                                                                            "weights":["uniform","distance"],

                                                                                            "metric": ["euclidean","manhattan","minkowski"]})

modelo = modelo.fit(X_treino_pca, Y_treino)

""" 
#print(modelo.best_score_)

#print(modelo.best_estimator_)

#print(modelo.best_params_)
melhor_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',

                     metric_params=None, n_jobs=None, n_neighbors=27, p=1,

                     weights='uniform')

scores = cross_val_score(melhor_knn, X_treino_pca, Y_treino, cv=10)

melhor_knn = melhor_knn.fit(X_treino_pca, Y_treino)

scores

df_teste = pd.concat([df_teste, df_treino], sort = False)





X_teste = df_teste[["age","education.num","capital.gain","capital.loss","hours.per.week"]]



dfDummies = pd.get_dummies(df_teste['native.country'], prefix = 'category')

X_teste = pd.concat([X_teste, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_teste['race'], prefix = 'category')

X_teste = pd.concat([X_teste, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_teste['sex'], prefix = 'category')

X_teste = pd.concat([X_teste, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_teste['relationship'], prefix = 'category')

X_teste = pd.concat([X_teste, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_teste['marital.status'], prefix = 'category')

X_teste = pd.concat([X_teste, dfDummies], axis=1)



dfDummies = pd.get_dummies(df_teste['education'], prefix = 'category')

X_teste = pd.concat([X_teste, dfDummies], axis=1)



X_teste = X_teste.iloc[:16280,:]

X_teste_pca = pca.transform(X_teste)



Y_teste_pred = melhor_knn.predict(X_teste_pca)

savepath = "predictions.csv"

pred = pd.DataFrame(Y_teste_pred, columns = ["income"])

pred.to_csv(savepath, index_label="Id")

pred