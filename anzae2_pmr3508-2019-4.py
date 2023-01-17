import pandas as pd

import sklearn

import matplotlib.pyplot as plt



adult = pd.read_csv("/Users/Ana/Desktop/Mecatrônica/8º semestre/Cozman/adult.data.txt",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



testAdult = pd.read_csv("/Users/Ana/Desktop/Mecatrônica/8º semestre/Cozman/adult.test.txt",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
#Exploração de dados



adult
adult["Country"].value_counts().plot(kind="bar")
adult["Education-Num"].value_counts().plot(kind="bar")
adult["Relationship"].value_counts().plot(kind="bar")
#preparação de dados

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



nAdult0 = adult.dropna()

nTestAdult0 = testAdult.dropna()



nAdult = nAdult0.apply(preprocessing.LabelEncoder().fit_transform)

nTestAdult = nTestAdult0.apply(preprocessing.LabelEncoder().fit_transform)



#parâmetros do exemplo

Xadult1 = nAdult.iloc[:,[0, 4, 10, 11, 12]]

Yadult = nAdult.Target

XtestAdult1 = nTestAdult[["Age","Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = nTestAdult.Target



#mudança de parâmetros

'''Xadult2 = nadult.iloc[["Race", "Education-Num", "Capital Gain", "Sex", "Hours per week", "Relationship"]]

XtestAdult2 = nTestAdult[["Race", "Education-Num", "Capital Gain", "Sex", "Hours per week", "Relationship"]]'''

Xadult2 = nAdult.iloc[:,[4,7,8,9,10,12]]

XtestAdult2 = nTestAdult.iloc[:,[4,7,8,9,10,12]]



#outra mudança de parâmetros

'''Xadult3 = nadult[["Age", "Education-Num", "Capital Gain", "Sex", "Workclass"]]

XtestAdult3 = nTestAdult[["Age", "Education-Num", "Capital Gain", "Sex", "Workclass"]]'''

Xadult3 = nAdult.iloc[:,[0,1,5,9,10]]

XtestAdult3 = nTestAdult.iloc[:,[0,1,5,9,10]]

#testes



#diferentes valores de knn (5, 10 e 25)

knn5 = KNeighborsClassifier(n_neighbors=5)

knn10 = KNeighborsClassifier(n_neighbors=10)

knn20 = KNeighborsClassifier(n_neighbors=25)



#para os parâmetros originais

scores5_1 = cross_val_score(knn5, Xadult1, Yadult, cv=10)

scores10_1 = cross_val_score(knn10, Xadult1, Yadult, cv=10)

scores20_1 = cross_val_score(knn20, Xadult1, Yadult, cv=10)



#para a primeira mudança

scores5_2 = cross_val_score(knn5, Xadult2, Yadult, cv=10)

scores10_2 = cross_val_score(knn10, Xadult2, Yadult, cv=10)

scores20_2 = cross_val_score(knn20, Xadult2, Yadult, cv=10)



#para a segunda mudança

scores5_3 = cross_val_score(knn5, Xadult3, Yadult, cv=10)

scores10_3 = cross_val_score(knn10, Xadult3, Yadult, cv=10)

scores20_3 = cross_val_score(knn20, Xadult3, Yadult, cv=10)
#resultados



#para a primeira seleção de parâmetros

knn5.fit(Xadult1,Yadult)

knn10.fit(Xadult1,Yadult)

knn20.fit(Xadult1,Yadult)



YtestPred5_1 = knn5.predict(XtestAdult1)

YtestPred10_1 = knn10.predict(XtestAdult1)

YtestPred20_1 = knn20.predict(XtestAdult1)



print("knn = 5:")

print(accuracy_score(YtestAdult,YtestPred5_1))

print("\n knn = 10:")

print(accuracy_score(YtestAdult,YtestPred10_1))

print("\n knn = 20:")

print(accuracy_score(YtestAdult,YtestPred20_1))
#segunda seleção de parâmetros

knn5.fit(Xadult2,Yadult)

knn10.fit(Xadult2,Yadult)

knn20.fit(Xadult2,Yadult)



YtestPred5_2 = knn5.predict(XtestAdult2)

YtestPred10_2 = knn10.predict(XtestAdult2)

YtestPred20_2 = knn20.predict(XtestAdult2)



print("knn = 5:")

print(accuracy_score(YtestAdult,YtestPred5_2))

print("\n knn = 10:")

print(accuracy_score(YtestAdult,YtestPred10_2))

print("\n knn = 20:")

print(accuracy_score(YtestAdult,YtestPred20_2))
#terceira seleção de parâmetros

knn5.fit(Xadult3,Yadult)

knn10.fit(Xadult3,Yadult)

knn20.fit(Xadult3,Yadult)



YtestPred5_3 = knn5.predict(XtestAdult3)

YtestPred10_3 = knn10.predict(XtestAdult3)

YtestPred20_3 = knn20.predict(XtestAdult3)



print("knn = 5:")

print(accuracy_score(YtestAdult,YtestPred5_3))

print("\n knn = 10:")

print(accuracy_score(YtestAdult,YtestPred10_3))

print("\n knn = 20:")

print(accuracy_score(YtestAdult,YtestPred20_3))
'''

conclusão



podemos concluir que, dentre os testes realizados, o que apresentou melhor resultado foi

knn = 25

parâmetros analisados:

Race, Education-Num, Capital Gain, Sex, Hours per week, Relationship

'''