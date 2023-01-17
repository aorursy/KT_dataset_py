import pandas as pd
adult = pd.read_csv("adult.train.csv",
        names=[
        "Age", "Workclass","fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        engine='python',
        na_values="?")
adult.shape
adult.head()
adult["Country"].value_counts()
import matplotlib.pyplot as plt

# Formatação mais bonita para os notebooks
%matplotlib inline 
    
adult["Age"].value_counts().plot(kind='bar')
adult["Sex"].value_counts().plot(kind='bar')
adult["Education"].value_counts().plot(kind="bar")

adult["Occupation"].value_counts().plot(kind="bar")


# Faremos o mesmo com o nosso data set de treino! Retiraremos suas linhas com missing value...

testAdult = pd.read_csv('adult.test.csv', names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        engine='python', index_col=0,
        na_values="?")


# Separando os atributos da base de treino

Xadult = adult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = adult.Target
# Separando os atributos da base teste

XtestAdult = testAdult[["Age","Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]]


import sklearn
# Importaremos o módulo que contém o classificador K-Nearest-Neighbors

from sklearn.neighbors import KNeighborsClassifier

# Criamos o objeto knn, da classe KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=30)
# Importaremos o módulo necessário para fazer a validação cruzada

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xadult, Yadult, cv=10) #Exemplo de validação cruzada: K-fold (visto em aula)

print(scores)

knn.fit(Xadult, Yadult)

Ypred = knn.predict(XtestAdult)

Ypred
df = pd.DataFrame(Ypred)

df.to_csv('predicoes.csv')
