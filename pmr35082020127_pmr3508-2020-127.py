import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
adulto = pd.read_csv('../input/adult-pmr3508/train_data.csv',

        names=[

        'id', "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adulto = adulto.drop(0, axis=0) #retirar linha repetida no início

adultoteste = pd.read_csv('../input/adult-pmr3508/test_data.csv',

        names=[

        'id', "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],    

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adultoteste = adultoteste.drop(0, axis=0)
adulto.head()
adultoteste.head()
adulto.describe()
adulto.shape
adulto['Race'].value_counts()
piechart = adulto['Race'].value_counts().plot(kind='pie', figsize = (7,7))

piechart.legend(loc="upper right")
adulto['Education-Num'].value_counts()
grafbarra = adulto['Education-Num'].value_counts().plot(kind="bar", figsize = (5,5), title = 'Education-Num Bar Graph')

grafbarra.set_xlabel('Education-Num')

grafbarra.set_ylabel('Quantity')
comp1 = pd.crosstab(adulto['Age'], adulto['Target'])

comp1.plot()
comp2 = pd.crosstab(adulto['Workclass'], adulto['Target'])

comp2.plot()
comp3 = pd.crosstab(adulto['Education'], adulto['Target'])

comp3.plot()
comp4 = pd.crosstab(adulto['Education-Num'], adulto['Target'])

comp4.plot()
comp5 = pd.crosstab(adulto['Marital Status'], adulto['Target'])

comp5.plot()
comp6 = pd.crosstab(adulto['Occupation'], adulto['Target'])

comp6.plot()
adulto['Occupation'].value_counts()
comp7 = pd.crosstab(adulto['Relationship'], adulto['Target'])

comp7.plot()
adulto['Relationship'].value_counts()
comp8 = pd.crosstab(adulto['Race'], adulto['Target'])

comp8.plot()
comp9 = pd.crosstab(adulto['Sex'], adulto['Target'])

comp9.plot()
comp10 = pd.crosstab(adulto['Capital Gain'], adulto['Target'])

comp10.plot()
comp11 = pd.crosstab(adulto['Capital Loss'], adulto['Target'])

comp11.plot()
comp12 = pd.crosstab(adulto['Hours per week'], adulto['Target'])

comp12.plot()
comp13 = pd.crosstab(adulto['Country'], adulto['Target'])

comp13.plot()
adulto.isnull().sum()
adultoteste.isnull().sum()
relationlist = [

["Husband" ,          13193],

["Not-in-family",      8304],

["Own-child"     ,     5068],

["Unmarried"      ,    3446],

["Wife"            ,   1568],

["Other-relative"   ,   981]]



def relationset(parametro):

    for i in range(len(relationlist)):

        if parametro == relationlist[i][0]:

            return relationlist[i][1]

    return parametro



racelist = [

["White",                 25932],

["Black",                  2817],

["Asian-Pac-Islander",      895],

["Amer-Indian-Eskimo",      286],

["Other",                   231]

]



def raceset(parametro):

    for i in range(len(racelist)):

        if parametro == racelist[i][0]:

            return racelist[i][1]

    return parametro



sexlist = [

["Male",      20379],

["Female",     9782]

]



def sexset(parametro):

    for i in range(len(sexlist)):

        if parametro == sexlist[i][0]:

            return sexlist[i][1]

    return parametro
adulto['Relationship'] = adulto['Relationship'].apply(relationset)

adulto['Race'] = adulto['Race'].apply(raceset)

adulto['Sex'] = adulto['Sex'].apply(sexset)



adultoteste['Relationship'] = adultoteste['Relationship'].apply(relationset)

adultoteste['Race'] = adultoteste['Race'].apply(raceset)

adultoteste['Sex'] = adultoteste['Sex'].apply(sexset)
nadulto = adulto.dropna()

nadultoteste = adultoteste
nadulto.head()
nadultoteste.head()
Xadulto = nadulto[['Age', 'Education-Num','Relationship', 'Capital Gain', 'Capital Loss']]

Yadulto = nadulto.Target
Xadultoteste = nadultoteste[['Age', 'Education-Num','Relationship', 'Capital Gain', 'Capital Loss']]
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors=28, p=1)
from sklearn.model_selection import cross_val_score
scores  = cross_val_score(Knn, Xadulto, Yadulto, cv = 13)
Knn.fit(Xadulto, Yadulto)
Yadultoteste = Knn.predict(Xadultoteste)
Yadultoteste = Knn.predict(Xadultoteste)
accuracy = np.mean(scores)
accuracy
id_index = pd.DataFrame({'Id' : list(range(len(Yadultoteste)))})

income = pd.DataFrame({'income' : Yadultoteste})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')