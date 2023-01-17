import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np



treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",sep=r'\s*,\s*',engine='python',na_values="?")

teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",sep=r'\s*,\s*',engine='python',na_values="?")



treino = treino.drop(0, axis=0)
treino.head()
treino.shape
teste.head()
teste.shape
treino.describe()



treino["age"].value_counts().plot(kind="bar", figsize = (15,5), title = 'age',)

treino["native.country"].value_counts()
treino["sex"].value_counts().plot(kind="pie", figsize = (15,5), title = 'Sex',)
treino["workclass"].value_counts().plot(kind="bar", figsize = (5,5), title = 'Workclass',)
treino["education"].value_counts().plot(kind="bar", figsize = (8,5), title = 'Education',)
pd.crosstab(treino['age'], treino['income'],normalize = "index").plot()
pd.crosstab(treino['workclass'], treino['income'], normalize = "index").plot(kind = 'bar')
pd.crosstab(treino['education'], treino['income'], normalize = "index").plot(kind = 'bar', figsize = (8,5))
pd.crosstab(treino['marital.status'], treino['income'], normalize = "index").plot(kind = 'bar')
pd.crosstab(treino['occupation'], treino['income'], normalize = "index").plot(kind = 'bar', figsize = (10,5))
pd.crosstab(treino['relationship'], treino['income'], normalize = "index").plot(kind = 'bar')
pd.crosstab(treino['race'], treino['income'], normalize = "index").plot(kind = 'bar')
pd.crosstab(treino['sex'], treino['income'], normalize = "index").plot(kind = 'bar')
pd.crosstab(treino['capital.gain'], treino['income'], normalize = "index").plot(kind = 'bar', figsize = (35,8))
pd.crosstab(treino['capital.loss'], treino['income'], normalize = "index").plot(kind = 'bar', figsize = (35,8))
pd.crosstab(treino['hours.per.week'], treino['income'], normalize = "index").plot(kind = 'bar', figsize = (35,5))
pd.crosstab(treino['native.country'], treino['income'], normalize = "index").plot(kind = 'bar', figsize = (25,5))
treino.isna().sum()
treino["workclass"] = treino["workclass"].fillna('Private')

treino["occupation"] = treino["occupation"].fillna('Prof-specialty')

treino = treino.dropna()
treino.isna().sum()
treino['workclass'].value_counts()
worktoint = [24049,2499,2067,1278,1074,943,14,7]

workstr = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"]

treino['marital.status'].value_counts()
martoint = [14692, 10487, 4393, 1005, 979, 397, 23]

marstr = ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]

treino['occupation'].value_counts()
occtoint = [5854, 4030, 3991, 3720, 3584, 3212, 1966, 1572, 1350, 989, 912, 644, 143, 9]

occstr = ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"]

treino['relationship'].value_counts()
reltoint = [12947, 8155, 5004, 3384, 1534, 952]

relstr = ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"]

treino['race'].value_counts()
ractoint = [27428, 3028, 956, 311, 253]

racstr = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]

treino['sex'].value_counts()
sextoint = [21368, 10608]

sexstr = ["Male", "Female"]

ints = worktoint + martoint + occtoint + reltoint + ractoint + sextoint

strs = workstr + marstr + occstr + relstr + racstr + sexstr
def strtoint(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return string



def strtoint1(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return 24094



def strtoint2(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return 14692



def strtoint3(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return 5854



def strtoint4(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return 12947



def strtoint5(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return 27428



def strtoint6(string):

    for i in range(len(ints)):

        if string == strs[i]:

            return ints[i]

    return 21368

treino["workclass"] = treino["workclass"].apply(strtoint)

treino["marital.status"] = treino["marital.status"].apply(strtoint)

treino["occupation"] = treino["occupation"].apply(strtoint)

treino["relationship"] = treino["relationship"].apply(strtoint)

treino["race"] = treino["race"].apply(strtoint)

treino["sex"] = treino["sex"].apply(strtoint)



teste["workclass"] = teste["workclass"].apply(strtoint1)

teste["marital.status"] = teste["marital.status"].apply(strtoint2)

teste["occupation"] = teste["occupation"].apply(strtoint3)

teste["relationship"] = teste["relationship"].apply(strtoint4)

teste["race"] = teste["race"].apply(strtoint5)

teste["sex"] = teste["sex"].apply(strtoint6)
Xtreino = treino[['age', 'education.num','occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']]

Ytreino = treino.income



Xteste = teste[['age','education.num', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']]



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=30, p=1)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xtreino, Ytreino, cv=15)

knn.fit(Xtreino, Ytreino)

Yteste = knn.predict(Xteste)

accuracy = np.mean(scores)

accuracy
id_index = pd.DataFrame({'Id' : list(range(len(Yteste)))})

income = pd.DataFrame({'income' : Yteste})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')