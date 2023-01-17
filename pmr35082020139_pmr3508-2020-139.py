import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

                      header=0,

                      names = ['Id', 'Age', 'Workclass', 'fnlwgt', 'Education','Education num', 'Marital status',

                               'Occupation', 'Relationship', 'Race', 'Sex', 'Capital gain', 'Capital loss',

                               'Hours per week', 'Country', 'Target'],

                      sep=',',

                      engine = 'python',

                      na_values = '?')

dataset
del dataset["Id"]

dataset.head()
dataset.isnull().sum()
print("Workclass:")

print(dataset["Workclass"].describe())

print("{0}% dos dados tem como valor \"{1}\"".format(

    np.round(100*dataset["Workclass"].describe().freq/(dataset.shape[0] - dataset.isnull().sum()["Workclass"]), 2),

    dataset["Workclass"].describe().top))



print("\nOccupation:")

print(dataset["Occupation"].describe())

print("{0}% dos dados tem como valor \"{1}\"".format(

    np.round(100*dataset["Occupation"].describe().freq/(dataset.shape[0] - dataset.isnull().sum()["Occupation"]), 2),

    dataset["Occupation"].describe().top))



print("\nCountry")

print(dataset["Country"].describe())

print("{0}% dos dados tem como valor \"{1}\"".format(

    np.round(100*dataset["Country"].describe().freq/(dataset.shape[0] - dataset.isnull().sum()["Country"]), 2),

    dataset["Country"].describe().top))
plt.figure(figsize=(7, 3))

dataset["Country"].value_counts().plot(kind='bar')

plt.xlabel('Country')

plt.ylabel('Quantity')

plt.show()



plt.figure(figsize=(5, 5))

dataset["Workclass"].value_counts().plot(kind='pie')

plt.xlabel('Workclass')

plt.ylabel('Quantity')

plt.show()
pd.crosstab(dataset["Country"], dataset["Target"], normalize='index').plot.bar()

plt.title("Salário em função do país de origem (em porcentagem)")

plt.show()



pd.crosstab(dataset["Workclass"], dataset["Target"]).plot.bar()

plt.title("Salário em função do setor de trabalho (em porcentagem)")

plt.show()
def countryToNumber(df):

    df["Country"] = df["Country"].fillna("United-States")

    dfCountries = list(df['Country'].values)

    paises = []

    for i in dfCountries:

        if not i in paises:

            paises.append(i)

    paises.remove("United-States")

    

    df["Country"] = df["Country"].replace(["United-States"], 1)

    df["Country"] = df["Country"].replace(paises, 0)

    

def workclassToNumber(df):

    df["Workclass"] = df["Workclass"].fillna("Private")

    dfWorks = list(df['Workclass'].values)

    work = []

    for i in dfWorks:

        if not i in work:

            work.append(i)

    work.remove("Private")

    

    df["Workclass"] = df["Workclass"].replace(["Private"], 1)

    df["Workclass"] = df["Workclass"].replace(work, 0)
countryToNumber(dataset)

workclassToNumber(dataset)

dataset.head()
pd.crosstab(dataset["Country"], dataset["Target"], normalize='index').plot.bar()

plt.title("Salário em função do país de origem (em porcentagem)")

plt.show()



pd.crosstab(dataset["Workclass"], dataset["Target"], normalize='index').plot.bar()

plt.title("Salário em função do setor de trabalho (em porcentagem)")

plt.show()
print("Workclass:")

print(dataset["Workclass"].mean())



print("\nCountry")

print(dataset["Country"].mean())
plt.figure(figsize=(5, 5))

dataset["Occupation"].value_counts().plot(kind='pie')

plt.xlabel('Occupation')

plt.ylabel('')

plt.show()



plt.figure(figsize=(7, 3))

dataset["Occupation"].value_counts().plot(kind='bar')

plt.xlabel('Occupation')

plt.ylabel('Quantity')

plt.show()
pd.crosstab(dataset["Occupation"], dataset["Target"]).plot.bar()

plt.title("Salário em função das ocupações (em porcentagem)")

plt.show()
filtered_dataset = dataset

del filtered_dataset["Country"]

filtered_dataset = dataset.dropna()

filtered_dataset
filtered_dataset.shape[0]/dataset.shape[0]
graficos = pd.plotting.scatter_matrix(filtered_dataset, alpha=0.2, figsize=(10, 10), diagonal='hist')
filtered_dataset.describe()
plt.figure(figsize=(15, 5))

filtered_dataset["Age"].value_counts().sort_index().plot(kind='bar')

plt.xlabel('Age')

plt.ylabel('Quantity')

plt.show()
plt.figure(figsize=(5, 5))

filtered_dataset["Education"].value_counts().plot.pie()

plt.ylabel('')

plt.title("Education")

plt.show()
plt.figure(figsize=(5, 3))

filtered_dataset["Sex"].value_counts().plot(kind='bar')

plt.xlabel('Age')

plt.ylabel('Sex')

plt.show()
filtered_dataset["Race"].value_counts().plot(kind='bar')

plt.xlabel('Race')

plt.ylabel('Sex')

plt.show()



print("\n\n\nComposição étnica percentual da população estudada:")

100*filtered_dataset["Race"].value_counts()/filtered_dataset["Race"].value_counts().sum()
del filtered_dataset["Race"]
pd.crosstab(filtered_dataset["Age"], filtered_dataset["Target"], normalize='index').plot.bar(stacked=True, figsize=(15, 5))

plt.title("Porcentagem por idade dos indivíduos que atingem o \"Target\"")

plt.show()
pd.crosstab(filtered_dataset["Education"], filtered_dataset["Target"], normalize='index').plot.bar()

plt.title("Salário em função da educação (em porcentagem)")

plt.show()



pd.crosstab(filtered_dataset["Education num"], filtered_dataset["Target"], normalize='index').plot.bar()

plt.show()
del filtered_dataset["Education"]
pd.crosstab(filtered_dataset["fnlwgt"], filtered_dataset["Target"]).plot()

plt.show()
pd.crosstab(filtered_dataset["Marital status"], filtered_dataset["Target"], normalize='index').plot.bar()

plt.title("Salário em função do estado civil (em porcentagem)")

plt.show()



pd.crosstab(filtered_dataset["Relationship"], filtered_dataset["Target"], normalize='index').plot.bar()

plt.show()
pd.crosstab(filtered_dataset["Sex"], filtered_dataset["Target"]).plot.bar()

plt.title("Salário em função do sexo")

plt.show()



pd.crosstab(filtered_dataset["Sex"], filtered_dataset["Education num"], normalize='index').sort_index().plot.bar(stacked=True)

plt.title("Nível de educação em função do sexo")

plt.legend(loc='right')

plt.show()
pd.crosstab(filtered_dataset["Hours per week"], filtered_dataset["Target"]).plot()

plt.title("Salário em função da jornada de trabalho")

plt.show()
pd.crosstab(filtered_dataset["Capital gain"], filtered_dataset["Target"]).plot()

plt.title("Salário em função do ganho de capital")

plt.show()



pd.crosstab(filtered_dataset["Capital loss"], filtered_dataset["Target"]).plot()

plt.title("Salário em função da perda de capital")

plt.show()
from sklearn import preprocessing
filtered_dataset.iloc[:, [2, 8, 9]] = preprocessing.scale(filtered_dataset.iloc[:, [2, 8, 9]])

filtered_dataset.describe()
filtered_dataset.head()
filtered_dataset.iloc[:, [4, 5, 6, 7, 11]] = filtered_dataset.iloc[:, [4, 5, 6, 7, 11]].apply(preprocessing.LabelEncoder().fit_transform)

filtered_dataset.head()
testset = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv',

                      header=0,

                      names = ['Id', 'Age', 'Workclass', 'fnlwgt', 'Education','Education num', 'Marital status',

                               'Occupation', 'Relationship', 'Race', 'Sex', 'Capital gain', 'Capital loss',

                               'Hours per week', 'Country'],

                      sep=',',

                      engine = 'python',

                      na_values = '?')



del testset["Id"], testset["Education"], testset["Race"], testset["Country"]
testset.isnull().sum()
print("Workclass:")

print(testset["Workclass"].describe())

print("{0}% dos dados tem como valor \"{1}\"".format(

    np.round(100*testset["Workclass"].describe().freq/(testset.shape[0] - testset.isnull().sum()["Workclass"]), 2),

    testset["Workclass"].describe().top))



print("\nOccupation:")

print(testset["Occupation"].describe())

print("{0}% dos dados tem como valor \"{1}\"".format(

    np.round(100*testset["Occupation"].describe().freq/(testset.shape[0] - testset.isnull().sum()["Occupation"]), 2),

    testset["Occupation"].describe().top))
testset["Workclass"] = testset["Workclass"].fillna("Private")

testset["Occupation"] = testset["Occupation"].fillna("Prof-specialty")

testset.isnull().sum()
testset.head()
workclassToNumber(testset)

testset.iloc[:, [2, 8, 9]] = preprocessing.scale(testset.iloc[:, [2, 8, 9]])

testset.iloc[:, 4:8] = testset.iloc[:, 4:8].apply(preprocessing.LabelEncoder().fit_transform)

testset.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
xTrain = filtered_dataset.iloc[:, :11]

yTrain = filtered_dataset["Target"]
accuracy = cross_val_score(knn, xTrain, yTrain, cv=10)

print(accuracy)

accuracy.mean()
xTrain = filtered_dataset.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]]

accuracy = cross_val_score(knn, xTrain, yTrain, cv=10)

print(accuracy)

accuracy.mean()
xTrain = filtered_dataset.loc[:,["Workclass", "Education num", "Occupation",

                                  "Relationship", "Sex", "Capital gain", "Capital loss"]]

accuracy = cross_val_score(knn, xTrain, yTrain, cv=40)

print(accuracy)

accuracy.mean()
kList = []

accList = []

for k in range(1, 30):

    knn = KNeighborsClassifier(n_neighbors=k)

    accuracy = cross_val_score(knn, xTrain, yTrain, cv=30)

    kList.append(k)

    accList.append(accuracy.mean())
plt.plot(kList, accList)

plt.xlabel("Número de vizinhos")

plt.ylabel("Acurácia média da validação cruzada")

plt.show()
print(kList[accList.index(max(accList))])

print(max(accList))
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(xTrain, yTrain)
test = testset.loc[:, ["Workclass", "Education num", "Occupation", "Relationship", "Sex", "Capital gain", "Capital loss"]]
yPredict = knn.predict(test)
predicted_labels = np.where(yPredict == 0, '<=50K', '>50K')
submission = pd.DataFrame({'income':predicted_labels})
submission
submission.to_csv('submission.csv', index=True, index_label='Id')