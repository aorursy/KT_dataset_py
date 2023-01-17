#Libraries



import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns
adult_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"], sep=r'\s*,\s*', engine='python', na_values="?")
adult_data.describe() # checando se está tudo dentro do esperado
adult_data.shape
list(adult_data)
adult_data.head()
adult_data["Occupation"].value_counts()
#White frame aroung plot (avoid readabillity problems when using notebook dark mode)

plt.figure(facecolor='white')



adult_data["Occupation"].value_counts().plot(kind="bar")
#White frame aroung plot (avoid readabillity problems when using notebook dark mode)

plt.figure(facecolor='white')



adult_data["Sex"].value_counts().plot(kind="pie")
#White frame aroung plot (avoid readabillity problems when using notebook dark mode)

plt.figure(facecolor='white')



adult_data["Education"].value_counts().plot(kind="bar")
#White frame aroung plot (avoid readabillity problems when using notebook dark mode)

plt.figure(facecolor='white')



adult_data["Occupation"].value_counts().plot(kind="bar")
nadult_data = adult_data.dropna()
nadult_data
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
nTestAdult = testAdult.dropna()
nTestAdult.describe()
nTestAdult.head()
Xadult_data = nadult_data[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult_data = nadult_data.Target
Xtest_adult_data = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Ytest_adult_data = nTestAdult.Target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
#scores = cross_val_score(knn, Xadult_data, Yadult_data, cv=10)
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
numAdult = nadult_data.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult_data = numAdult.iloc[:,0:14]
Yadult_data = numAdult.Target
Xtest_adult_data = numTestAdult.iloc[:,0:14]
Ytest_adult_data = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult_data,Yadult_data)
Ytest_pred = knn.predict(Xtest_adult_data)
accuracy_score(Ytest_adult_data,Ytest_pred)
Xadult_data = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
Xtest_adult_data = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult_data,Yadult_data)
Ytest_pred = knn.predict(Xtest_adult_data)
accuracy_score(Ytest_adult_data,Ytest_pred)
Xadult_data = numAdult[["Age", "Workclass", "Education-Num", 

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
Xtest_adult_data = numTestAdult[["Age", "Workclass", "Education-Num", 

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult_data,Yadult_data)
Ytest_pred = knn.predict(Xtest_adult_data)
accuracy_score(Ytest_adult_data,Ytest_pred)