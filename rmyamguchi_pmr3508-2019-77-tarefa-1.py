import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import statistics
import os

os.listdir('/kaggle/input/pmr3508-tarefa-1-3508-adult-dataset')
adult = pd.read_csv ("/kaggle/input/adult-data/train_data.csv",

                    names=[

                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                    "Hours per Week", "Country", "Target"],

                    skipinitialspace=True,

                    engine='python',

                    na_values="?",

                    skiprows=1)

adult.head()
adult.shape
print ("[Age]")

print (" Média: %.02f" %(statistics.mean(adult.iloc[:,0])))

print (" Mediana: %.02f" %(statistics.median(adult.iloc[:,0])))

print (" Moda: %.02f" %(statistics.mode(adult.iloc[:,0])))
adult["Workclass"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Martial Status"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
adult["Relationship"].value_counts().plot(kind="bar")
adult["Race"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")
print ("[Capital Gain]")

print (" Média: %.02f" %(statistics.mean(adult.iloc[:,10])))

print (" Mediana: %.02f" %(statistics.median(adult.iloc[:,10])))

print (" Moda: %.02f" %(statistics.mode(adult.iloc[:,10])))
print ("[Capital Loss]")

print (" Média: %.02f" %(statistics.mean(adult.iloc[:,11])))

print (" Mediana: %.02f" %(statistics.median(adult.iloc[:,11])))

print (" Moda: %.02f" %(statistics.mode(adult.iloc[:,11])))
adult["Country"].value_counts()
adult.mode()
nAdult = adult.fillna(adult.mode().iloc[0])

nAdult.shape
Xadult = nAdult[["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per Week"]]

Yadult = nAdult.Target
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier (n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores
from sklearn import preprocessing

numAdult = nAdult.apply (preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["Age", "Education-Num", "Workclass", "Martial Status", "Occupation",

                   "Relationship", "Capital Gain", "Capital Loss", "Hours per Week"]]

Yadult = nAdult.Target # Mantem o formato '<=50K'/'>50K'
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult,cv=10)

scores
#best_score = statistics.mean(scores)

#k = 0

#add = 30

#

#while add >= 1:

#    while True:

#        k += add

#        knn = KNeighborsClassifier(n_neighbors=k)

#        print ("Calculando para k=%d" %(k))

#        score = statistics.mean(cross_val_score(knn, Xadult, Yadult, cv=10))

#        if score > best_score:

#            print (" Bom. Score = %.04f" %(score))

#            best_score = score

#        else:

#            break

#    k -= add

#    add //= 2

#    

#print ("Melhor k=%d" %(k))

#print ("Média do melhor: %.04f" %(best_score))
knn = KNeighborsClassifier(n_neighbors=37)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores
testAdult = pd.read_csv ("/kaggle/input/adult-data/test_data.csv",

                         names=[

                         "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                         "Hours per Week", "Country"],

                         skipinitialspace=True,

                         engine='python',

                         na_values="?",

                         skiprows=1)

testAdult.head()
testAdult.mode()
nTestAdult = testAdult.fillna(testAdult.mode().iloc[0])

nTestAdult.shape
numTestAdult = nTestAdult.apply (preprocessing.LabelEncoder().fit_transform)

XtestAdult = numTestAdult[["Age", "Education-Num", "Workclass", "Martial Status", "Occupation",

                           "Relationship", "Capital Gain", "Capital Loss", "Hours per Week"]]
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)

YtestPred
Id = pd.DataFrame({'Id' : list(range(len(YtestPred)))})

income = pd.DataFrame({'income' : YtestPred})

result = Id.join(income)

result
result.to_csv (path_or_buf="/kaggle/input/submition.csv",

               index=False)

os.listdir('/kaggle/input')