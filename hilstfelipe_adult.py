"""

PMR3508 - Tarefa 1 - ADULT UCI - P

"""

import pandas as pd
import sklearn 
import statistics
testadult = pd.read_csv("../input/adultt/test_data.csv",
            sep = r'\s*,\s*',
            engine = 'python',
            na_values = "?")
testadult.shape
testadult.head()
trainadult = pd.read_csv("../input/adultt/train_data.csv",
            sep = r'\s*,\s*',
            engine = 'python',
            na_values = "?")
trainadult.shape
trainadult.head()
ntrainadult = trainadult.dropna()
Xtrain = ntrainadult[["age","education.num","capital.gain","capital.loss","hours.per.week"]]
Ytrain = ntrainadult.income
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
means =[]
for num in range(1,31):
    knn = KNeighborsClassifier(n_neighbors = num)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    mean = statistics.mean(scores)
    means.append(mean) 
bestn = means.index(max(means))+1
bestn

knn = KNeighborsClassifier(n_neighbors = bestn)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
statistics.mean(scores)
knn.fit(Xtrain, Ytrain)
Xtest = testadult[["age","education.num","capital.gain","capital.loss","hours.per.week"]]
Ypred = knn.predict(Xtest)
Ypred
file = open("submit_adult.csv","w")
file.write("Id,income\n")
for i in range(len(Ypred)):
    file.write(str(i))
    file.write(",")
    file.write(Ypred[i])
    file.write("\n")
file.close()