import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt
adult = pd.read_csv('../input/adultbasefiles/adult.data.txt',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult.head()
adult["Country"].value_counts()
adult["Age"].value_counts()



print(adult["Age"].min())

print(adult["Age"].max())

#Let's "clean" the database by removing lines with any missing datas

bruteCleanedAdult = adult.dropna()
bruteCleanedAdult.shape
# Let's define our firsts key columns

keyColumns = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]
#Let's "clean" the database by removing lines with any missing key columns data

cleanedAdult = adult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)

print(cleanedAdult.shape)
testAdult = pd.read_csv("../input/adultbasefiles/adult.test.txt",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



print(testAdult.shape)

cleanedTestAdult = testAdult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)

print(cleanedTestAdult.shape)
#Let's separate the lists for knn tests



Xadult = cleanedAdult[keyColumns]

Yadult = cleanedAdult["Target"]

print(Xadult.shape)

print(Yadult.shape)



XtestAdult = cleanedTestAdult[keyColumns]

YtestAdult = cleanedTestAdult["Target"]

print(XtestAdult.shape)

print(YtestAdult.shape)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)

print(YtestPred.shape)

print(YtestAdult.shape)

from sklearn.metrics import accuracy_score



accuracy_score(YtestAdult,YtestPred)
print(YtestPred)

print(YtestAdult.values)
#I figured out there are points at the end of the target value which makes accuracy = 0

YtestAdult = YtestAdult.values



for i in range (len(YtestAdult)):

    YtestAdult[i] = YtestAdult[i][:-1]



print(YtestAdult)
accuracy_score(YtestAdult,YtestPred)
import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



def testAdultBase(keyColumns=list,nbNeighborsKNN=int,indexColumnsToConvert=[],returnPredic=False):

    

    adult = pd.read_csv("../input/adultbasefiles/adult.data.txt",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")

    testAdult = pd.read_csv("../input/adultbasefiles/adult.test.txt",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")

    

    cleanedAdult = adult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)

    cleanedTestAdult = testAdult.dropna(axis=0, how='any', subset=keyColumns, inplace=False)

    

    Xadult = cleanedAdult[keyColumns].values

    Yadult = cleanedAdult["Target"].values

    XtestAdult = cleanedTestAdult[keyColumns].values

    YtestAdult = cleanedTestAdult["Target"].values

    

    for i in range (len(YtestAdult)):

        YtestAdult[i] = YtestAdult[i][:-1]

    

    #let's convert the NaN columns into numbers, if necessary

    if indexColumnsToConvert != []:

        for col in indexColumnsToConvert:

            valueList=[]

            for i in range(len(Xadult)):

                value = Xadult[i][col]

                if value not in valueList:

                    valueList.append(value)

                Xadult[i][col] = valueList.index(value)

            for i in range(len(XtestAdult)):

                value = XtestAdult[i][col]

                if value not in valueList:

                    valueList.append(value)

                XtestAdult[i][col] = valueList.index(value)

    

    

    knn = KNeighborsClassifier(n_neighbors=nbNeighborsKNN)

    knn.fit(Xadult,Yadult)

    

    YtestPred = knn.predict(XtestAdult)

    

    if returnPredic==True:

        return YtestPred    

    return (accuracy_score(YtestAdult,YtestPred))

    
testAdultBase(["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"],3)
keyColumns = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]



accuracyList=[]

listI=[]
for i in range (5,31):

    accuracyList.append(testAdultBase(keyColumns,i))

    listI.append(i)
maxi = max(accuracyList)

print("For the key columns {},".format(keyColumns))

print("The best neighbors amount is {} with an accuracy of {}%".format(listI[accuracyList.index(maxi)],np.round(maxi,6)*100))



plt.title('Accuracy of the KNN prediction depending on number of neighbors.')

plt.plot(listI,accuracyList)
keyColumns2 = ["Age","Education-Num","Sex","Capital Gain", "Capital Loss", "Hours per week"]

accuracyList2=[]

listI2=[]
for i in range (10,41):

    accuracyList2.append(testAdultBase(keyColumns2,i,[2]))

    listI2.append(i)
maxi2 = max(accuracyList2)

print("For the key columns {},".format(keyColumns2))

print("The best neighbors amount is {} with an accuracy of {}%".format(listI2[accuracyList2.index(maxi2)],np.round(maxi2,6)*100))



plt.title('Accuracy of the KNN prediction depending on number of neighbors.')

plt.plot(listI2,accuracyList2)
keyColumns3 = ["Age","Education-Num","Sex","Capital Gain", "Hours per week"]

accuracyList3=[]

listI3=[]
for i in range (10,41):

    accuracyList3.append(testAdultBase(keyColumns3,i,[2]))

    listI3.append(i)
maxi3 = max(accuracyList3)

print("For the key columns {},".format(keyColumns3))

print("The best neighbors amount is {} with an accuracy of {}%".format(listI3[accuracyList3.index(maxi3)],np.round(maxi3,6)*100))



plt.title('Accuracy of the KNN prediction depending on number of neighbors.')

plt.plot(listI3,accuracyList3)
finalYPred = testAdultBase(keyColumns2,19,[2],True)

Id = [i for i in range(len(finalYPred))]



d = {'Id' : Id, 'Income' : finalYPred}

myDf = pd.DataFrame(d) 

myDf.to_csv('bestPrediction.csv',

             index=False, sep=',', line_terminator = '\n', header = ["Id", "Income"])

finalYPred