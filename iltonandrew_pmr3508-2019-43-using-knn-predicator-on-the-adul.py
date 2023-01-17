import pandas as pd

import sklearn

import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adultbasefiles/adult.data", 

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape
adult.head()
adult['Country'].value_counts()
adult["Sex"].value_counts()
adult["Race"].value_counts()
adult["Age"].value_counts()
adult['Age'].max()
adult["Age"].min()
# We need to clean the dataset removing rows with missing information

cleanAdult = adult.dropna()
cleanAdult.shape
# Lets get our test dataset cleaned too

testAdult = pd.read_csv("../input/adultbasefiles/adult.test",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

cleanTestAdult = testAdult.dropna()
Xadult = cleanAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult = cleanAdult.Target



XtestAdult = cleanTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YtestAdult = cleanTestAdult.Target

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
from sklearn.metrics import accuracy_score



accuracy_score(YtestAdult,YtestPred)
YtestPred
YtestAdult
#Searched for the solution for the 0.0 score and found it

YtestAdult = YtestAdult.values



for i in range (len(YtestAdult)):

    YtestAdult[i] = YtestAdult[i][:-1]
accuracy_score(YtestAdult,YtestPred) #voilà
def testAdultBase(keyColumns=list,nbNeighborsKNN=int,indexColumnsToConvert=[],returnPredic=False):

    

    adult = pd.read_csv("../input/adultbasefiles/adult.data",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")

    testAdult = pd.read_csv("../input/adultbasefiles/adult.test",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")

    

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
keyColumns = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]

accuracyList=[]

listI=[]
for i in range (5,31):

    accuracyList.append(testAdultBase(keyColumns,i))

    listI.append(i)
maxi = max(accuracyList)
#making it visual

plt.title('Accuracy of the KNN prediction depending on number of neighbors.')

plt.plot(listI,accuracyList)
print(listI[accuracyList.index(maxi)])

print(maxi)
finalYPred = testAdultBase(keyColumns,16,[2],True)

Id = [i for i in range(len(finalYPred))]



d = {'Id' : Id, 'Income' : finalYPred}

myDf = pd.DataFrame(d) 

myDf.to_csv('bestPrediction.csv',

             index=False, sep=',', line_terminator = '\n', header = ["Id", "Income"])
import warnings

import numpy as np

import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer

from scipy.stats import pearsonr



import matplotlib.pyplot as plt

%matplotlib inline
train_raw = pd.read_csv("../input/costarican/train.csv")
train_raw.head()
test_raw = pd.read_csv("../input/costarican/test.csv")
test_raw.head()
cols_with_na = train_raw.columns[train_raw.isnull().any(axis=0)]

cols_with_na
train_raw[cols_with_na].isnull().sum(axis=0) / train_raw.shape[0]
def preprocess(data):

    data = data.copy()

    dep = data['dependency'].copy()

    dep[dep == 'no'] = 0

    dep[(dep != 0) & (~dep.isnull())] = 1

    data['dependency'] = pd.to_numeric(dep)

    

    for col in ['edjefe', 'edjefa']:

        edjef = data[col].copy()

        edjef[edjef == 'yes'] = 1

        edjef[edjef == 'no'] = 0

        data[col] = pd.to_numeric(edjef)



    return data
preprocess(train_raw).select_dtypes(exclude=[np.number]).columns
train = preprocess(train_raw)

test = preprocess(train_raw)

numeric_columns = list(train.select_dtypes(include=[np.number]).columns)

columns = list(set(numeric_columns) - {'v2a1', 'v18q1', 'rez_esc', 'Target'})

train_initial = train.copy()[columns + ['Target']]

train_initial.dropna(inplace=True)

x = train_initial[columns]

y = train_initial['Target']

def cross_val(knn, x, y, cv, scoring='f1_macro'):

    scores = cross_val_score(knn, x, y, cv=cv, scoring=scoring)

    return sum(scores)/len(scores)
knn = KNeighborsClassifier(n_neighbors=30, p=2)

print('Accuracy:', cross_val(knn, x, y, cv=5, scoring='accuracy'))

print('F1:', cross_val(knn, x, y, cv=5))
x = Imputer().fit_transform(train[columns])

y = train['Target']



print('Accuracy:', cross_val(knn, x, y, cv=5, scoring='accuracy'))

print('F1:', cross_val(knn, x, y, cv=5))