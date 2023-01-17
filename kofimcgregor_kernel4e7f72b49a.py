# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import numpy as np

import pandas

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sys



import sklearn

from sklearn import preprocessing

import csv

import warnings

warnings.filterwarnings('ignore')



def categoryEncode(column):

    column = pd.factorize(column)[0]

    return column



# Load CSV from URL using Pandas from URL

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"



#when reading set missing values to na_values

census = pd.read_csv("../input/adult-census-income/adult.csv", na_values = [" ?", "?"])

censusTest = pd.read_csv("../input/testing/adultTest.csv", skiprows=1, na_values = [" ?", "?"])



#Setting the column names

column_names = ["age", "workclass" , "fnlwgt" , "education"

                , "education.num" , "marital.status", "occupation"

                , "relationship", "race" , "sex", "capital.gain"

                , "capital.loss", "hours.per.week", "native.country"

                , "income"]



censusTest.columns = (column_names)
#Inspecting the data before cleaning

print(census.shape)

print(censusTest.shape)

print()

print(census.info())

print()

print(census.isnull().sum())
##Data Pre-Processing



#Drop columns

#fnlwgt provids nothing that would be of use to the modeling

#education eduaction number holds the nemerical reprsentaiton of the education column, making including it itself redundent

#capital gain an loss contain barly any no zero values therfore if left in will heavily sqew the modeling

census.drop(['fnlwgt', 'education', 'capital.gain', 'capital.loss'], axis=1, inplace=True)

censusTest.drop(['fnlwgt', 'education', 'capital.gain', 'capital.loss'], axis=1, inplace=True)



#Dropping rows that have missing values

dataset = (census.dropna( axis=0, how='any' ))

testData = (censusTest.dropna( axis=0, how='any' ))



#matching the way the testdata and the datas income column removing full stops and extra spaces

dataset["income"] = dataset["income"].replace(" <=50K","<=50K")

dataset["income"] = dataset["income"].replace(" >50K",">50K")

testData["income"] = testData["income"].replace(" <=50K.","<=50K")

testData["income"] = testData["income"].replace(" >50K.",">50K")
#combining the 2 databases for a larger sample size and so I can vary the test data percentage later

Xset = dataset.append(testData, ignore_index = True)

#print(Xset.head())
#Inspecting the now cleaned and combined datasets

print(dataset.shape)

print(testData.shape)

print()

print(dataset.info())

print()

print(dataset.isnull().sum())
#encodes the carogrical data

XsetNoEnCode = Xset.copy()



columnsToEncode = ["workclass" , "marital.status", "occupation",

                   "relationship", "race" , "sex", "native.country", "income"]



for column in columnsToEncode:

    Xset[column] = categoryEncode(Xset[column])



print(Xset.head())
#Data is now ready for graphing, Scaling and modeling
#Checking each feild against each feild to look for trends or patterns

print(sns.pairplot(Xset))
#Age Box Plot

boxplot = sns.boxplot(x='age', 

                      data=Xset, 

                      width=0.2)

boxplot.set_xlabel = "Age"

boxplot.set_title('Age boxplot')

plt.show()
#Hours Box Plot

boxplot = sns.boxplot(x='hours.per.week', 

                      data=Xset, 

                      width=0.2)

boxplot.set_xlabel = "Hours"

boxplot.set_title('Hours boxplot')

plt.show()
#Pie Charts

labels = ['Above 50K', 'Below 50K']

colours = ['red', 'blue']



#Split the datatsets

HIS = XsetNoEnCode[XsetNoEnCode["income"] == ">50K"] #High income split

LIS = XsetNoEnCode[XsetNoEnCode["income"] == "<=50K"] #low income split



occupations = HIS['occupation'].unique()



for occup in occupations:

    highCount = len(HIS[HIS['occupation'] == occup])

    lowCount = len(LIS[LIS['occupation'] == occup])  

    

    counts = [highCount, lowCount]

    plt.pie(counts, labels=labels, colors= colours,

    autopct='%1.1f%%', shadow=True, startangle=0)

    plt.title('Pie Chart showing ratio of people who work in/as ' + occup + ' who earn above or below $50k')

    plt.axis('equal')

    plt.show()
#Age Income Box Plot

boxplot = sns.boxplot(y='age', x='income', data=XsetNoEnCode, width=0.5)     

boxplot.set_xlabel = "Age"

boxplot.set_ylabel = "Income"

boxplot.set_title('Age against Income boxplot')

plt.show()
Xset.plot(x='workclass', y='education.num', kind='scatter')

plt.show()
#Clustering age vs hours



#Split the datatsets

HIS = Xset[Xset["income"] == 1] #High income split

LIS = Xset[Xset["income"] == 0] #low income split



plt.scatter(HIS["age"], HIS["hours.per.week"])

#plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=4)





y_km= km.fit_predict(HIS[["age", "hours.per.week"]])

HIS.loc[:, "Cluster"] = y_km



HISc1 = HIS[HIS.Cluster == 0]

HISc2 = HIS[HIS.Cluster == 1]

HISc3 = HIS[HIS.Cluster == 2]



plt.scatter(HISc1.age, HISc1["hours.per.week"],color="blue")

plt.scatter(HISc2.age, HISc2["hours.per.week"],color="red")

plt.scatter(HISc3.age, HISc3["hours.per.week"],color="green")



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],

            color="black",

            marker="*",label="centroid")



#print(km.cluster_centers_)

plt.xlabel("age")

plt.ylabel("hoursPerWeek")

plt.title('Clustering The High Income Split with age against Hours Per Week')

plt.show()



plt.scatter(LIS["age"], LIS["hours.per.week"])

#plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=4)





y_km= km.fit_predict(LIS[["age", "hours.per.week"]])

LIS.loc[:, "Cluster"] = y_km



LISc1 = LIS[LIS.Cluster == 0]

LISc2 = LIS[LIS.Cluster == 1]

LISc3 = LIS[LIS.Cluster == 2]



plt.scatter(LISc1.age, LISc1["hours.per.week"],color="blue")

plt.scatter(LISc2.age, LISc2["hours.per.week"],color="red")

plt.scatter(LISc3.age, LISc3["hours.per.week"],color="green")



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],

            color="black",

            marker="*",label="centroid")



#print(km.cluster_centers_)

plt.xlabel("age")

plt.ylabel("hoursPerWeek")

plt.title('Clustering The Low Income Split with age against Hours Per Week')

plt.show()
#Clustering age vs education



#Split the datatsets

HIS = Xset[Xset["income"] == 1] #High income split

LIS = Xset[Xset["income"] == 0] #low income split



plt.scatter(HIS["age"], HIS["education.num"])

#plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=6)





y_km= km.fit_predict(HIS[["age", "education.num"]])

HIS.loc[:, "Cluster"] = y_km



HISc1 = HIS[HIS.Cluster == 0]

HISc2 = HIS[HIS.Cluster == 1]

HISc3 = HIS[HIS.Cluster == 2]

HISc4 = HIS[HIS.Cluster == 3]

HISc5 = HIS[HIS.Cluster == 4]



plt.scatter(HISc1.age, HISc1["education.num"],color="blue")

plt.scatter(HISc2.age, HISc2["education.num"],color="red")

plt.scatter(HISc3.age, HISc3["education.num"],color="green")

plt.scatter(HISc4.age, HISc4["education.num"],color="yellow")

plt.scatter(HISc5.age, HISc5["education.num"],color="pink")



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],

            color="black",

            marker="*",label="centroid")



#print(km.cluster_centers_)

plt.xlabel("age")

plt.ylabel("education.num")

plt.title('Clustering The High Income Split with age against Eduaction num')

plt.show()



plt.scatter(LIS["age"], LIS["education.num"])

#plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=6)





y_km= km.fit_predict(LIS[["age", "education.num"]])

LIS.loc[:, "Cluster"] = y_km



HISc1 = LIS[LIS.Cluster == 0]

HISc2 = LIS[LIS.Cluster == 1]

HISc3 = LIS[LIS.Cluster == 2]

HISc4 = LIS[LIS.Cluster == 3]

HISc5 = LIS[LIS.Cluster == 4]



plt.scatter(HISc1.age, HISc1["education.num"],color="blue")

plt.scatter(HISc2.age, HISc2["education.num"],color="red")

plt.scatter(HISc3.age, HISc3["education.num"],color="green")

plt.scatter(HISc4.age, HISc4["education.num"],color="yellow")

plt.scatter(HISc5.age, HISc5["education.num"],color="pink")



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],

            color="black",

            marker="*",label="centroid")



#print(km.cluster_centers_)

plt.xlabel("age")

plt.ylabel("education.num")

plt.title('Clustering The Low Income Split with age against Eduaction num')

plt.show()
#Modeling building and testing
XtoSplit = Xset.drop(["income"], axis=1)

YtoSplit = Xset["income"]
#Standard scaling normalization

scaler = preprocessing.StandardScaler()

XtoSplitScaledStandard = pd.DataFrame(scaler.fit_transform(XtoSplit), columns = XtoSplit.columns)



#XtoSplitScaledStandard.head()
#MinMax scaling normalization

scaler = preprocessing.MinMaxScaler()

XtoSplitScaledMinMax = pd.DataFrame(scaler.fit_transform(XtoSplit), columns = XtoSplit.columns)



#XtoSplitScaledMinMax.head()
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
#Logistic Regression

from sklearn.linear_model import LogisticRegression



Xtrain, Xtest, Ytrain, Ytest = train_test_split(XtoSplit, YtoSplit, test_size = 0.2, random_state = 0)



logreg = LogisticRegression()

logreg.fit(Xtrain, Ytrain)

Ypred = logreg.predict(Xtest)



print('Logistic Regression accuracy score: {0:0.4f}%'. format(accuracy_score(Ytest, Ypred)*100))
print(confusion_matrix(Ytest, Ypred))

print(classification_report(Ytest, Ypred))
#Nearest Neighbors algorithm

from sklearn.neighbors import KNeighborsClassifier



Xtrain, Xtest, Ytrain, Ytest = train_test_split(XtoSplitScaledMinMax, YtoSplit, test_size = 0.2, random_state = 0)



KNN = KNeighborsClassifier(n_neighbors=75)

KNN.fit(Xtrain, Ytrain)

Ypred = KNN.predict(Xtest)



print('Nearest Neighbors algorithm accuracy score: {0:0.4f}%'. format(accuracy_score(Ytest, Ypred)*100))
print(confusion_matrix(Ytest, Ypred))

print(classification_report(Ytest, Ypred))
#Decision Tree classifier

from sklearn.tree import DecisionTreeClassifier



Xtrain, Xtest, Ytrain, Ytest = train_test_split(XtoSplit, YtoSplit, test_size = 0.2, random_state = 0)



DT = DecisionTreeClassifier(criterion="entropy",  max_depth=7)

DT.fit(Xtrain, Ytrain)

Ypred = DT.predict(Xtest)

print('Decision Tree classifier accuracy score: {0:0.4f}%'. format(accuracy_score(Ytest, Ypred)*100))
print(confusion_matrix(Ytest, Ypred))

print(classification_report(Ytest, Ypred))
#Random Forest

from sklearn.ensemble import RandomForestClassifier



Xtrain, Xtest, Ytrain, Ytest = train_test_split(XtoSplit, YtoSplit, test_size = 0.2, random_state = 0)

 

RF = RandomForestClassifier(criterion = 'gini', max_depth = 10, max_features = "sqrt", min_samples_split = 5, n_estimators= 500)

RF.fit(Xtrain, Ytrain)

Ypred = RF.predict(Xtest)

print('Random Forest classifier accuracy score: {0:0.4f}%'. format(accuracy_score(Ytest, Ypred)*100))
print(confusion_matrix(Ytest, Ypred))

print(classification_report(Ytest, Ypred))