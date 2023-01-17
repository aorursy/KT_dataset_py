# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# We are reading our data That we Uploaded 

df = pd.read_csv("../input/heart-dissease/heart_Disease.csv")
#Displaying Data Records

df
#Displaying First Five Records

df.head()
#Displaying First Five Records

df.tail()
#Displaying Count Of Each Instance in Dataset

df.count()
#Displaying Count Of Null Values ( if any ):

df.isna().sum()
#Summary Of Dataset 

df.describe()
#Displaying Count Of Patients Have / Not Haven’t Heart Diseases

df.target.value_counts()
#BarPlot For Count Of Patients Have / Not Haven’t Heart Diseases

sns.countplot(x="target", data=df, palette="bwr")

plt.show()
#Displaying Percentage Of Patients Having / Not Having Heart Diseases

countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
#Displaying Percentage Of Male/Female Patients

countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
#BarPlot For Count Based Of Sex( Male/Female )

sns.countplot(x='sex', data=df, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
#Histogram for Dataset

df.plot.hist()
#BoxPlot  for Dataset

df.plot.box()
#Histogram for Heart Disease Frequency According to  Chest Pain (CP) Type 

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#aa4711','#11aa93' ])

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
#Scatterplot for Maximum Heart Rate According Disease/Not Disease 

plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
#Histogram for Heart Disease Frequency For Ages

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
y = df.target.values

x_data = df.drop(['target'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
#We will split our data. 80% of our data will be train data and 20% of it will be test data.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#Transpose Matrices for Avoiding Confusion

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
#Implementing K-Nearest Neighbour (KNN) Model Classification :

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# Trying to find Best Feasible K Value

accuracies = {}

scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(x_train.T, y_train.T)

    scoreList.append(knn2.score(x_test.T, y_test.T))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()

#Accuracy Predictor

acc = max(scoreList)*100

accuracies['KNN'] = acc

print("Maximum KNN Score is {:.2f}%".format(acc))
#Implementing Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)

#Accuracy Predictor

acc = nb.score(x_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
#Implementing Decision Tree Model 

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)

#Accuracy Predictor

acc = dtc.score(x_test.T, y_test.T)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
#Comparing The Proposed Model Classifications

colors = ["red","blue","green"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()