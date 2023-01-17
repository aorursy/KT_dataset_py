# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
flat=pd.read_csv("../input/flat-buyers-details-in-bangalore/Flat buyers.csv",parse_dates= ['DOB'])
flat.head()
flat.info()
flat.shape
flat.isnull().sum()
flat = flat.fillna(0)

flat = flat.fillna(flat.median())
flat.isnull().sum()
threshold = 0.7

flat = flat.loc[flat.isnull().mean(axis=1) < threshold]
flat.shape
flat.describe(include = 'all')
flat.dtypes
flat.columns
flat['BuyApartment'].value_counts()
sns.countplot(x="BuyApartment", data=flat, palette="bwr")

plt.show()
countNoBuying  = len(flat[flat.BuyApartment == 0])

countBuying = len(flat[flat.BuyApartment == 1])

print("Percentage of customers Haven't Bought flat: {:.2f}%".format((countNoBuying / (len(flat.BuyApartment))*100)))

print("Percentage of customers Have Bought flat: {:.2f}%".format((countBuying / (len(flat.BuyApartment))*100)))
sns.countplot(x='contribution', data=flat, palette="mako_r")

plt.xlabel("contribution(0 = Own, 1= Loan)")

plt.show()
sns.countplot(x='salaried', data=flat, palette="mako_r")

plt.xlabel("salaried(0 = No, 1= Yes)")

plt.show()
pd.crosstab(flat.AGE,flat.BuyApartment).plot(kind="bar",figsize=(20,6))

plt.title('Aprtment buying Frequency for Ages')

plt.xlabel('AGE')

plt.ylabel('Frequency')

plt.savefig('Buying flatAndAges.png')

plt.show()
pd.crosstab(flat.Income,flat.BuyApartment).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Buying apartment Frequency for income')

plt.xlabel('income (0 = High, 1 = Medium, 2 = Low)')

plt.xticks(rotation=0)

plt.legend(["Not Buying lat", "Buying Flat"])

plt.ylabel('Frequency')

plt.show()
a = pd.get_dummies(flat['Income'], prefix = "Income")

b = pd.get_dummies(flat['salaried'], prefix = "salaried")

c = pd.get_dummies(flat['Maritalstatus'], prefix = "Maritalstatus")

d = pd.get_dummies(flat['Country'], prefix = "Country")

e = pd.get_dummies(flat['vehical'], prefix = "vehical")

f = pd.get_dummies(flat['sourceenquiry'], prefix = "sourceenquiry")

g = pd.get_dummies(flat['contribution'], prefix = "contribution")
frames = [flat, a, b, c , d, e , f, g]

flats = pd.concat(frames, axis = 1)

flats.head()
flats2 = flats.drop(columns = ['Income', 'salaried', 'Maritalstatus','Country','vehical','sourceenquiry','contribution'])

flats2.head()
y = flats2.BuyApartment.values

x_data = flats2.drop(['BuyApartment','Name','DOB','AGE'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
accuracies = {}



lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

acc = lr.score(x_test.T,y_test.T)*100



accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# try to find best k value

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



acc = max(scoreList)*100

accuracies['KNN'] = acc

print("Maximum KNN Score is {:.2f}%".format(acc))
from sklearn.svm import SVC
svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)



acc = svm.score(x_test.T,y_test.T)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc = nb.score(x_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)



acc = dtc.score(x_test.T, y_test.T)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train.T, y_train.T)



acc = rf.score(x_test.T,y_test.T)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()
# Predicted values

y_head_lr = lr.predict(x_test.T)

knn3 = KNeighborsClassifier(n_neighbors = 3)

knn3.fit(x_train.T, y_train.T)

y_head_knn = knn3.predict(x_test.T)

y_head_svm = svm.predict(x_test.T)

y_head_nb = nb.predict(x_test.T)

y_head_dtc = dtc.predict(x_test.T)

y_head_rf = rf.predict(x_test.T)
from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test,y_head_lr)

cm_knn = confusion_matrix(y_test,y_head_knn)

cm_svm = confusion_matrix(y_test,y_head_svm)

cm_nb = confusion_matrix(y_test,y_head_nb)

cm_dtc = confusion_matrix(y_test,y_head_dtc)

cm_rf = confusion_matrix(y_test,y_head_rf)
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()

# Applied all the classifiation algortham to find the accuracy score of the customer who buys flat in bangalore & found Logistic regression have better accuracy and naiave bayes ahs the least accuarcy