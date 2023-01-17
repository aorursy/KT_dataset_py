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
winequality = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
winequality.head()
winequality.info()
#SNS is very popular library in python, It is very easy to plot and infer relations between two parameters using this

import seaborn as sns 

import matplotlib.pyplot as plt
winequality.corr()
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(winequality.corr(), annot=True, fmt= '.1f',ax=ax)

plt.show()
fig = plt.figure(figsize = (20,6))

sns.regplot(x= winequality['alcohol'], y = winequality['quality'])
fig = plt.figure(figsize = (10,6)) 

sns.barplot(y= winequality['fixed acidity'], x = winequality['quality'])
fig = plt.figure(figsize = (20,6))

sns.barplot(x= winequality['quality'], y = winequality['sulphates'])
sns.countplot(winequality['quality'])
quality = winequality["quality"].values

category = []

for num in quality:

    if num<5:

        category.append("Bad")

    elif num == 5 or num == 6:

        category.append("Average")

    else:

        category.append("Good")

#Creating new dataset for prediction

category = pd.DataFrame(data=category, columns=["category"])

winedata = pd.concat([winequality,category],axis=1)

winedata.drop(columns="quality",axis=1,inplace=True)
winedata.head()
X= winedata.iloc[:,:-1].values

y= winedata.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder

labelencoder_y =LabelEncoder()

y= labelencoder_y.fit_transform(y)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier

random_result = RandomForestClassifier(n_estimators=250)

random_result.fit(X_train, y_train)

res_forest = random_result.predict(X_test)

print(classification_report(y_test, res_forest))
from sklearn.neighbors import KNeighborsClassifier

knn_result = KNeighborsClassifier()

knn_result.fit(X_train,y_train)

res_knn=knn_result.predict(X_test)

print(classification_report(y_test, res_knn))
from sklearn.linear_model import LogisticRegression

lr_result = LogisticRegression()

lr_result.fit(X_train, y_train)

res_logRes = lr_result.predict(X_test)

print(classification_report(y_test, res_logRes))
from sklearn.tree import DecisionTreeClassifier

DecTree_res = DecisionTreeClassifier()

DecTree_res.fit(X_train,y_train)

res_DecTree = DecTree_res.predict(X_test)

print(classification_report(y_test, res_DecTree))
from sklearn.naive_bayes import GaussianNB

NaiBay_res = GaussianNB()

NaiBay_res.fit(X_train,y_train)

res_NaiBay=NaiBay_res.predict(X_test)

print(classification_report(y_test, res_NaiBay))
final_result = pd.DataFrame({'models': ["Random Forest","KNN","LogisticRegression","DecisionTree", "NaiveBayes"],

                           'accuracy_score': [accuracy_score(y_test,res_forest),accuracy_score(y_test,res_knn), accuracy_score(y_test,res_logRes), 

                                              accuracy_score(y_test,res_DecTree), accuracy_score(y_test,res_NaiBay)]})
fig = plt.figure(figsize = (6,6))

sns.barplot(x= final_result['models'], y = final_result['accuracy_score'])