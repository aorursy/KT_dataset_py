#All import statements

import pandas as pd

import numpy as np

from IPython.display import display

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

#from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Read the csv file and create feature / label sets

X = pd.read_csv('../input/HR_comma_sep.csv')

display(X.head())



Y = X['left']



print('Number of records: ', X.shape[0])
X.describe()
X[X['left'] == 0].describe()
X[X['left'] == 1].describe()
#Plot 1: Left Vs Satisfaction level



#Frequency / Count 

X['satisfaction_level'].plot(kind='hist', figsize=(12,3),bins=100, xlim=(0,1))



# peaks for left/notleft by Satisfaction leve

facet = sns.FacetGrid(X, hue="left",aspect=3)

facet.map(sns.kdeplot,'satisfaction_level',shade= True)

facet.set(xlim=(0, 1))

facet.add_legend()
#Plot 2: Left Vs Avg. Monthly hours



#Frequency / Count 

X['average_montly_hours'].plot(kind='hist', figsize=(12,3),bins=100, xlim=(90,320))



# peaks for left/notleft by avg. hours

facet = sns.FacetGrid(X, hue="left",aspect=3)

facet.map(sns.kdeplot,'average_montly_hours',shade= True)

facet.set(xlim=(90, 320))

facet.add_legend()
#Plot 3: Left Vs Promotion

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(12,5))



#Count / Frequency

sns.countplot(x="promotion_last_5years", data=X, ax=axis1)



#Count of people with Promotions Vs Left

sns.countplot(x="promotion_last_5years", hue="left", data=X, ax=axis2)



#Mean Left Vs Promotion

promotion_data = X[["promotion_last_5years", "left"]].groupby(['promotion_last_5years'],as_index=False).mean()

sns.barplot(x="promotion_last_5years", y="left", data=promotion_data,ax=axis3)
X.dtypes
#Let's check the object / string columns: sales and salary for any null values



display(X.groupby(['sales']).size())

display(X.groupby(['salary']).size())



#No null / NAN values so no manipulation needed
#Understand the relationship between salary and left

sns.factorplot(x="salary", y="left", data=X, size=3.5, aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(12,5))



sns.countplot(x="salary", data=X, ax=axis1)

sns.countplot(x="salary", hue="left", data=X, ax=axis2)

salary_data = X[["salary", "left"]].groupby(['salary'],as_index=False).mean()

sns.barplot(x="salary", y="left", data=salary_data, order=['high','medium','low'],ax=axis3)
#Understand the relationship between sales and left

sns.factorplot(x="sales", y="left", data=X, size=3.5, aspect=3)
X = X.drop('left', axis = 1)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)

print('Training set volume:', X_train.shape[0])

print('Test set volume:', X_test.shape[0])
#Transpose for Salary and Sales



## Salary

salary_dummies_training = pd.get_dummies(X_train["salary"])

salary_dummies_test = pd.get_dummies(X_test["salary"])

#display(salary_dummies_training.head())

X_train = X_train.merge(salary_dummies_training, on=X_train.index.get_values(), how = 'left')

X_test = X_test.merge(salary_dummies_test, on=X_test.index.get_values(), how = 'left')



## Sales

sales_dummies_training = pd.get_dummies(X_train["sales"])

sales_dummies_test = pd.get_dummies(X_test["sales"])

#display(sales_dummies_training.head())

X_train = X_train.merge(sales_dummies_training, on=X_train.key_0, how = 'left')

X_test = X_test.merge(sales_dummies_test, on=X_test.key_0, how = 'left')

display(X_train.head())
X_train.dtypes

X_test.dtypes

X_train.drop(['salary'], axis=1, inplace=True)

X_test.drop(['salary'], axis=1, inplace=True)



X_train.drop(['sales_x'], axis=1, inplace=True)

X_test.drop(['sales_x'], axis=1, inplace=True)
#Confirm things are accurate

display(X_train.head())
# Create Naive Bayes classifier

clf_gb = GaussianNB()

clf_gb.fit(X_train, Y_train)

predicts_gb = clf_gb.predict(X_test)

print("GB Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(Y_test, predicts_gb))
#Create k-nn

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)

print("KNN5 Accuracy Rate, which is calculated by accuracy_score() is: %f" %accuracy_score(Y_test,Y_pred))
#Decision Tree

clf_dt = tree.DecisionTreeClassifier()

clf_dt.fit(X_train, Y_train)

predicts_dt = clf_dt.predict(X_test)

print("Decision tree Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(Y_test, predicts_dt))
#SVM -> takes a few seconds to run!

clf_svm = svm.SVC(kernel='rbf')

clf_svm.fit(X_train,Y_train)

predicts_svm = clf_svm.predict(X_test)

print("SVM Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(Y_test, predicts_svm))
#Random forest classifier

clf_rf = RandomForestClassifier(random_state = 0)

clf_rf.fit(X_train, Y_train)



accuracy_rf = clf_rf.score(X_test,Y_test)

print("Random Forest Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_rf)