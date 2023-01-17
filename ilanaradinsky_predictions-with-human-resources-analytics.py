# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#from IPython.core.interactiveshell import InteractiveShell

#InteractiveShell.ast_node_interactivity = "all"

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/HR_comma_sep.csv")

data.head()

data.info()
data["sales"].unique()

data["salary"].unique()
# separate test and train data

train = data[:7499]

test = data[7499:]
# create new features from "sales" category

sales_dummies_train = pd.get_dummies(train["sales"])

sales_dummies_test = pd.get_dummies(test["sales"])



train.drop(["sales"], axis = 1, inplace=True)

test.drop(["sales"], axis = 1, inplace=True)



train = train.join(sales_dummies_train)

test = test.join(sales_dummies_test)



train.head()

test.head()
# create new features from "salary" category

salary_dummies_train = pd.get_dummies(train["salary"])

salary_dummies_test = pd.get_dummies(test["salary"])



train = train.join(salary_dummies_train)

test = test.join(salary_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x="salary", data=train, ax=axis1, order=['low', 'medium', 'high'])



salary_avg = train[["salary", "left"]].groupby(['salary'], as_index=False).mean()

sns.barplot(x='salary', y='left', data=salary_avg, ax=axis2, order=['low', 'medium', 'high'])

train.drop(["salary"], axis = 1, inplace=True)

test.drop(["salary"], axis = 1, inplace=True)
# define training and testing sets



X_train = train.drop("left", axis=1)

Y_train = train["left"]

X_test = test.drop("left", axis=1).copy()

Y_test = test["left"]



X_train.info()

X_test.info()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



print(logreg.score(X_test, Y_test))
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



svc.score(X_train, Y_train)

svc.score(X_test, Y_test)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



random_forest.score(X_train, Y_train)

random_forest.score(X_test, Y_test)
knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, Y_train)



knn.score(X_test, Y_test)
# Gaussian Naive Bayes



gaussian = GaussianNB()



gaussian.fit(X_train, Y_train)



gaussian.score(X_train, Y_train)

gaussian.score(X_test, Y_test)
# Random Forest had the highest accuraccy for predicting if an employee was going to leave - 

# 0.995. Therefore, setting predictions model = random forests



predictions_model = random_forest
train_imp = train.drop("left", axis=1)



importances = random_forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(train_imp.shape[1]):

    print("%d. %s (%f)" % (f + 1, train_imp.columns[indices[f]], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure(figsize=(10, 5))

plt.title("Feature importances")

plt.bar(range(train_imp.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(train_imp.shape[1]), train_imp.columns[indices], rotation='vertical')

plt.xlim([-1, train_imp.shape[1]])

plt.show()