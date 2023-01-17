import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},).drop('Ticket', 1).drop('Name', 1).drop('Cabin', 1).drop('PassengerId', 1)

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},).drop('Ticket', 1).drop('Name', 1).drop('Cabin', 1).drop('PassengerId', 1)



# preview the data

titanic_df.head()
# fiting the missing point

age_avg = titanic_df.Age.mean()

age_std = titanic_df.Age.std()

print(age_avg, age_std)



titanic_df.Age.fillna(np.random.normal(loc=age_avg, scale=age_std, size=1)[0], inplace=True)

titanic_df.Embarked.fillna('S', inplace=True)
# decision tree training

X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.copy()



#clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

#clf = clf.fit(training_feature, training_label)

print(X_train)