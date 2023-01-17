# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import dataset

churn_dataset = pd.read_csv("../input/Churn_Modelling.csv")
churn_dataset.head()
churn_dataset.info()
churn_dataset.describe()
#working with categorical features

#checking set of unique characters in each categorical feature

for col in churn_dataset.columns:

    if churn_dataset[col].dtypes == 'object':

        num_of_unique_cat = len(churn_dataset[col].unique())

        #checking the length of unique characters

        print("feature '{col_name}' has '{unique_cat}' unique categories".format(col_name = col, unique_cat=num_of_unique_cat))

#Deleting the surname feature

churn_dataset = churn_dataset.drop("Surname", axis=1)
# Creating a pivot table demonstrating the percentile

# Of different genders and geographical regions in exiting the bank 

visualization_1 = churn_dataset.pivot_table("Exited", index='Gender', columns='Geography')

visualization_1
churn_dataset = churn_dataset.drop(["Geography","Gender"], axis=1)
# Removing RowNumber and CustomerId features from the dataset

churn_dataset = churn_dataset.drop(['RowNumber',"CustomerId"], axis=1)
correlation = churn_dataset.corr()

sns.heatmap(correlation.T, square=True, annot=False, fmt="d", cbar=True)
#Splitting the dataset

churn_dataset = churn_dataset.reindex(np.random.permutation(churn_dataset.index))
#splitting feature data from the target

data = churn_dataset.drop("Exited", axis=1)

target = churn_dataset["Exited"]
#splitting feature data and target into training and testing 

X_train, X_test, y_train, y_test = train_test_split(data,target)
model = [GaussianNB(), KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=5, random_state=0,), LogisticRegression()]

model_names = ["Gaussian Naive bayes", "K-nearest neighbors", "Support vector classifier", "Decision tree classifier", "Random Forest", "Logistic Regression",]

for i in range(0, 6):

    y_pred = model[i].fit(X_train, y_train).predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)*100

    print(model_names[i], ":", accuracy, "%")
#working with the selection Model

model = RandomForestClassifier(n_estimators = 100, random_state = 0)

y_pred = model.fit(X_train, y_train).predict(X_test)

print("Our accuracy is:", accuracy_score(y_pred, y_test)*100, "%")