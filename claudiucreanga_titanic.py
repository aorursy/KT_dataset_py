# load the data

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline



X = pd.read_csv("../input/train.csv")

X_test = pd.read_csv("../input/test.csv")
X.describe()
# check null values



null_columns=X.columns[X.isnull().any()]

X.isnull().sum()
# title from name looks interesting because you can get if the person is married or not and guess their age if it is not known



import re



#A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



X["Title"] = X["Name"].apply(get_title)

X_test["Title"] = X_test["Name"].apply(get_title)



X["Title"].value_counts()
# We can see here that most people with Mr in their title died while Miss and Mrs survived



title_survive = X[["Title", "Survived"]]

title_survive_transformed = pd.get_dummies(title_survive, columns=["Title"])



bar = title_survive_transformed.groupby("Survived").apply(lambda column: column.sum()).transpose().drop(["Survived"])

bar.columns = ["Died","Survived"]

bar.plot.bar()
# you can see that you had a greater chance to survive if you were in embarked C or Q



embarked_survive = X[["Survived", "Embarked"]]

embarked_survive_transformed = pd.get_dummies(embarked_survive, columns=["Embarked"])



e_bar = embarked_survive_transformed.groupby("Survived").apply(lambda column: column.sum()).transpose().drop(["Survived"])

e_bar.columns = ["Died","Survived"]

e_bar.plot.bar()
X["FamilySize"] = 1 + X["SibSp"] + X["Parch"]

X_test["FamilySize"] = 1 + X_test["SibSp"] + X_test["Parch"]

family_size = X["FamilySize"].apply(lambda row: "Single" if row == 1 else ("Large" if row < 5 else "Extreme"))     

family_size_test = X_test["FamilySize"].apply(lambda row: "Single" if row == 1 else ("Large" if row < 5 else "Extreme"))     

X["FamilySize"] = family_size



family_size = pd.DataFrame(family_size)

family_size["Survived"] = X["Survived"]

family_size_transformed = pd.get_dummies(family_size, columns=["FamilySize"])



X_test["FamilySize"] = family_size_test



f_bar = family_size_transformed.groupby("Survived").apply(lambda column: column.sum()).transpose().drop(["Survived"])

f_bar.columns = ["Died","Survived"]

f_bar.plot.bar()
# fill NaN values with mean so that we can do transformations



X.fillna(X.mean(), inplace=True)

X_test.fillna(X_test.mean(), inplace=True)

X.head()
# Age and Fares are on different scales, so let's scale them



from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit_transform(X[['Age', 'Fare']])

X[["Age", "Fare"]] = std_scale

std_scale_test = preprocessing.StandardScaler().fit_transform(X_test[['Age', 'Fare']])

X_test[["Age", "Fare"]] = std_scale_test

std_scale
# transform form categorical to numerical



X_transformed = pd.get_dummies(X, columns = ["Sex", "FamilySize", "Cabin", "Title", "Embarked"])

X_test_transformed = pd.get_dummies(X_test, columns = ["Sex", "FamilySize", "Cabin", "Title", "Embarked"])
X_transformed.head()
# correlations



corr_matrix = X_transformed.corr()

corr_matrix["Survived"].sort_values(ascending=False)
# remove columns that offer little help and the labels



y = X_transformed["Survived"]

X_fewer_columns = X_transformed.drop(["Survived", "Name", "Ticket", "PassengerId"], axis=1).copy()

X_test_fewer_columns = X_test_transformed.drop(["Name", "Ticket", "PassengerId"], axis=1).copy()
# Stochastic Gradient Descent Classifier



from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)

X_matrix = X_fewer_columns.as_matrix()

y_matrix = y.as_matrix()

sgd_clf.fit(X_matrix, y_matrix)
# display all scores in one go



from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve



def plot_roc_curve(fpr, tpr, **options):

    plt.plot(fpr, tpr, linewidth=2, **options)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    

def display_all_scores(model, X):

    y_train_predictions = cross_val_predict(model, X, y_matrix, cv = 3)

    print("Scores for model:",model.__class__.__name__)

    print("Confusion metrics:", confusion_matrix(y_matrix, y_train_predictions))

    print("Precision score:", precision_score(y_matrix, y_train_predictions))

    print("Recall score:", recall_score(y_matrix, y_train_predictions))

    print("F1 score:", f1_score(y_matrix, y_train_predictions))

   
display_all_scores(sgd_clf, X_matrix)
# let's see how we do if we remove more columns that do not look interesting



remove_some_cabins = [c for c in X_fewer_columns.columns 

                      if c[:6] != "Cabin_" 

                      and c != "Parch" 

                      and c != "SibSp" 

                      and c != "Title_Major"

                      and c != "Title_Rev"

                      and c != "Title_Sir"

                      and c != "Title_Jonkheer"

                      and c != "Title_Dr"

                      and c != "Title_Don"

                      and c != "Title_Countess"

                      and c != "Title_Col"

                      and c != "Title_Capt"

                      ]    

X_even_fewer_columns = X_fewer_columns[remove_some_cabins]

X_even_fewer_columns.columns
sgd_clf1 = SGDClassifier(random_state=42)

X_matrix = X_even_fewer_columns.as_matrix()

y_matrix = y.as_matrix()

sgd_clf1.fit(X_matrix, y_matrix)
# As you can see this score is worse then the previous one 



display_all_scores(sgd_clf1, X_matrix)
# Let's check the Random Forest and you can see that it fares better



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score



X_matrix = X_fewer_columns.as_matrix()

rf = RandomForestClassifier(n_jobs=2)

rf.fit(X_matrix, y_matrix) 



y_train_predictions = cross_val_predict(rf, X_matrix,y_matrix,cv=3)

scores = cross_val_score(rf, X_matrix, y_matrix, scoring='f1', cv=3)

print("F1 score for Random Forest", scores.mean())