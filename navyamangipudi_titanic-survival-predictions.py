

from sklearn.model_selection import train_test_split

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Here, I am importing the modules that I may need throughout the lab. I will, however, be importing some of the classifiers in

#the block with it's respective code in order for it to be clearer as to which module is used for which classifier. 



from sklearn import tree



import pandas as pd

import sys

import math as m

import numpy as np

import scipy as sp

import pandas as pd

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns  

import statsmodels.api as sm

import statsmodels.formula.api as smf
#Here, I am reading the file. I need to use "/kaggle/input/titanic/train.csv" instead of the csv name because here, the files are

#accesible from the input directory. 



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
#Using describe to take a look at the data. 

train_data.describe()
#Using .info() to look at the types and column names in the dataset as well as how many entries there are (to see if any columns

#are missing values). From this, we can see that the columns "Age", "Cabin" and "Embarked" are missing values. 



train_data.info()
#Here, I am just looking for correlations of class to sirvival rate using the training data. From these graphs, it can be 

#seen that passengers who died (survived = 0) were of the majority of the 3rd class and, we can see from the right graph that

#1st class has the highest survival rate (and lowest death rate). 



grid = sns.FacetGrid(train_data, col="Survived", margin_titles=True)

grid.map(plt.hist, "Pclass");
#Here, I am looking at gender in relation to survival. Here, it can be seen that females had the largest rate of survival. 

#It can be seen from this graph that ~75% of women survived and less that 20% of men survived meaning that gender had a large

#impact on survivability. 



ax = sns.barplot(x = "Sex", y = "Survived", data = train_data)
#Here, it can be seen that there is a corelation between class and survival as well. From this, we can see that more than

#60% of 1st class survived, more than 45% of 2nd class survived and less than 30% of 3rd class survived.



ax = sns.barplot(x = "Pclass", y = "Survived", data = train_data)
#From this graph, it can be seen that passengers who survived ( y = 1) tended to be younger than 75. For the rest of the data 

#however, there doesn't seem to be too much of a defining correlation that would benefit the classifier as opposed to gender or

#class in age. 



plt.scatter(train_data['Age'], train_data['Survived'])
#Here, it looks like more people who embarked at C had a larger survival rate however, again, gender and class were far more

#definite and would be better for the classifier. 



ax = sns.barplot(x = "Embarked", y = "Survived", data = train_data)
#Here, it can be seen that those who survived (graph on right) paid higher fare than those who died (graph on left) however,

#this goes into the area of class and could be better defined using the class column (fewer unique values unlike fare). 



pd.DataFrame.hist(train_data, column = "Fare", by="Survived")
#Here, siblings and survival rate doesn't have a huge correlation. It looks like less people with 4 & 3 siblings survived 

#but again, the margin between survival rates isn't as drastic as would be wanted for this type of project. 



ax = sns.barplot(x = "SibSp", y = "Survived", data = train_data)
#I have decided to use "Class" and "Gender" as my features in the end because they were the features I think were most drastic 

# and defining concerning survival rates. Below, I am removing every column that isn't class, gender or survival (which is the

#label so we need it.)



train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Fare', axis=1, inplace=True)

train_data.drop('Age', axis=1, inplace=True)

train_data.drop('Embarked', axis=1, inplace=True)

train_data.drop('PassengerId', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)

train_data.drop('Ticket', axis=1, inplace=True)

train_data.drop('Name', axis=1, inplace=True)



#Taking a look at data, realising that "Sex" is written as "male" or "female". This can be converted to numbers (as 0 and 1).

train_data.head()
#Here, I am replacing "female" with 0 and "male" with 1. 



train_data['Sex'] = train_data['Sex'].replace(['female','male'],[0,1])

#Checking types and seeing that sex is now an int rather than an object as it previously was. 



train_data.info()
#Making it so that x = features and y = labels and then, splitting the data into testing and training sections so that I can 

#train the classifier and then test the classifier. I am making it so that the training data is 85% and test data is 15%. 

#df_train = df_train.astype(int)



x = train_data.drop("Survived", axis = 1)

y = train_data["Survived"]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .15)
train_data.columns
#Just taking a look at x_test to make sure that it is just the features without the label. 



x_test.head()
#Just checking again to make sure that the ratio of entries in training vs. testing set looks right.

x_test.info()
x_train.info()
#Here, I am just testing the decision tree classifier. Accuracy is below. First am fitting the data after importing decision

#tree. This is what trains the classifier. Then, using tree_predictions, I am testing from using the features from x_test and 

#then comparing them with the answers in tree_accuracy. 



#decision tree classifier

from sklearn import tree



tree_classifier = tree.DecisionTreeClassifier()

tree_classifier.fit(x_train, y_train)

tree_predictions = tree_classifier.predict(x_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)

print(tree_accuracy)



#accuracy: 78.4%
#Basically doing the same thing as above except with KNearest Neighbors classifier to compare accuracy. Accuracy is below. 



#KNearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier()

knn_classifier.fit(x_train, y_train)

knn_predictions = knn_classifier.predict(x_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print(knn_accuracy)



#accuracy: 72.9973%
#Basically doing the same thing as above except with Support Vector classifier to compare accuracy. Accuracy is below. 



# Support Vector

from sklearn.svm import SVC

svc_classifier = SVC()

svc_classifier.fit(x_train, y_train)

svc_predictions = svc_classifier.predict(x_test)

svc_accuracy = accuracy_score(y_test, svc_predictions)

print(svc_accuracy)



#accuracy: 78.364
#Basically doing the same thing as above except with Random Forest classifier to compare accuracy. Accuracy is below. 



#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier()

forest_classifier.fit(x_train, y_train)

forest_predictions = forest_classifier.predict(x_test)

forest_accuracy = accuracy_score(y_test, forest_predictions)

print(forest_accuracy)



#accuracy: 78.496%
#Basically doing the same thing as above except with Logistic Regression classifier to compare accuracy. Accuracy is below. 

#This is not my own classifier, I saw this in a different notebook and learned how to use it and used it with my own data. 

#The notebook will be linked in acknowlegements. 



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

Y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train), 2)

acc_log



#accuracy: 80%
#Basically doing the same thing as above except with Gradient Boosting classifier to compare accuracy. Accuracy is below. 

#This is not my own classifier, I saw this in a different notebook and learned how to use it and used it with my own data. 

#The notebook will be linked in acknowlegements. 



# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test), 2)

print(acc_gbk)



#accuracy: 78.0%
#Here, dropping all features I don't want in the test dataset. 



test_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('Fare', axis=1, inplace=True)

test_data.drop('Age', axis=1, inplace=True)

test_data.drop('Embarked', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)

test_data.drop('Ticket', axis=1, inplace=True)

test_data.drop('Name', axis=1, inplace=True)





test_data['Sex'] = test_data['Sex'].replace(['female','male'],[0,1])



test_data.columns



test_data.head()
from sklearn.linear_model import LogisticRegression



y = train_data["Survived"]



features = ['Pclass', 'Sex']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



logreg = LogisticRegression()

logreg.fit(X, y)

Y_pred = logreg.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")


