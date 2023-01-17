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
# we will use pandas library to load the data from csv files into matrices 

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

m_train = train.shape[0];  #number of training examples

m_test = test.shape[0];    #number of test examples

print("No. of train examples " + str(m_train) + " & No. of test examples " + str(m_test));
# we need to first clean the given data set

# get the total number of values in each column

print("Total Values count in each column\n")

print(train.count());

print("\nNull Values count in each column\n");

print(train.isnull().sum());
#from above we can conclude that there are three columns we need to focus on missing data

# since "Embarked" Column has very less percentage we can fill it with most frequent value in its column

import seaborn as sb;  #or data visualization

print(train['Embarked'].value_counts());

sb.countplot(x = 'Embarked', data = train);

# so the port 'S' is the most frequent value in that column. Replace those two null values with this value

train['Embarked'].fillna(train['Embarked'].value_counts().idxmax(), inplace = True);

#Since cabin column has most null values we don't use it in prediction

train.drop('Cabin', axis = 1, inplace = True);

#we will fill the ages with median value.

train['Age'].fillna(train['Age'].median(skipna = True), inplace = True);

#Now check again if there are any null values in any column

print(train.isnull().sum());

#print(train.head());
#Analysis of male and female w.r.to survival

sb.countplot(x = 'Survived', hue = 'Sex', data = train);

#here we can see percentage survival of females are more than percentage survivals of men
sb.countplot(x = 'Survived', hue = 'Pclass', data = train);

# Here we can see that percentage of people survived in class1 is more and

# percentage of people deceased are more in class 3 than any other classes
sb.countplot(x = 'Fare', hue = 'Survived', data = train);
## Create categorical variable for traveling alone

#train['TravelAlone']=np.where((train["SibSp"]+train["Parch"])>0, 0, 1)

#train.drop('SibSp', axis=1, inplace=True)

#train.drop('Parch', axis=1, inplace=True)

#Now let's get dummies inplace of male and female & embarked and Pclass columns



training=pd.get_dummies(train, columns=["Pclass","Embarked","Sex"])

# Remove the colums from the data set that we don't use for prediction

#train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True);

#concat the new ones in our data set

training.drop('Sex_female', axis=1, inplace=True)

training.drop('PassengerId', axis=1, inplace=True)

training.drop('Name', axis=1, inplace=True)

training.drop('Ticket', axis=1, inplace=True)

train = training;

print(train.head());
#Survival vs Travelling Alone

#sb.countplot(x = 'Survived', hue = 'TravelAlone',  data = train);

#Here we can see that most of the deceased are travelling alone
#Now our train dataset is ready. Apply the same changes to test dataset

print(test.count());

print("\nCount of null values \n");

print(test.isnull().sum());
#here also drop the cabin

test.drop("Cabin", axis = 1, inplace = True);

#Assign median value to misssing ages and missing fares

test['Age'].fillna(train['Age'].median(skipna = True), inplace = True);

test['Fare'].fillna(train['Fare'].median(skipna = True), inplace = True);

print(test.isnull().sum());
#Now make same changes as you made to train dataset

#test['TravelAlone'] = np.where((test['SibSp'] + test['Parch'])>0, 0, 1);  #1 will be assigned if they are travelling alone

#Remove those two columns

#test.drop('SibSp', axis = 1, inplace = True);

#test.drop('Parch', axis = 1, inplace = True);



testing = pd.get_dummies(test, columns = ['Pclass', 'Embarked', 'Sex']);

testing.drop('Sex_female', axis=1, inplace=True)

#testing.drop('PassengerId', axis=1, inplace=True)

testing.drop('Name', axis=1, inplace=True)

testing.drop('Ticket', axis=1, inplace=True)



test = testing

print(test.head());
#Now our datasets are ready

X = train.drop("Survived", axis = 1);

Y = train['Survived'];

#Now implement Logistic regression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)  #70/30 split of train data to evaluate ourself
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression();

logr.fit(X_train, Y_train);

print(logr);
#Now see how good our model predicts

predictions = logr.predict(X_test);

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

print(classification_report(Y_test, predictions));

print("Train Accuracy of logistic regression is : "+ str(accuracy_score(Y_train, logr.predict(X_train))*100));

print("Dev Accuracy of logistic regression is : "+ str(accuracy_score(Y_test, predictions)*100));
#We will use decision tree to do the same classification

from sklearn.tree import DecisionTreeClassifier

#using gini index

clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 3,  splitter = 'random');

# we can use either random or best value for splitter. Here random gives more accuracy

clf_gini.fit(X_train, Y_train);

print('Train Accuracy using DT using gini index is : ' + str(accuracy_score(Y_train,clf_gini.predict(X_train))*100));

print('Dev Accuracy using DT using gini index is : ' + str(accuracy_score(Y_test,clf_gini.predict(X_test))*100));
# using information gain and entropy

clf_gain = DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth = 3, splitter = 'best');

clf_gain.fit(X_train, Y_train);

print(clf_gain);

print('____________________________________________________________________________________');

print('Train Accuracy using DT using Entropy is : ' + str(accuracy_score(Y_train,clf_gain.predict(X_train))*100));

print('Dev Accuracy using DT using Entropy is : ' + str(accuracy_score(Y_test,clf_gain.predict(X_test))*100));
#Now we are going to use Random Forest Classifier which is nothing but a group of decision trees

from sklearn.ensemble import RandomForestClassifier

clf_rnd = RandomForestClassifier(max_depth = 5, n_estimators= 100);

print(clf_rnd);

print('_____________________________________________________________________________________');

clf_rnd.fit(X_train, Y_train);

print('Train Accuracy using RFC is : ' + str(accuracy_score(Y_train,clf_rnd.predict(X_train))*100));

print('Dev Accuracy using RFC is : ' + str(accuracy_score(Y_test,clf_rnd.predict(X_test))*100));
#Submission step

#logr.fit(X, Y);



#predict = logr.predict(test);

#output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict});

#output.to_csv('my_submission.csv', index = False);

#test['Survived'] = logr.predict(test.drop('PassengerId', axis = 1));

#Y_test['PassengerId'] = test['PassengerId'];

print(test.count());

xt = test.drop('PassengerId', axis = 1);

test['Survived'] = clf_rnd.predict(xt);

submission = test[['PassengerId','Survived']];



submission.to_csv("submission.csv", index=False);



submission.tail();

print("submission was successful!");