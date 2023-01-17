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
#Importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



%matplotlib inline
#Importing Datasets

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

#Check both the datasets

train_data.head()

#train_data.shape

#train_data.describe()

#train_data.isnull().sum()



test_data.head()

#test_data.shape

#test_data.describe()

#test_data.isnull().sum()
#Feature : Age

train_data['Age'].mean()

test_data['Age'].mean()
train_data['Age'].fillna(train_data['Age'].median(), inplace = True)
test_data['Age'].fillna(test_data['Age'].median(), inplace = True)
#Feature : Embarked

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)

test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace = True)
#Feature : Fare

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
#Feature : 'Sex','Embarked'

test_data['Sex_Male'] = (test_data['Sex'] == "male").astype(int)

test_data['Embarked_S'] = (test_data['Embarked'] == "S").astype(int)
#SibSp(Sibling-Spouse) and Parch (Parent-children) will help us to determine the size of the family

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

train_data.head()
#Creating new column Agegroup from Age 

bins= [0,12,20,30,60,100]

labels = ['Children','Teen','Young Adult','Adult','Elder']

train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=bins, labels=labels, right=False)

print (train_data)
#Drop the useless columns

useless_cols = ['PassengerId','Name','Ticket','Cabin','SibSp','Parch','Age']

train_data.drop(useless_cols,axis=1,inplace=True)
train_data.head()
#Drop the useless columns

useless_cols = ['Cabin','Fare']

test_data.drop(useless_cols,axis=1,inplace=True)
#No null values in the train_data

train_data.isnull().sum()
#No null values in the test_data

test_data.isnull().sum()
#Correlating each column in the data

train_data.corr()
#Survival by Gender

#Female Survival is more than Male

sns.set_style('white')

sns.barplot(train_data['Sex'],train_data['Survived'])

#Survival by Pclass

#Class 1 survival is better than Class 2 and Class 3

sns.barplot(train_data['Pclass'],train_data['Survived'])
#Survival by FamilySize

#The familysize of 3 has better survival than others

sns.barplot(train_data['FamilySize'],train_data['Survived'])
#Survival by Embarked 

#Port of Embarkation: C has better survival

sns.barplot(train_data['Embarked'],train_data['Survived'])
#Survival by Agegroup

#Children Agegroup has better survival rate

sns.barplot(train_data['AgeGroup'],train_data['Survived'])
#Visualize relation between each column using heatmap 

sns.heatmap(train_data.corr())
#Convert Categorical data to Numeric

train_data['Sex'].replace(['male','female'],[0,1],inplace=True)

train_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train_data['AgeGroup'].replace(['Children','Teen','Young Adult','Adult','Elder'],[0,1,2,3,4],inplace=True)
#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
#Split dataset into train and test data

from sklearn.model_selection import train_test_split



X = train_data[train_data.loc[:, train_data.columns != 'Survived'].columns]

y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
#Using Logistic Regression

logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

acc_LR = logmodel.score(X_test, y_test)

print('Accuracy:',acc_LR)
#Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, predictions)

print(cnf_matrix)



#Visualizing Confusion Matrix using HeatMap

confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
#Using Gaussian Naive Bayes

gnb = GaussianNB()

gnb.fit(X_train, y_train)

 

# making predictions on the testing set

y_pred = gnb.predict(X_test)

 

# comparing actual response values (y_test) with predicted response values (y_pred)

from sklearn import metrics

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred))



cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(cnf_matrix)



#Visualizing Confusion Matrix using HeatMap

confusion_matrix = pd.crosstab(y_test,y_pred, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)

 
#Using SVM

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



#Metrics

acc_svm = print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Using Decision Tree

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(X_test)

acc_DT = print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Using KNN

knn = KNeighborsClassifier(n_neighbors=3)



#Train the model using the training sets

knn.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = knn.predict(X_test)

acc_knn = print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Using Random Forest



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

acc_RF = print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Random Forest provides better accuracy than other models.

methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]

accuracy = [0.787, 0.776,0.787,0.81,0.832, 0.849]

sns.set_style("whitegrid")

plt.figure(figsize=(16,10))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %",fontsize=14)

plt.xlabel("ML Models",fontsize=14)

sns.barplot(x=methods, y=accuracy)

plt.show()
final_knn = KNeighborsClassifier(n_neighbors=3)

final_knn.fit(X, y)
X_test = test_data[['Pclass', 'Sex_Male', 'Age', 'SibSp', 'Parch', 'Embarked_S']].values

y_pred = final_knn.predict(X_test)

score = final_knn.score(X_test,y_pred)

score
result_submit = pd.DataFrame({'PassengerId': test_data.index, 'Survived': y_pred})

display(result_submit.head())

result_submit.to_csv("/kaggle/working/submission.csv", header=True, index=False)