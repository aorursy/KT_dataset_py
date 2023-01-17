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

# Import Libraries

import pandas as pd

import numpy  as np

import matplotlib.pyplot as plt

%matplotlib inline

from statistics import mode

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# import the metrics class

from sklearn import metrics

import math

import matplotlib.pyplot as plt   

import seaborn as sns

import re
#read train data_set

train_data=pd.read_csv("../input/titanic/train.csv")



#read test data_set

test_data=pd.read_csv("../input/titanic/test.csv")
# show train table data_set

train_data
# show test table data_set

test_data
# as we see train dataset columns = 12 and test dataset columns = 11, Survived column in test data is missing...

#lets compare train data_set in columns and rows

train_data.info()
#lets compare test data_set in columns and rows

test_data.info()
# lets discover how many values is null in train dataset column

train_data.isnull().sum()
test_data.isnull().sum()
# as we see 86 null values in Ages column, 327 in cabin and 1 in Fare

print("the minimum age value in Age colum is:",train_data["Age"].min())
#so no zero value in Age column

print("the maximum age value in Age column is:",train_data["Age"].max())
#replacing NAN values in Age column by the average Age, (train data)

#creat a list called new_Age, include all Age column values

new_Age_train=pd.unique(train_data["Age"])



#get the mean value of Age column 

mean_Age_train= round(np.nanmean(new_Age_train),0)

print("Average of the Age column is =", mean_Age_train)
# let us replace the NAN values in Age column of train dataset by the mean value 

train_data["Age"] = train_data["Age"].fillna(34)

train_data["Age"] 

train_data.info()
#creat a list called new_Age, include all Age column values

new_Age_test=pd.unique(test_data["Age"])



#get the mean value of Age column 

mean_Age_test= round(np.nanmean(new_Age_test),0)

print("Average of the Age column is =", mean_Age_test)
# let us replace the NAN values in Age column of test dataset by the mean value 

test_data["Age"] = test_data["Age"].fillna(31)

test_data["Age"]

test_data.info()
plt.figure(figsize = (3, 4)) # setting the size of the figure

sns.set(style="darkgrid")    # setting the style of my vidualizations

#Creating a barplot with count of survived or not survived

sns.countplot(x = 'Survived', data = train_data, palette= ['Red', 'Blue'])    

plt.title("Survived (0 vs 1)",size=20)   # setting the title of my plot

plt.show()
Rate_survived=round(np.mean(train_data['Survived']),3)*100

print("the Survival Rate is:",Rate_survived,"%")
sns.set(style="darkgrid")

plt.figure(figsize = (3, 4))

sns.countplot(x = 'Sex', data = train_data, palette= ['skyblue','pink'])    

plt.title("Sex Gender (Male vs Female)",size=20)   # setting the title of my plot

plt.show()
total=train_data['Survived'].sum()

men=train_data[train_data['Sex']=='male']

women=train_data[train_data['Sex']=='female']

m=men['Sex'].count()

w=women['Sex'].count()

print("male:",m)

print("female:",w)

print("Rate of women:",round(w/(m+w)*100))

print("Rate of men:",round(m/(m+w)*100))
#Lets visualize the relation between survivors and Sex gender.

plt.figure(figsize=(5,5))

sns.countplot(x = 'Survived', hue = 'Sex', data = train_data)

plt.title("Relation : Sex and Survived",size=20)
# plot distributions of Fare passengers who survived or death

b = sns.FacetGrid( train_data, hue = 'Survived', aspect=4 )

b.map(sns.kdeplot, 'Fare', shade= True )

b.set(xlim=(0 , train_data['Fare'].max()))

b.add_legend()
#Lets visualize the relation Pclass  and number of survivors

plt.figure(figsize=(5,5))

sns.countplot(x = 'Survived', hue = 'Pclass', data = train_data)

plt.title("Relation : Pclass and Survived",size=20)
#Lets visualize the relation SibSp  and number of survivors

plt.figure(figsize=(5,5))

sns.countplot(x = 'Survived', hue = 'SibSp', data = train_data)

plt.title("Relation : SibSp and Survived",size=20)
#Lets visualize the relation Parch and number of survivors

plt.figure(figsize=(5,5))

sns.countplot(x = 'Survived', hue = 'Parch', data = train_data)

plt.title("Relation : Parch and Survived",size=20)
#Lets visualize the relation Embarked and number of survivors

plt.figure(figsize=(5,5))

sns.countplot(x = 'Survived', hue = 'Embarked', data = train_data)

plt.title("Relation : Embarked and Survived",size=20)
plt.figure(figsize=(30,15))

plt.subplot(235)

plt.hist(x = [train_data[train_data['Survived']==1]['Age'], train_data[train_data['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()
# plot distributions passengers age who survived or death

c = sns.FacetGrid( train_data, hue = 'Survived', aspect=4 )

c.map(sns.kdeplot, 'Age', shade= True )

c.set(xlim=(0 , train_data['Age'].max()))

c.add_legend()
sns.heatmap(train_data.corr(), annot = True)
#as we see there is 2 missing values in Embarked column in train data

print(train_data.isnull().sum())
#lets fill the NAN values of embarked column 

si = SimpleImputer(strategy="most_frequent")



train_data['Embarked'] = si.fit_transform(train_data['Embarked'].values.reshape(-1,1))

test_data['Embarked']  = si.fit_transform(test_data['Embarked'].values.reshape(-1,1))



#train_data['Embarked'].head()

print(train_data.isnull().sum())
train_data["Sex"][train_data["Sex"] == "male"] = 0

train_data["Sex"][train_data["Sex"] == "female"] = 1



test_data["Sex"][test_data["Sex"] == "male"] = 0

test_data["Sex"][test_data["Sex"] == "female"] = 1



train_data["Embarked"][train_data["Embarked"] == "S"] = 0

train_data["Embarked"][train_data["Embarked"] == "C"] = 1

train_data["Embarked"][train_data["Embarked"] == "Q"] = 2



test_data["Embarked"][test_data["Embarked"] == "S"] = 0

test_data["Embarked"][test_data["Embarked"] == "C"] = 1

test_data["Embarked"][test_data["Embarked"] == "Q"] = 2
# to remember... stille 1 missing float value in Fare colum (test_data)

#complete missing fare with median



si_Fare = SimpleImputer(missing_values=np.nan, strategy='mean')

test_data['Fare'] = si_Fare.fit_transform(test_data['Fare'].values.reshape(-1,1))

test_data.info()
#lets scaling data in Fare and Age columns



mmx = MinMaxScaler()



train_data['Fare'] = mmx.fit_transform(train_data['Fare'].values.reshape(-1,1))

test_data['Fare']  = mmx.fit_transform(test_data['Fare'].values.reshape(-1,1))



train_data['Age'] = mmx.fit_transform(train_data['Age'].values.reshape(-1,1))

test_data['Age'] = mmx.fit_transform(test_data['Age'].values.reshape(-1,1))



train_data
# Searching for the titles and extracting them from the names column (train and test)



train_data['Title'] = train_data['Name'].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())

test_data['Title'] = test_data['Name'].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())

print(train_data['Title'].unique())
print(test_data['Title'].unique())
# so we have titles for Nobels like Master, Capt...and others for regular people..

# so lets replace Nobels people by Dummy value 1 and regular people by Dummy value 0

title_mapping = {'Mr.': 0, 'Mrs.': 0, 'Miss.': 0, 'Master.' : 1,'Don.': 1, 'Rev.' : 1,'Dr.' : 1,'Mme.': 0, 'Ms.': 0, 'Major.': 1,

 'Lady.': 1, 'Sir.': 1, 'Mlle.': 0, 'Col.': 1, 'Capt.': 1, 'Countess.': 1, 'Jonkheer.': 1,'Dona.': 1,}



train_data['Title'] = train_data['Title'].map(title_mapping)

train_data['Title'] = train_data['Title'].fillna(0)



    

print(train_data['Title'].unique())
title_mapping = {'Mr.': 0, 'Mrs.': 0, 'Miss.': 0, 'Master.' : 1,'Don.': 1, 'Rev.' : 1,'Dr.' : 1,'Mme.': 0, 'Ms.': 0, 'Major.': 1,

 'Lady.': 1, 'Sir.': 1, 'Mlle.': 0, 'Col.': 1, 'Capt.': 1, 'Countess.': 1, 'Jonkheer.': 1,'Dona.': 1,}



test_data['Title'] = test_data['Title'].map(title_mapping)

test_data['Title'] = test_data['Title'].fillna(0)



print(test_data['Title'].unique())
for n, i in enumerate(train_data["SibSp"]):

    if i != 0:

     train_data["SibSp"][n] = 1





for m, k in enumerate(train_data["Parch"]):

    if k != 0:

     train_data["Parch"][m] = 1
#lets drop Name, Ticket and Cabin columns from train data set 

DS_train=train_data.drop(columns=["Cabin","Name","Ticket"])

DS_train
#lets drop Name, Ticket, Cabin, columns from test data set

DS_test=test_data.drop(columns=["Cabin","Name","Ticket"])

DS_test
#lets re-ordered columns in train dataset

DS_train = DS_train[['PassengerId','Sex','Pclass','Age','Fare','SibSp','Parch','Embarked','Title','Survived']]

DS_train
#lets re-ordered columns in test dataset

DS_test = DS_test[['PassengerId','Sex','Pclass','Age','Fare','SibSp','Parch','Embarked','Title']]

DS_test
#creat dataset in features and target variable

feature_columns = ['Sex','Pclass','Age','Fare','SibSp','Parch','Embarked','Title']

X = DS_train[feature_columns] # training Features (Predectors)

y = DS_train['Survived']     # Target variable

X
y
# split X and y into training and testing sets

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.30,random_state=0)



#Here, the Dataset is broken into two parts in a ratio of 70:30. It means 70% data will 

#be used for model training and 30% for model testing.

#First, import the Logistic Regression module and create a Logistic Regression classifier object using LogisticRegression() 

#function.Then, fit your model on the train set using fit() and perform prediction on the test set using predict().

#instantiate the model (using the default parameters)



#test the accuracy of Logistic Regression.

logreg = LogisticRegression(max_iter = 30000)



# fit the model with data

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_val)





acc_LOG = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_LOG)
#Model Evaluation using Confusion Matrix

#A confusion matrix is a table that is used to evaluate the performance of a classification model. 

#You can also visualize the performance of an algorithm. 

#The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise.





cnf_matrix = metrics.confusion_matrix(y_val, y_pred)

cnf_matrix
# Here, you can see the confusion matrix in the form of the array object. 

# The dimension of this matrix is 2*2 because this model is binary classification. 

# You have two classes 0 and 1. Diagonal values represent accurate predictions, 

# while non-diagonal elements are inaccurate predictions. In the output, 139 and 73 are actual predictions, 

# and 29 and 27 are incorrect predictions.



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)



# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
accuracy=((139+73)/(139+29+27+73))

print('accuracy is: ', (round(accuracy, 2)*100))
#test the accuracy of Support Vector Machines.



svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# test the accuracy of Linear SVC





linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

y_pred = linear_svc.predict(X_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
#test the accuracy of Decision Tree





decisiontree = DecisionTreeClassifier()

decisiontree.fit(X_train, y_train)

y_pred = decisiontree.predict(X_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# test the accuracy of Random Forest

randomforest = RandomForestClassifier()

randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# test the accuracy of Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
# test the accuracy of KNN or k-Nearest Neighbors

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
# test the accuracy of Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
# test the accuracy of Gradient Boosting Classifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_val)

acc_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbc)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
from catboost import CatBoostClassifier, cv, Pool



clf =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=2)

clf.fit(X_train,y_train, eval_set=(X_val,y_val), early_stopping_rounds=100,verbose=False)

#,cat_features=cate_features_index,

y_pred = clf.predict(X_val)

acc_clf = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_clf)
# Let's compare the accuracies of each model



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier','Cat Boost Classifier'],

    'Accuracy': [acc_svc, acc_knn, acc_LOG, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbc, acc_clf]})

models.sort_values(by='Accuracy', ascending=False)
#set the output as a dataframe and convert to csv file named submission.csv



ids = DS_test['PassengerId']

predictions = gbc.predict(DS_test.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)