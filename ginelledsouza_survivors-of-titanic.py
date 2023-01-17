# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ttrain = pd.read_csv("../input/titanic/train.csv")
ttrain.head()
ttest = pd.read_csv("../input/titanic/test.csv")
ttest.head()
print("Shape of Training set")
print(ttrain.shape)
print("\nShape of Testing set")
print(ttest.shape)
print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())
print("Missing value in training set")
train = round(((ttrain.isnull().sum()*100)/(ttrain.shape[0])),3).sort_values(ascending=False).head(5)
print(train)

print("\nMissing value in test set")
test = round(((ttest.isnull().sum()*100)/(ttrain.shape[0])),3).sort_values(ascending=False).head(5)
print(test)
#Age
train_mean = round(ttrain['Age'].mean(),0) 
ttrain['Age'] = ttrain['Age'].fillna(train_mean)

#Cabin
ttrain.drop(['Cabin'],inplace=True,axis=1)
ttrain.head()
#Embarked
print(ttrain['Embarked'].value_counts())

value = ttrain['Embarked'].value_counts().index[0]

#Since the frequency of 'S' is the highest we substitute the missing values with 'S'
ttrain['Embarked'] = ttrain['Embarked'].fillna(value) 
#Age
test_mean = round(ttest['Age'].mean(),0) 
ttest['Age'] = ttest['Age'].fillna(test_mean)

#Cabin
ttest.drop(['Cabin'],inplace=True,axis=1)
ttest.head()
#Fare
print(ttest['Fare'].mode())
test = ttest['Fare'].mode()[0]

#Since the frequency of '7.7500' is the highest we substitute the missing values with '7.7500'
ttest['Fare'] = ttest['Fare'].fillna(test) 
print("Missing value in training set")
print(ttrain.isnull().sum())
print("\nMissing value in test set")
print(ttest.isnull().sum())
ttrain.head()
print(ttrain.groupby(['Survived']).count()['PassengerId'])
sns.countplot(x='Survived',data=ttrain)
sns.countplot(x='Sex',data=ttrain)
print(ttrain.groupby(['Pclass']).mean()['Survived'],"\n")
sns.barplot(x='Pclass',y='Survived',data=ttrain)
print(ttrain.groupby(['Pclass','Survived']).count()['PassengerId'],"\n")
sns.countplot(x='Pclass',hue='Survived',data=ttrain)
print(ttrain.groupby(['Pclass','Sex']).count()['PassengerId'],"\n")
sns.countplot(x='Pclass',hue='Sex',data=ttrain)
print(ttrain.groupby(['Sex','Survived']).count()['PassengerId'],"\n")
sns.countplot(x='Sex',hue='Survived',data=ttrain)
print(ttrain.groupby(['Pclass','Sex']).mean()['Survived'],"\n")
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=ttrain)
print(ttrain.groupby(['SibSp','Survived']).count()['PassengerId'],"\n")
sns.countplot(x='SibSp',hue='Survived',data=ttrain)
sns.stripplot(x='Survived',y='Age',data=ttrain,jitter=True)
tfare = sns.FacetGrid(data=ttrain,hue='Survived')
tfare.map(sns.kdeplot,'Fare')
sns.jointplot(x='Age',y='Fare',data=ttrain,kind='reg')
sns.countplot(x='Embarked',hue='Survived',data=ttrain)
sns.countplot(x='Parch',hue='Survived',data=ttrain)
target = ttrain.groupby(['Embarked','Pclass','Survived'])

plt.figure(figsize=(8,8))
target.count()['PassengerId'].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Grouped Conditions")
plt.ylabel("Total Count")

plt.show()
correlate = ttrain.corr()
correlate
sns.heatmap(correlate)
print("Columns of training set")
print(ttrain.columns)
print("\nColumns of testing set")
print(ttest.columns)
data = [ttrain, ttest]
for dataset in data:
    dataset.drop(['PassengerId','Ticket'],axis=1,inplace=True)
for dataset in data:
    dataset['Relation'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relation'] > 0, 'Travelled_alone'] = 'No'
    dataset.loc[dataset['Relation'] == 0, 'Travelled_alone'] = 'Yes'
print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())
for dataset in data:
    dataset['Travelled_alone'] = dataset['Travelled_alone'].map({'No':0,'Yes':1})
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2})
def age(num): 
    
    if num <= 11: 
        return 0
  
    elif num > 11 and num <= 18:
        return 1
    
    elif num > 18 and num <= 22:
        return 2
    
    elif num > 22 and num <= 27:
        return 3
    
    elif num > 27 and num <= 33:
        return 4
    
    elif num > 33 and num <= 40:
        return 5
    
    elif num > 40 and num <= 66:
        return 6
    
    else: 
        return 7
    
    
def fare(num): 
    
    if num <= 7.91: 
        return 0
  
    elif num > 7.91 and num <= 33:
        return 1
    
    elif num > 33 and num <= 66:
        return 2
    
    elif num > 66 and num <= 99:
        return 3
    
    elif num > 99 and num <= 250:
        return 4
    
    elif num > 250 and num <= 360:
        return 5
   
    else: 
        return 6

for dataset in data:
    dataset['Age'] = dataset['Age'].apply(age)
    dataset['Fare'] = dataset['Fare'].apply(fare)
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Age'] = dataset['Age'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Age'] = dataset['Age'].astype(int)
print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)
for dataset in data:
    dataset.drop(['Name'],axis=1,inplace=True)
print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
X = ttrain.drop('Survived',axis=1)
y = ttrain['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Import Packages 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
#Object creation and fitting of training set
model1 = LogisticRegression()
model1.fit(X_train,y_train)
#Creation of a prediction variable
predict1 = model1.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predict1))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predict1))
#Accuracy Percentage
predict11 = round((model1.score(X_test, y_test)*100),0)
print("Precision of Logistic Regression is: ",predict11,"%") 
#Import Packages 
from sklearn.neighbors import KNeighborsClassifier
#Object creation and fitting of training set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
#Creation of a prediction variable
predictionsknn = knn.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionsknn))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionsknn))
#Accuracy Percentage
knnp = round((knn.score(X_test, y_test)*100),0)
print("Precision of K Nearest Neighbors is: ",knnp,"%") 
#Import Packages 
from sklearn.tree import DecisionTreeClassifier
#Object creation and fitting of training set
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#Creation of a prediction variable
predictiondt = dtree.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictiondt))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictiondt))
#Accuracy Percentage
dtt = round((dtree.score(X_test, y_test)*100),0)
print("Precision of Decision Tree is: ",dtt,"%") 
#Import Packages 
from sklearn.ensemble import RandomForestClassifier
#Object creation and fitting of training set
randfc = RandomForestClassifier(n_estimators=100)
randfc.fit(X_train,y_train)
#Creation of a prediction variable
predictionrf = randfc.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionrf))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionrf))
#Accuracy Percentage
random = round((randfc.score(X_test, y_test)*100),0)
print("Precision of Random Forest is: ",random,"%") 
#Import Packages 
from sklearn.svm import SVC
#Object creation and fitting of training set
svcm = SVC()
svcm.fit(X_train,y_train)
#Creation of a prediction variable
predictionsvc = svcm.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionsvc))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionsvc))
#Accuracy Percentage
svc = round((svcm.score(X_test, y_test)*100),0)
print("Precision of Support Vector Classifier: ",svc,"%")

#Import Packages 
from sklearn.naive_bayes import GaussianNB
#Object creation and fitting of training set
gaus = GaussianNB()
gaus.fit(X_train,y_train)
#Creation of a prediction variable
predictiongus = gaus.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictiongus))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictiongus))
#Accuracy Percentage
rig = round((gaus.score(X_test, y_test)*100),0)
print("Precision of Gaussian Naive Bayes is: ",rig,"%") 
#Object creation and fitting of training set
lgrcv = LogisticRegressionCV()
lgrcv.fit(X_train,y_train)
#Creation of a prediction variable
predictionlgcv = lgrcv.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionlgcv))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionlgcv))
#Accuracy Percentage
lgcv = round((lgrcv.score(X_test, y_test)*100),0)
print("Precision of  Logistic Regression CV is: ",lgcv,"%") 
#Import Packages 
from sklearn.ensemble import GradientBoostingClassifier
#Object creation and fitting of training set
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
#Creation of a prediction variable
predictiongbc = gbc.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictiongbc))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictiongbc))
#Accuracy Percentage
gbcp = round((gbc.score(X_test, y_test)*100),0)
print("Precision of Gradient Boosting Classifier is: ",gbcp,"%") 
#Import Packages 
from sklearn.linear_model import Perceptron
#Object creation and fitting of training set
per = Perceptron(max_iter=6)
per.fit(X_train,y_train)
#Creation of a prediction variable
predictionper = per.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionper))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionper))
#Accuracy Percentage
perr = round((per.score(X_test, y_test)*100),0)
print("Precision of Perceptron is: ",perr,"%") 
#Import Packages 
from sklearn.linear_model import SGDClassifier
#Object creation and fitting of training set
model2 = SGDClassifier()
model2.fit(X_train,y_train)
#Creation of a prediction variable
predict2 = model2.predict(X_test)
#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predict2))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predict2))
linr = round((model2.score(X_test, y_test)*100),0)
print("Precision of Stochastic Gradient Descent is: ",linr,"%") 
results = pd.DataFrame({'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'Support Vector Machines',  
                                  'Gausian Naive Baye', 'Logistic Regression CV', 'Stochastic Gradient Decent', 'Perceptron','Stochastic Gradient Descent'],
                        'Score': [predict11, knnp, dtt, random, svc, rig, lgcv, gbcp, perr,linr]
                      })

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(10)
