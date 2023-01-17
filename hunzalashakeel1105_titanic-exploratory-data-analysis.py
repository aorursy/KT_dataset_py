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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()

data.isnull().sum()
f, ax = plt.subplots(1,2, figsize=(10,5))
data['Survived'].value_counts().plot.pie( autopct='%1.1f%%')
ax[1].set_title('Survived')
ax[1].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax[0])
data.groupby(['Sex', 'Survived'])['Survived'].count()
f, ax = plt.subplots(1,2, figsize=(10,5))
data.groupby(['Sex'])['Survived'].mean().plot.bar(ax=ax[0])
ax[0].set_title = 'Survived vs Sex'
sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')
sns.countplot('Pclass', hue='Survived', data=data)
pd.crosstab(data.Pclass, [data.Sex, data.Survived], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)
print('Oldest Passenger was', data.Age.max())
print('Youngest Passenger was', data.Age.min())
print('Average Age of passengers was', data.Age.mean())
f, ax = plt.subplots(1, 2, figsize=(10,5))
sns.violinplot('Pclass', 'Age', hue='Survived', data=data, split=True, ax=ax[0])
sns.violinplot('Sex', 'Age', hue='Survived', data=data, split=True, ax=ax[1])
data['Initials'] = 0
for i in data:
    data['Initials'] = data.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(data.Sex, data.Initials).style.background_gradient('summer_r')
data['Initials'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                         ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],
                         inplace=True)
data.groupby('Initials')['Age'].mean()
# assign age values to null accrding to initials
data.loc[(data.Age.isnull())&(data.Initials=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initials=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initials=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initials=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initials=='Other'),'Age']=46
data.Age.isnull().any() #So no null values left finally 
f, ax = plt.subplots(1,2, figsize=(12,7))
ax[0].set_title('Survived = 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
ax[1].set_xticks(x1)
data[data.Survived==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black')
ax[1].set_title('Survived = 1')
data[data.Survived==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black')
pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Embarked','Survived',data=data)
# due to maximum passenger board in embarked S we fill Nan values with S
data['Embarked'].fillna('S',inplace=True)
data.Embarked.isnull().any()# Finally No NaN values
pd.crosstab(data.SibSp, data.Survived).style.background_gradient(cmap='summer_r')
sns.barplot(x='SibSp', y='Survived', data=data)
print('Highest Fare is ', max(data.Fare))
print('Lowest Fare is ', min(data.Fare))
print('Average Fare is ', data.Fare.mean())
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()
sns.heatmap(data=data.corr(), cmap='RdYlGn', annot=True, linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
data['Age_band']= 0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head()
data['Age_band'].value_counts().to_frame()
sns.barplot('Age_band', 'Survived', data=data)
data['Family_size'] = 0
data['Family_size']=data['Parch']+data['SibSp']#family size
data['Alone']=0
data.loc[data.Family_size==0,'Alone']=1#Alone
f, ax = plt.subplots(1, 2, figsize=(12,5))
sns.barplot('Family_size', 'Survived', data=data, ax=ax[0])
ax[0].set_title('Family_size vs Survived')
sns.barplot('Alone', 'Survived', data=data, ax=ax[1])
ax[1].set_title('Alone vs Survived')
data['Fare_range'] = pd.qcut(data['Fare'], 4)
data.head()
# we convert fare range into bins as we did with Age_band
data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3
sns.barplot('Fare_cat', 'Survived', data=data)
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initials'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.show()
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data.Survived)
train_X = train[train.columns[1:]]
train_Y = train[train.columns[0]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[0]]
X=data[data.columns[1:]]
Y=data['Survived']
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(train_X, train_Y)
SVMPrediction = model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(SVMPrediction,test_Y))
model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_X, train_Y)
SVMLinearPrediction = model.predict(test_X)
print('Accuracy for linear SVM', metrics.accuracy_score(SVMLinearPrediction, test_Y))
model = LogisticRegression()
model.fit(train_X, train_Y)
LogisticPrediction = model.predict(test_X)
print('Accuracy for Logistic Regression', metrics.accuracy_score(LogisticPrediction, test_Y))
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
DecisionTreePrediction = model.predict(test_X)
print('Accuracy for Decsion Tree', metrics.accuracy_score(DecisionTreePrediction, test_Y))
model = KNeighborsClassifier()
model.fit(train_X, train_Y)
KNNPrediction = model.predict(test_X)
print('Accuracy for KNN', metrics.accuracy_score(KNNPrediction, test_Y))
a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
model=GaussianNB()
model.fit(train_X,train_Y)
NBPrediction=model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(NBPrediction,test_Y))
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
RandomForestPrediction=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(RandomForestPrediction,test_Y))
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22)
modelMean = []
modelAccuracy = []
ModelStd = []
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for model in models:
    cv_result = cross_val_score(model, X, Y, cv = kfold, scoring='accuracy')
    modelMean.append(cv_result.mean())
    ModelStd.append(cv_result.std())
    modelAccuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':modelMean,'Std':ModelStd},index=classifiers)       
new_models_dataframe2
f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()

from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}
SVMGridSearchModel = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
SVMGridSearchModel.fit(X, Y)
print(SVMGridSearchModel.best_score_)
print(SVMGridSearchModel.best_estimator_)
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
RFGridSearchModel=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
RFGridSearchModel.fit(X,Y)
print(RFGridSearchModel.best_score_)
print(RFGridSearchModel.best_estimator_)
