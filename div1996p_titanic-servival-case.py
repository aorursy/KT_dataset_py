# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_df.groupby(['Sex','Survived'])['Survived'].count()

titanic_df.isnull().sum()

titanic_test.isnull().sum()

print("Oldest pessanger was :",titanic_df['Age'].max(),'year')
print("Youngest pessanger was :",titanic_df['Age'].min(),'year')
print("average age on the ship was: ",titanic_df['Age'].mean(),'Year')


titanic_df['Initial']=0

for i in titanic_df:
    titanic_df['Initial']=titanic_df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
titanic_test['Initial']=0

for i in titanic_test:
    titanic_test['Initial']=titanic_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
titanic_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
titanic_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
titanic_df.groupby('Initial')['Age'].mean()
titanic_test.groupby('Initial')['Age'].mean()
titanic_df.loc[(titanic_df.Age.isnull())&(titanic_df.Initial == 'Mr'),'Age']=33
titanic_df.loc[(titanic_df.Age.isnull())&(titanic_df.Initial == 'Mrs'),'Age'] = 36
titanic_df.loc[(titanic_df.Age.isnull())&(titanic_df.Initial == 'Master'),'Age'] = 5
titanic_df.loc[(titanic_df.Age.isnull())&(titanic_df.Initial == 'Miss'), 'Age'] = 22
titanic_df.loc[(titanic_df.Age.isnull())&(titanic_df.Initial == 'Other'),'Age'] = 46
titanic_test.loc[(titanic_test.Age.isnull())&(titanic_test.Initial == 'Mr'),'Age']=33
titanic_test.loc[(titanic_test.Age.isnull())&(titanic_test.Initial == 'Mrs'),'Age'] = 36
titanic_test.loc[(titanic_test.Age.isnull())&(titanic_test.Initial == 'Master'),'Age'] = 5
titanic_test.loc[(titanic_test.Age.isnull())&(titanic_test.Initial == 'Miss'), 'Age'] = 22
titanic_test.loc[(titanic_test.Age.isnull())&(titanic_test.Initial == 'Other'),'Age'] = 46
titanic_df.Age.isnull().any()
titanic_test.Age.isnull().any()
titanic_df['Embarked'].fillna('S',inplace=True)
titanic_df.Embarked.isnull().any()
titanic_test['Embarked'].fillna('S',inplace=True)
titanic_test.Embarked.isnull().any()
print('Highest Fare was:',titanic_df['Fare'].max())
print('Lowest Fare was :',titanic_df['Fare'].min())
print('Average Fare Was',titanic_df['Fare'].mean())
titanic_df['Age_band']=0
titanic_df.loc[titanic_df['Age']<=16,'Age_band']=0
titanic_df.loc[(titanic_df['Age']>16)&(titanic_df['Age']<=32),'Age_band']=1
titanic_df.loc[(titanic_df['Age']>32)&(titanic_df['Age']<=48),'Age_band']=2
titanic_df.loc[(titanic_df['Age']>48)&(titanic_df['Age']<=64),'Age_band']=3
titanic_df.loc[titanic_df['Age']>64,'Age_band']=4
titanic_df.head(2)
titanic_test['Age_band']=0
titanic_test.loc[titanic_test['Age']<=16,'Age_band']=0
titanic_test.loc[(titanic_test['Age']>16)&(titanic_df['Age']<=32),'Age_band']=1
titanic_test.loc[(titanic_test['Age']>32)&(titanic_df['Age']<=48),'Age_band']=2
titanic_test.loc[(titanic_test['Age']>48)&(titanic_df['Age']<=64),'Age_band']=3
titanic_test.loc[titanic_test['Age']>64,'Age_band']=4
titanic_test.head(2)
titanic_test['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
titanic_test['Family_Size']=0
titanic_test['Family_Size']=titanic_test['Parch']+titanic_test['SibSp']#family size['Alone']=0
titanic_test['Alone']=0
titanic_test.loc[titanic_test.Family_Size==0,'Alone']=1#Alone
#ound_gradient(cmap='summer_r')
titanic_test['Fare_Range'] = pd.qcut(titanic_test['Fare'],4)


titanic_df['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
titanic_df['Family_Size']=0
titanic_df['Family_Size']=titanic_df['Parch']+titanic_df['SibSp']#family size['Alone']=0
titanic_df['Alone']=0
titanic_df.loc[titanic_df.Family_Size==0,'Alone']=1#Alone
#ound_gradient(cmap='summer_r')
titanic_df['Fare_Range'] = pd.qcut(titanic_df['Fare'],4)
titanic_df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')

titanic_df['Fare_cat']=0
titanic_df.loc[titanic_df['Fare']<=7.91,'Fare_cat']=0
titanic_df.loc[(titanic_df['Fare']>7.91)&(titanic_df['Fare']<=14.454),'Fare_cat']=1
titanic_df.loc[(titanic_df['Fare']>14.454)&(titanic_df['Fare']<=31),'Fare_cat']=2
titanic_df.loc[(titanic_df['Fare']>31)&(titanic_df['Fare']<=513),'Fare_cat']=3
titanic_test['Fare_cat']=0
titanic_test.loc[titanic_test['Fare']<=7.91,'Fare_cat']=0
titanic_test.loc[(titanic_test['Fare']>7.91)&(titanic_test['Fare']<=14.454),'Fare_cat']=1
titanic_test.loc[(titanic_test['Fare']>14.454)&(titanic_test['Fare']<=31),'Fare_cat']=2
titanic_test.loc[(titanic_test['Fare']>31)&(titanic_test['Fare']<=513),'Fare_cat']=3
from sklearn import preprocessing
titanic_test['Embarked'] = pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(titanic_test.Embarked))
titanic_test['Sex'] = pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(titanic_test.Sex))
titanic_test1 = titanic_test[['Pclass','Sex','SibSp','Parch','Embarked','Age_band','Fare_cat']]

titanic_test1.head()

from sklearn import preprocessing
titanic_df.head()
titanic_df['Embarked'] = pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(titanic_df.Embarked))
titanic_df['Sex'] = pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(titanic_df.Sex))
#titanic_df.drop(['Initial','Name','PassengerId','Ticket','Cabin','Age','Fare'],axis = 1,inplace=True)

#titanic_test1['Embarked'] = pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(titanic_test1.Embarked))
#titanic_test1['Sex'] = pd.DataFrame(preprocessing.LabelBinarizer().fit_transform(titanic_test1.Sex))
titanic_test1.head()
titanic_df.head()
titanic_X = titanic_df[['Pclass','Sex','SibSp','Parch','Fare_cat','Embarked','Age_band']]
y = titanic_df['Survived']
titanic_X.head()
titanic_test1.head()
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
train_X,test_X,train_y,test_y = train_test_split(titanic_X,y, test_size=0.3, random_state=0,stratify=titanic_df['Survived'])


model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_y)
prediction1=model.predict(titanic_test1)
#print('Accuracy for rbf SVM is',metrics.accuracy_score(prediction1,test_y))
#print(prediction1)

#print(submission['Survived'])
titanic_test.head()
model = svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_y)
prediction2 = model.predict(titanic_test1)
#print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_y))

model = LogisticRegression()
model.fit(train_X,train_y)
prediction3 = model.predict(titanic_test1)
#print('The Accuracy of Logistic Regression is',metrics.accuracy_score(prediction3,test_y))
model = DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction4=model.predict(titanic_test1)
#print('The accuracy of the decision Tree is',metrics.accuracy_score(prediction4,test_y))
model=KNeighborsClassifier(n_neighbors=1) 
model.fit(train_X,train_y)
prediction5=model.predict(titanic_test1)
#print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_y))
a_index=list(range(1,11))
a=pd.Series()
x = [0,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X,train_y)
    predicition = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(predicition,test_y)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
    
    
model=GaussianNB()
model.fit(train_X,train_y)
prediction6=model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,test_y))
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction7=model.predict(titanic_test1)
#print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_y))
#submission = pd.DataFrame()
#submission['PassengerId'] = titanic_test.PassengerId
#submission['Survived'] = prediction5
#submission.to_csv('submission.csv',index=False)

#print('Submission file,created' )
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
kfold = KFold(n_splits=10, random_state=22)
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear svm','Radial svm','Logistics Regression','Knn','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,titanic_X,y,cv = kfold,scoring="accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz,'Std':std},index =classifiers )
new_models_dataframe2
    
    
    
plt.subplots(figsize=(12, 6))
box = pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()

#Confusion matrix : it gives number of correct and incorrect classification made by classifier
import seaborn as sns
f,ax = plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred), ax = ax[0,0],annot=True,fmt = '2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),ax = ax[0,1],annot=True,fmt = '2.0f')
ax[0,1].set_title('Matrix for Linear SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix For KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,0],annot = True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random Forest')
y_pred = cross_val_predict(LogisticRegression(),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),titanic_X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),ax = ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()



from sklearn.model_selection import GridSearchCV
C = [0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(titanic_X,y)
print(gd.best_score_)
print(gd.best_estimator_)
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(titanic_X,y)
print(gd.best_score_)
print(gd.best_estimator_)

# Apply ensemble learning
#Voting Classifier
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),('LR',LogisticRegression(C = 0.05)),('DT',DecisionTreeClassifier(random_state=0)),('NB',GaussianNB()),('svm',svm.SVC(kernel='linear',probability=True))],voting='soft').fit(train_X,train_y)
#print("the Accuracy of ensemble model is:",ensemble_lin_rbf.score(test_X,test_y))
#cross=cross_val_score(ensemble_lin_rbf,titanic_X,y,cv=10,scoring = 'accuracy')
#print("the cross validate score is",cross.mean())
predict8 = ensemble_lin_rbf.predict(titanic_test1)
#submission = pd.DataFrame()
#submission['PassengerId'] = titanic_test.PassengerId
#submission['Survived'] = predict8
#submission.to_csv('submission.csv',index=False)

#print('Submission file,created' )
#bagging KNN
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_y)
predict9=model.predict(titanic_test1)
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_y)
predict10 = model.predict(titanic_test1)
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result = cross_val_score(ada,titanic_X,y,cv=10,scoring='accuracy')
print('the cross validate score for AdaBoost is:',result.mean())
#stochastic Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,titanic_X,y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())
import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,titanic_X,y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())
#hyperparameter Tunning for AdaBoost
n_estimators = list(range(100,1100,100))
learning_rate = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learning_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(titanic_X,y)

print(gd.best_score_)
print(gd.best_estimator_)
predict11 = gd.predict(titanic_test1)
submission = pd.DataFrame()
submission['PassengerId'] = titanic_test.PassengerId
submission['Survived'] = predict11
submission.to_csv('submission.csv',index=False)

print('Submission file,created' )









