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
train=pd.read_csv('/kaggle/input/titanic/train.csv')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train.head()
train.describe()
train.isnull().sum()
f,ax=plt.subplots(1,2,figsize=(12,5))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title("Survived")

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title("Count")
train.groupby(['Sex','Survived'])['Sex'].count()
f,ax=plt.subplots(1,2,figsize=(12,5))

train[['Sex','Survived']].groupby('Sex').mean().plot.bar(ax=ax[0])

ax[0].set_title("female vs male survival rate")

sns.countplot("Sex",data=train,ax=ax[1],hue='Survived')
train['Sex']=train['Sex'].map({'male':1,'female':0}).astype(int)
train.head()
pd.crosstab(train.Pclass,train.Survived,margins=True).style.background_gradient(cmap="summer_r")
f,ax=plt.subplots(1,2,figsize=(12,5))

train['Pclass'].value_counts().plot.bar(ax=ax[0])

ax[0].set_title("Count of members of each class")

sns.countplot(x='Pclass',data=train,ax=ax[1],hue='Survived')
pd.crosstab([train.Sex,train.Survived],train.Pclass).style.background_gradient(cmap="summer_r")
sns.factorplot(x='Pclass',y='Survived',data=train,hue='Sex')

plt.show()
train['Age'].isnull().sum()
age_avg=train['Age'].mean()

age_std=train['Age'].std()

null_counts=train['Age'].isnull().sum()



age_null_random_list=np.random.randint(low=age_avg-age_std,high=age_avg+age_std,size=null_counts)
print("The youngest person on board is",train['Age'].min())

print("The oldest person on board is",train['Age'].max())

train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age'].isnull().sum()
pd.qcut(train['Age'],5)
train['Age_band']=0

train.loc[train['Age']<=19,'Age_band']=0

train.loc[(train['Age']>19)&(train['Age']<=25),'Age_band']=1

train.loc[(train['Age']>25)&(train['Age']<=31),'Age_band']=2

train.loc[(train['Age']>31)&(train['Age']<=40),'Age_band']=3

train.loc[train['Age']>40,'Age_band']=4

train.head(2)
train.drop(['Age'],axis=1,inplace=True)
train.head()
sns.factorplot('Age_band','Survived',data=train,col='Pclass')

plt.show()
train['Total_Members']=train['Parch']+train['SibSp']+1
train.head()
train[['Total_Members','Survived']].groupby('Total_Members').mean().plot.bar()
train['IsAlone']=0

train.loc[train['Total_Members']==1,'IsAlone']=1
train.head()
print(train[['IsAlone','Survived']].groupby('IsAlone').mean())

train[['IsAlone','Survived']].groupby('IsAlone').mean().plot.bar()
f,ax=plt.subplots(1,2,figsize=(12,5))

sns.factorplot('Total_Members','Survived',data=train,ax=ax[0])

sns.factorplot('IsAlone','Survived',data=train,ax=ax[1])
train.head()
sns.factorplot('Age_band','Survived',data=train,col='Pclass')

plt.show()
sns.factorplot('IsAlone','Survived',data=train,hue='Sex',col='Pclass')

plt.show()
train['Fare_Range']=pd.qcut(train['Fare'],4)

train[['Fare_Range','Survived']].groupby(by='Fare_Range').mean()
train['Fare_cat']=0

train.loc[train['Fare']<=7.91,'Fare_cat']=0

train.loc[(train['Fare']>7.91)&(train['Fare']<=14.454),'Fare_cat']=1

train.loc[(train['Fare']>14.454)&(train['Fare']<=31),'Fare_cat']=2

train.loc[(train['Fare']>31)&(train['Fare']<=513),'Fare_cat']=3
train.head()
sns.factorplot('Fare_cat','Survived',hue='Sex',data=train)
train['Embarked'].isnull().sum()
train['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace=True)
train['Embarked'].isnull().sum()
train.head()
elements=['PassengerId','Name','Ticket','Cabin','Fare_Range']

train.drop(elements,axis=1,inplace=True)
train.drop('Fare',axis=1,inplace=True)
train.head()
train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2})
train.head()
sns.heatmap(train.corr(),annot=True)
data=train
sns.heatmap(data.isnull(),cmap='viridis',cbar=False,yticklabels=False)
from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
X=data.drop('Survived',axis=1)

y=data['Survived']
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)

model.fit(X_train,y_train)

svc_predict=model.predict(X_test)

print("The accuracy using SVM is",metrics.accuracy_score(svc_predict,y_test))
model=svm.SVC(kernel='linear',C=1.0,gamma=0.1)

model.fit(X_train,y_train)

svc_predict=model.predict(X_test)

print("The accuracy using SVM is",metrics.accuracy_score(svc_predict,y_test))
model=LogisticRegression()

model.fit(X_train,y_train)

logmodel_predict=model.predict(X_test)

print("The accuracy using Logistic Regression is",metrics.accuracy_score(logmodel_predict,y_test))
rfc=RandomForestClassifier(n_estimators=20)

rfc.fit(X_train,y_train)

rfc_predict=rfc.predict(X_test)

print("The accuracy score using random forest classifier is",metrics.accuracy_score(rfc_predict,y_test))
model=KNeighborsClassifier()

model.fit(X_train,y_train)

knn_predict=model.predict(X_test)

print("The accuracy score using KNN is",metrics.accuracy_score(knn_predict,y_test))
a_index=list(range(1,11))

a=pd.Series()

x=[0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(X_train,y_train)

    prediction=model.predict(X_test)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))

plt.plot(a_index, a)

plt.xticks(x)

plt.show()

print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
model=GaussianNB()

model.fit(X_train,y_train)

prediction6=model.predict(X_test)

print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,y_test))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import KFold
kfold=KFold(n_splits=10,random_state=0) #Splits the data into 10 equal parts
mean=[]

accuracies=[]

std=[]

classifiers=['rbf SVC','linear SVC','Logistic Regression','Decision Tree','Random Forest Classifer','KNN','Gaussian Naive Bayes']

models=[svm.SVC(kernel='rbf'),svm.SVC(kernel='linear'),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=20),

       KNeighborsClassifier(n_neighbors=8),GaussianNB()]

for i in models:

    model=i

    cross_validation_result=cross_val_score(model,X,y,scoring="accuracy",cv=kfold)

    mean.append(cross_validation_result.mean())

    std.append(cross_validation_result.std())

    accuracies.append(cross_validation_result)

result=pd.DataFrame({'CV Mean':mean,'CV Std':std},index=classifiers)

result
result['CV Mean'].plot.barh(width=0.8)

plt.title('CV Mean')
f,ax=plt.subplots(3,3,figsize=(12,10))

y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')

ax[0,0].set_title('Matrix for rbf-SVM')

y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')

ax[0,1].set_title('Matrix for Linear-SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')

ax[0,2].set_title('Matrix for KNN')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')

ax[1,0].set_title('Matrix for Random-Forests')

y_pred = cross_val_predict(LogisticRegression(),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')

ax[1,1].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')

ax[1,2].set_title('Matrix for Decision Tree')

y_pred = cross_val_predict(GaussianNB(),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')

ax[2,0].set_title('Matrix for Naive Bayes')

plt.subplots_adjust(hspace=0.2,wspace=0.2)

plt.show()
from sklearn.model_selection import GridSearchCV

C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel=['rbf','linear']

hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)

gd.fit(X,y)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators=range(100,1000,100)

hyper={'n_estimators':n_estimators}

gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True,cv=10)

gd.fit(X,y)

print(gd.best_score_)

print(gd.best_estimator_)
from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),

                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),

                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),

                                              ('LR',LogisticRegression(C=0.05)),

                                              ('DT',DecisionTreeClassifier(random_state=0)),

                                              ('NB',GaussianNB()),

                                              ('svm',svm.SVC(kernel='linear',probability=True))

                                             ], 

                       voting='soft').fit(X_train,y_train)

pred=ensemble_lin_rbf.predict(X_test)

print("The accuracy using matrix is",metrics.accuracy_score(pred,y_test))

cross=cross_val_score(ensemble_lin_rbf,X_test,y_test, cv = 10)

print('The cross validated score is',cross.mean())
from sklearn.ensemble import BaggingClassifier

model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,y_test))

result=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for bagged KNN is:',result.mean())
from sklearn.ensemble import BaggingClassifier

bag_model=BaggingClassifier(base_estimator=RandomForestClassifier(),random_state=0,n_estimators=100)

bag_model.fit(X_train,y_train)

predict=bag_model.predict(X_test)

print("The accuracy of the Bagged decision tree is",metrics.accuracy_score(predict,y_test))

cross=cross_val_score(bag_model,X,y)
from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

result=cross_val_score(ada,X,y,cv=10,scoring='accuracy')

print('The cross validated score for AdaBoost is:',result.mean())
from sklearn.ensemble import GradientBoostingClassifier

grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)

result=cross_val_score(grad,X,y,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting is:',result.mean())
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test['Sex']=test['Sex'].map({'male':1,'female':0}).astype(int)
age_avg=test['Age'].mean()

age_std=test['Age'].std()

null_counts=test['Age'].isnull().sum()



age_null_random_list=np.random.randint(low=age_avg-age_std,high=age_avg+age_std,size=null_counts)
test['Age'][np.isnan(test['Age'])] = age_null_random_list
test['Age'].isnull().sum()
pd.qcut(test['Age'],5)
test['Age_band']=0

test.loc[test['Age']<=20,'Age_band']=0

test.loc[(test['Age']>20)&(test['Age']<=25),'Age_band']=1

test.loc[(test['Age']>25)&(test['Age']<=30.2),'Age_band']=2

test.loc[(test['Age']>30.2)&(test['Age']<=40.3),'Age_band']=3

test.loc[test['Age']>40.3,'Age_band']=4

test.head(2)
test.drop(['Age'],axis=1,inplace=True)
test['Total_Members']=test['Parch']+test['SibSp']+1
test['IsAlone']=0

test.loc[test['Total_Members']==1,'IsAlone']=1
test['Fare_Range']=pd.qcut(test['Fare'],4)
test['Fare_cat']=0

test.loc[test['Fare']<=7.896,'Fare_cat']=0

test.loc[(test['Fare']>7.91)&(test['Fare']<=14.454),'Fare_cat']=1

test.loc[(test['Fare']>14.454)&(test['Fare']<=31.5),'Fare_cat']=2

test.loc[(test['Fare']>31.5)&(test['Fare']<=513),'Fare_cat']=3
test.head()
test['Embarked'].isnull().sum()
elements=['PassengerId','Name','Ticket','Cabin','Fare_Range']

test.drop(elements,axis=1,inplace=True)
test.drop('Fare',axis=1,inplace=True)
test.head()
test.isnull().sum()
test['Embarked']=test['Embarked'].map({'S':0,'C':1,'Q':2})
predictions=bag_model.predict(test)#using bagged decision tree
predictions
test_for_id=pd.read_csv('/kaggle/input/titanic/test.csv')
Passenger_Id=test_for_id['PassengerId']
submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head()
submission=pd.DataFrame({"PassengerId":Passenger_Id,"Survived":predictions})
submission.head()
filename = 'Titanic Predictions.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)