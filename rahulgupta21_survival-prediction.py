# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV

from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,precision_score,recall_score,f1_score,accuracy_score

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import *

import xgboost as xgb

import lightgbm as lgb

warnings.filterwarnings('ignore')
train1=pd.read_csv('/kaggle/input/titanic/train.csv')

test1=pd.read_csv('/kaggle/input/titanic/test.csv')
train1.describe()                          # Training set description
test1.describe()                                           # Testing set description
train1.isnull().sum()                                     # Checking the missing values in the training set
test1.isnull().sum()                                     # Checking for the missing data in the testing set
#Removing the unnecessary attributes from the training and testing data



train1.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

test1.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
test1['Fare'].fillna(test1['Fare'].median(),inplace=True)
test1.isnull().sum()
test1.info()
train1.Age.median()
train1['Age'].fillna(train1.Age.median(),inplace=True)

test1['Age'].fillna(test1.Age.median(),inplace=True)
train1.Embarked.value_counts()
train1[train1.Embarked.isnull()]
train1.loc[61,'Embarked']='S'

train1.loc[829,'Embarked']='C'
train1.Embarked.value_counts()
train1['Fare'].median()
train1['Fare']=train1['Fare'].replace({0:14.5})
train1.Embarked.value_counts().plot.bar()
plt.figure(figsize=(7,5))

train1.Age.plot(kind='hist',bins=13,color='r',alpha=0.7,legend='best')
plt.figure(figsize=(7,5))

train1['Pclass'].value_counts().plot(kind='bar',color='y',alpha=0.7,legend='best')
plt.figure(figsize=(7,5))

d1=train1[train1['Survived']==0].Age

d1.plot(kind='hist',legend='best',bins=10,color='b',alpha=0.7)

plt.title('Frequency of people died and their age')
sns.countplot(train1['Survived'],alpha=0.6)

plt.title('Count of people died or survived')
plt.figure(figsize=(10,6))

plt.subplot(1,2,1)

sns.countplot(train1[train1['Survived']==0].Pclass,alpha=0.7)

plt.title('Total count of Pclass of people died ')

plt.subplot(1,2,2)

sns.countplot(train1[train1['Survived']==1].Pclass,alpha=0.7)

plt.title('Total count of Pclass of people survived')
sns.factorplot('Sex','Survived',hue='Pclass',alpha=0.6,data=train1)
plt.figure(figsize=(14,10))

plt.subplot(1,2,1)

train1[train1['Survived']==1].Embarked.value_counts().plot.pie()

plt.subplot(1,2,2)

train1[train1['Survived']==0].Embarked.value_counts().plot.pie()

plt.title('Distribution of Embarked of the people survived and died')
plt.figure(figsize=(8,6))

sns.barplot('Embarked','Survived',hue='Sex',alpha=0.7,data=train1)
sns.pointplot('Pclass','Survived',hue='Sex',alpha=0.7,data=train1)
train1['Total_Family_Members']=train1['SibSp']+train1['Parch']

test1['Total_Family_Members']=test1['SibSp']+test1['Parch']
sns.barplot('Total_Family_Members','Survived',alpha=0.7,data=train1)
cut=[-1,0,5,12,18,33,60,100]

label_name=['Missing','Infant','Child','Teenager','Young Adult','Adult','Senior']

train1['Age_Categories']=pd.cut(train1['Age'],cut,labels=label_name)

cutt=[-1,0,5,12,18,33,60,100]

labelt_name=['Missing','Infant','Child','Teenager','Young Adult','Adult','Senior']

test1['Age_Categories']=pd.cut(test1['Age'],cutt,labels=labelt_name)
train1[train1['Age_Categories']=='Infant'].head()
train1['Age_Categories'].value_counts().plot(kind='bar')
plt.figure(figsize=(15,7))

plt.subplot(1,2,1)

sns.barplot('Age_Categories','Survived',alpha=0.7,data=train1)

plt.subplot(1,2,2)

sns.barplot('Total_Family_Members','Survived',alpha=0.7,data=train1)
sns.factorplot('Age_Categories','Survived',hue='Sex',data=train1)
sns.barplot('Fare',data=train1)
train1['Fare'].replace({512.3292:270,4.0125:14.5,5.0000:14.5},inplace=True)
lbe=LabelEncoder()

train1['Sex']=lbe.fit_transform(train1['Sex'])

test1['Sex']=lbe.transform(test1['Sex'])



lbe=LabelEncoder()

train1['Embarked']=lbe.fit_transform(train1['Embarked'])

test1['Embarked']=lbe.transform(test1['Embarked'])



lbe=LabelEncoder()

train1['Total_Family_Members']=lbe.fit_transform(train1['Total_Family_Members'])

test1['Total_Family_Members']=lbe.transform(test1['Total_Family_Members'])



lbe=LabelEncoder()

train1['Age_Categories']=lbe.fit_transform(train1['Age_Categories'])

test1['Age_Categories']=lbe.transform(test1['Age_Categories'])
label=train1['Survived']

train1.drop(columns='Survived',inplace=True)
stsc=StandardScaler()

train=stsc.fit_transform(train1)

test=stsc.transform(test1)
train=pd.DataFrame(train)

test=pd.DataFrame(test)
xtrain,xtest,ytrain,ytest=train_test_split(train,label,test_size=0.2,random_state=50)
log_clf=LogisticRegression()

dst_clf=DecisionTreeClassifier()

rnd_clf=RandomForestClassifier()

svm_clf=LinearSVC()

gbc_clf=GradientBoostingClassifier()
log_clf.fit(xtrain,ytrain)
ylogpred=log_clf.predict(xtest)
accuracy_score(ytest,ylogpred)
confusion_matrix(ytest,ylogpred)
param_grid=[

    {'C':[0,1,2,3,4,5],'dual':[True,False],'max_iter':[70,80,90,100,110,120],'random_state':[25,50,75,100],'tol':[0.0001,0.0002,0.0005,0.0009,0.001]}]
log_grid=GridSearchCV(log_clf,param_grid,cv=3,scoring='accuracy')
log_grid.fit(xtrain,ytrain)
log_grid.best_params_
log_grid_pred=log_grid.predict(xtest)
accuracy_score(ytest,log_grid_pred)
cvpredlog=log_grid.predict_proba(xtest)
pre,rec,thr=precision_recall_curve(ytest,cvpredlog[:,1])

plt.figure(figsize=(10,6))

plt.plot(thr,pre[:-1],'r-')

plt.plot(thr,rec[:-1],'b-')

plt.title('Precision Recall curve')
fpr,tpr,thr=roc_curve(ytest,cvpredlog[:,1])

plt.figure(figsize=(10,6))

plt.title('ROC Curve')

plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],'g--',alpha=0.4)
auc(fpr,tpr)
f1_score(ytest,log_grid_pred)
cross_val_score(log_grid,xtrain,ytrain,cv=5,scoring='accuracy').mean()
confusion_matrix(ytest,log_grid_pred)
print(classification_report(ytest,log_grid_pred))
dst_clf.fit(xtrain,ytrain)
ydstpred=dst_clf.predict(xtest)
accuracy_score(ytest,ydstpred)
param_grid=[

    {'ccp_alpha':[0.0,0.2,0.4],'max_depth':[2,4,8,10],'max_leaf_nodes':[2,5,12,15],'random_state':[25,50,75,100]}]
dst_grid_clf=GridSearchCV(dst_clf,param_grid,cv=3,scoring='accuracy')
dst_grid_clf.fit(xtrain,ytrain)
dstpred_grid_clf=dst_grid_clf.predict(xtest)
dst_grid_clf.best_params_
accuracy_score(ytest,dstpred_grid_clf)
cvdst_pred=cross_val_predict(dst_grid_clf,xtrain,ytrain,cv=5)
cross_val_score(dst_grid_clf,xtrain,ytrain,cv=5,scoring='accuracy').mean()
preddst=dst_grid_clf.predict_proba(xtest)
pre,rec,thr=precision_recall_curve(ytest,preddst[:,1])

plt.figure(figsize=(10,8))

plt.title('Precision Recall Curve')

plt.plot(thr,pre[:-1])

plt.plot(thr,rec[:-1])
fpr,tpr,thr=roc_curve(ytest,preddst[:,1])

plt.figure(figsize=(10,8))

plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],'g--')
auc(fpr,tpr)
f1_score(ytest,dstpred_grid_clf)
confusion_matrix(ytest,dstpred_grid_clf)
print(classification_report(ytest,dstpred_grid_clf))
rnd_clf.fit(xtrain,ytrain)
yrndpred=rnd_clf.predict(xtest)
accuracy_score(ytest,yrndpred)
param_grid=[

    {'max_leaf_nodes':[2,4,6],'max_depth':[2,6,9],'min_samples_leaf':[1,6,9],'n_estimators':[75,100,125],'random_state':[50,75,100]}

]
rnd_grid_clf=GridSearchCV(rnd_clf,param_grid,cv=2,scoring='accuracy')
rnd_grid_clf.fit(xtrain,ytrain)
rndpred_grid_clf=rnd_grid_clf.predict(xtest)

accuracy_score(ytest,rndpred_grid_clf)
cross_val_score(rnd_grid_clf,xtrain,ytrain,cv=3,scoring='accuracy').mean()
predpro_rnd_grid_clf=rnd_grid_clf.predict_proba(xtest)
pre,rec,thr=precision_recall_curve(ytest,predpro_rnd_grid_clf[:,1])

plt.figure(figsize=(10,8))

plt.title('Precision Recall Curve')

plt.plot(thr,pre[:-1])

plt.plot(thr,rec[:-1])
fpr,tpr,thr=roc_curve(ytest,predpro_rnd_grid_clf[:,1])

plt.figure(figsize=(10,8))

plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],'g--')
auc(fpr,tpr)
f1_score(ytest,yrndpred)
confusion_matrix(ytest,yrndpred)
print(classification_report(ytest,yrndpred))
svm_clf.fit(xtrain,ytrain)
ysvmpred=svm_clf.predict(xtest)
accuracy_score(ytest,ysvmpred)
param_grid=[{

    'C':[1,2,3],'max_iter':[800,1000,1100,1200],'random_state':[10,25,50,75]

}]
svm_grid_clf=GridSearchCV(svm_clf,param_grid,cv=3,scoring='accuracy')
svm_grid_clf.fit(xtrain,ytrain)
svm_grid_clf_pred=svm_grid_clf.predict(xtest)
accuracy_score(ytest,svm_grid_clf_pred)
svmpredprob_grid_clf=svm_grid_clf.decision_function(xtest)
pre,rec,thr=precision_recall_curve(ytest,svmpredprob_grid_clf)
plt.figure(figsize=(10,8))

plt.title('Precision Recall Curve')

plt.plot(thr,pre[:-1])

plt.plot(thr,rec[:-1])

plt.legend('best')
fpr,tpr,thr=roc_curve(ytest,svmpredprob_grid_clf)
plt.figure(figsize=(10,8))

plt.title('ROC Curve')

plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],'g--')
auc(fpr,tpr)
confusion_matrix(ytest,svm_grid_clf_pred)
print(classification_report(ytest,svm_grid_clf_pred))
gbc_clf.fit(xtrain,ytrain)
gbc_clf_pred=gbc_clf.predict(xtest)

accuracy_score(ytest,gbc_clf_pred)
f1_score(ytest,gbc_clf_pred)
gbc_pred=gbc_clf.predict_proba(xtest)
pre,rec,thr=precision_recall_curve(ytest,gbc_pred[:,1])

plt.figure(figsize=(10,8))

plt.title('Precision Recall Curve')

plt.plot(thr,pre[:-1])

plt.plot(thr,rec[:-1])
fpr,tpr,thr=roc_curve(ytest,gbc_pred[:,1])

plt.figure(figsize=(10,8))

plt.title('ROC Curve')

plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],'g--')
xgbc = xgb.XGBClassifier()

xgbc.fit(xtrain,ytrain)

y_pred=xgbc.predict(xtest)

accuracy_score(ytest,y_pred)
lgbc=lgb.LGBMClassifier()

lgbc.fit(xtrain,ytrain)

y_pred=lgbc.predict(xtest)

accuracy_score(ytest,y_pred)
ada=AdaBoostClassifier()

ada.fit(xtrain,ytrain)

y_pred=ada.predict(xtest)

accuracy_score(ytest,y_pred)
ext=ExtraTreesClassifier()

ext.fit(xtrain,ytrain)

y_pred=ext.predict(xtest)

accuracy_score(ytest,y_pred)
vote_clf=VotingClassifier(estimators=[('lr',log_grid),('dt',dst_clf),('lgbc',lgbc),('gbc',gbc_clf)],voting='hard')
vote_clf.fit(xtrain,ytrain)
vote_pred=vote_clf.predict(xtest)
accuracy_score(ytest,vote_pred)
f1_score(ytest,vote_pred)
cross_val_score(vote_clf,xtrain,ytrain,cv=3,scoring='accuracy').mean()
vote_ada_clf=AdaBoostClassifier(vote_clf,algorithm='SAMME',learning_rate=0.5)
vote_ada_clf.fit(xtrain,ytrain)
vote_ada_clf_pred=vote_ada_clf.predict(xtest)
accuracy_score(ytest,vote_ada_clf_pred)
f1_score(ytest,vote_ada_clf_pred)
cross_val_score(vote_clf,xtrain,ytrain,cv=3,scoring='accuracy').mean()
gbc_cv_pred=cross_val_predict(gbc_clf,xtrain,ytrain,cv=3)
pred_final=vote_ada_clf.predict(test)
pred_final=list(pred_final)

pred_final=pd.Series(pred_final)
test_org=pd.read_csv('/kaggle/input/titanic/test.csv')

Submit=pd.concat([test_org['PassengerId'],pred_final],axis=1)

Submit.rename({0:'Survived'},axis=1,inplace=True)
Submit.to_csv('Submit.csv',index=False)
plt.figure(figsize=(7,5))

sns.countplot('Survived',data=Submit)