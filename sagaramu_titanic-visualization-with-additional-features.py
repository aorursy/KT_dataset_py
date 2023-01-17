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
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
Titanic =  pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)

len(Titanic)

Titanic.head()
Titanic['Survived'].value_counts()
Titanic.info()
Test =  pd.read_csv("/kaggle/input/titanic/test.csv",index_col=0)

print(len(Test))

Titanic.head()
Test.info()
Feature =  Titanic.drop(['Survived'],axis=1)

Feature = pd.concat([Feature,Test])
Feature.info()
sns.set()

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,4), dpi=100)

sns.distplot(Feature[Feature['Age'].notnull()]['Age'],ax=axes[0],kde=False)

sns.distplot(Feature[Feature['Fare'].notnull()]['Fare'],ax=axes[1],kde=False)
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(10,6), dpi=100)

# sns.set_context('notebook',font_scale=1)

# sns.set_style('whitegrid')

sns.countplot(x='Sex',data=Titanic,ax=axes[0][0])

sns.countplot(x='Pclass',data=Titanic,ax=axes[0][1])

sns.countplot(x='Survived',data=Titanic,ax=axes[0][2])

sns.countplot(x='SibSp',data=Titanic,ax=axes[1][0])

sns.countplot(x='Parch',data=Titanic,ax=axes[1][1])

sns.countplot(x='Embarked',data=Titanic,ax=axes[1][2])
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(10,6), dpi=100)

# sns.set_context('notebook',font_scale=1)

# sns.set_style('whitegrid')

sns.countplot(x='Sex',data=Titanic,ax=axes[0][0],hue='Survived')

sns.countplot(x='Pclass',data=Titanic,ax=axes[0][1],hue='Survived')

sns.countplot(x='Survived',data=Titanic,ax=axes[0][2])

sns.countplot(x='SibSp',data=Titanic,ax=axes[1][0],hue='Survived')

sns.countplot(x='Parch',data=Titanic,ax=axes[1][1],hue='Survived')

sns.countplot(x='Embarked',data=Titanic,ax=axes[1][2],hue='Survived')
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4), dpi=100)

sns.stripplot(x="Sex", y="Age", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[0])

sns.stripplot(x="Sex", y="Fare", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[1])
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4), dpi=100)

sns.stripplot(x="Embarked", y="Age", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[0])

sns.stripplot(x="Embarked", y="Fare", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[1])
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4), dpi=100)

sns.stripplot(x="Parch", y="Age", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[0])

sns.stripplot(x="Parch", y="Fare", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[1])
Corr = Titanic.corr()
fig=plt.figure(figsize =(10,10),dpi=65)

sns.heatmap(Corr,cmap='summer',linewidths=1,linecolor='black',annot=True)
Feature['Age'].fillna(value =Feature['Age'].median(),inplace=True )
Feature['Fare'].fillna(value =Feature['Fare'].median(),inplace=True)
Feature['Embarked'].fillna(value ='S',inplace=True)
Feature.info()
Feature['Family'] = Feature['SibSp']+Feature['Parch']
Feature['Salutation'] = Feature['Name'].str.split(",").str[1].str.split().str[0]
Feature.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

sccalerm = MinMaxScaler()
Feature['Fare']=sccalerm.fit_transform(Feature[['Fare']])
Feature['Age']=sccalerm.fit_transform(Feature[['Age']])
Feature.head()
Feature.drop(['Name','Cabin','Ticket'],inplace=True,axis=1)
Feature.drop(['Age'],inplace=True,axis=1)
Feature.head()
X = Feature.iloc[:,:].values
X
LabelCol = [1,5,7]

for i in LabelCol:

    LabelX = LabelEncoder()

    X[:,i]=LabelX.fit_transform(X[:,i])
X
Xtest = X[len(Titanic):]
X1 = X[0:len(Titanic)]
Y=Titanic.iloc[:,0].values
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
def plot_roc_curve(fpr, tpr,col,lab):

    plt.plot(fpr, tpr, color=col, label=lab)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
X_train,X_test,Y_Train,Y_Test = train_test_split(X1,Y,test_size= 0.20)
classifierLogistic = LogisticRegression()

classifierLogistic.fit(X_train,Y_Train)

y_pred_logit = classifierLogistic.predict(X_test)



cm_logit = confusion_matrix(Y_Test, y_pred_logit)

print('Confusion matrix for Logistic',cm_logit)



accuracy_logit = accuracy_score(Y_Test, y_pred_logit)

precision_logit =precision_score(Y_Test, y_pred_logit)

recall_logit =  recall_score(Y_Test, y_pred_logit)

f1_logit = f1_score(Y_Test, y_pred_logit)

print('accuracy_logistic :',accuracy_logit)

print('precision_logistic :',precision_logit)

print('recall_logistic :',recall_logit)

print('f1-score_logistic :',f1_logit)

auc_logit = roc_auc_score(Y_Test, y_pred_logit)

print('AUC_logistic : %.2f' % auc_logit)
classifierRF = RandomForestClassifier(n_estimators=100,max_depth=10,criterion='entropy',class_weight='balanced')

classifierRF.fit(X_train,Y_Train)

y_pred_RF = classifierRF.predict(X_test)



cm_RF = confusion_matrix(Y_Test, y_pred_RF)

print('Confusion matrix Random Forest',cm_RF)



accuracy_RF = accuracy_score(Y_Test, y_pred_RF)

precision_RF =precision_score(Y_Test, y_pred_RF)

recall_RF =  recall_score(Y_Test, y_pred_RF)

f1_RF = f1_score(Y_Test, y_pred_RF)

print('accuracy random forest :',accuracy_RF)

print('precision random forest :',precision_RF)

print('recall random forest :',recall_RF)

print('f1-score random forest :',f1_RF)

auc_RF = roc_auc_score(Y_Test, y_pred_RF)

print('AUC: %.2f' % auc_RF)
xgb =  XGBClassifier()

xgb.fit(X_train,Y_Train)

y_pred_xgb = xgb.predict(X_test)



cm_xgb = confusion_matrix(Y_Test, y_pred_xgb)

print('Confusion matrix Random Forest',cm_xgb)



accuracy_xgb = accuracy_score(Y_Test, y_pred_xgb)

precision_xgb =precision_score(Y_Test, y_pred_xgb)

recall_xgb =  recall_score(Y_Test, y_pred_xgb)

f1_xgb = f1_score(Y_Test, y_pred_xgb)

print('XGBOOST accuracy :',accuracy_xgb)

print('precision XGBOOST :',precision_xgb)

print('recall XGBOOST :',recall_xgb)

print('f1-score XGBOOST :',f1_xgb)

auc_xgb = roc_auc_score(Y_Test, y_pred_xgb)

print('AUC: %.2f' % auc_xgb)
a=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

b=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

fig =plt.figure(figsize=(15,15),dpi=50)

fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_logit )

plt.plot(fpr, tpr,color ='orange',label ='Logistic' )

fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_RF )

plt.plot(fpr, tpr,color ='blue',label ='random Forest' )

fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_RF )

plt.plot(fpr, tpr,color ='blue',label ='XGB' )

plt.plot(a,b,color='black',linestyle ='dashed')

plt.legend(fontsize=15)

plt.xlabel('False Positive Rate',fontsize=15)

plt.ylabel('True Positive Rate',fontsize=15)
y_pred_test = xgb.predict(Xtest)
output = pd.DataFrame({'PassengerId': Test.index, 'Survived': y_pred_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
