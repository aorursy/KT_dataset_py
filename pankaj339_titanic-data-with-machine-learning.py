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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.tail()
test.head()
train.info()
train.describe().T
train['Age'].isnull().value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from plotly.offline import download_plotlyjs,init_notebook_mode,iplot,plot

import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')
plt.figure()

fig,axes=plt.subplots(nrows=2,ncols=2, figsize=(15,15))

sns.countplot(x='Sex',hue='Survived',data=train,ax=axes[0][0])

axes[0][0].set_title('Sex Vs Survival')

sns.countplot(x='Pclass',hue='Survived',data=train,ax=axes[0][1])

axes[0][1].set_title('Pclass Vs Survival')

sns.countplot(x='Embarked',hue='Survived',data=train,ax=axes[1][0])

axes[1][0].set_title('Embarked Vs Survival')

sns.countplot(x='SibSp',hue='Survived',data=train,ax=axes[1][1])

axes[1][1].set_title('Sibling/spouse Vs Survival')

#train.iplot(kind='bar',x='Sex')
sns.boxplot(train['Pclass'], train['Age'], hue=train['Survived'])

#train.iplot(kind='box',x='Pclass',y='Age')
def age_impute(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 45

        elif Pclass==2:

            return 32

        else:

            return 27

    else:

        return Age

    

    

 
train['Age']=train[['Age','Pclass']].apply(age_impute, axis=1)
sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')
train.drop(['PassengerId','Ticket','Name','Cabin'], axis=1, inplace=True)
train.head()
fig1,axes1=plt.subplots(nrows=1,ncols=2,figsize=(12,12))

sns.barplot(x='Embarked',y='Fare',hue='Survived',data=train,ax=axes1[0])

sns.barplot(x='Pclass',y='Fare',hue='Survived',data=train,ax=axes1[1])
sns.heatmap(train.corr(), cmap='YlOrBr', annot=True, linecolor='black', linewidth=5)
train_dum=pd.get_dummies(data=train,columns=['Pclass','Sex','Embarked'], drop_first=True)
train_dum.drop('Fare', axis=1,inplace=True)
#Logistic Regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
X=train_dum.drop('Survived',axis=1)

y=train_dum['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr=LogisticRegression(solver='liblinear')
lr.fit(X_train,y_train)
predictor=lr.predict(X_test)

lr_score=lr.score(X_test,y_test)

lr_score
plt.figure(figsize=(3,3))

cm=confusion_matrix(y_test,predictor,labels=[0,1])

df=pd.DataFrame(cm,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])

sns.heatmap(df,cmap='coolwarm',cbar=False, annot=True)

print(df)
print(classification_report(y_test,predictor))
#Applying KNN

from sklearn.neighbors import KNeighborsClassifier

#Prior to applying KNN we have to scale the data

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

s=scaler.fit_transform(X)

df1=pd.DataFrame(s,columns=X.columns)
df1.head(2)
X1_train, X1_test, y1_train, y1_test = train_test_split(df1, y, test_size=0.3, random_state=42)

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X1_train,y1_train)
predict_knn=knn.predict(X1_test)

knn_score=knn.score(X1_test,y1_test)

knn_score
plt.figure(figsize=(3,3))

cm1=confusion_matrix(y1_test,predict_knn,labels=[0,1])

df2=pd.DataFrame(cm1,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])

sns.heatmap(df2,cmap='coolwarm',cbar=False, annot=True)

print(df2)
print(classification_report(y1_test,predict_knn))
# FOR K ranging from 1-40

error_rate=[]

for i in range (1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X1_train, y1_train)

    predict_i= knn.predict(X1_test)

    error_rate.append(np.mean(predict_i != y1_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')

plt.title('Error Rate vs K Value')

plt.xlabel('k')

plt.ylabel('Error Rate')
knn19=KNeighborsClassifier(n_neighbors=10)

knn19.fit(X1_train,y1_train)

predict_knn19=knn19.predict(X1_test)

knn1_score=knn19.score(X1_test,y1_test)

knn1_score

plt.figure(figsize=(3,3))

cm19=confusion_matrix(y1_test,predict_knn19,labels=[0,1])

df29=pd.DataFrame(cm19,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])

sns.heatmap(df29,cmap='coolwarm',cbar=False, annot=True)

print(df29)
print(classification_report(y1_test,predict_knn19))
#Navebayes Classifier

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics



nb=GaussianNB()

nb.fit(X_train,y_train)
naive_predict=nb.predict(X_test)
print("Model accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,naive_predict)))
naive_score=nb.score(X_test,y_test)

naive_score
plt.figure(figsize=(4,4))

cm11=metrics.confusion_matrix(y_test, naive_predict, labels=[0,1])

df_cm1=pd.DataFrame(cm11, index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])

sns.heatmap(df_cm1, annot=True, cmap='coolwarm')

df_cm1
print(classification_report(y_test, naive_predict))
#Random Forest

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
predict_rf=rf.predict(X_test)
rf_score=rf.score(X_test,y_test)

rf_score
print(classification_report(y_test,predict_rf))
cm_rf=metrics.confusion_matrix(y_test, predict_rf,labels=[0,1])

cm_rf1=pd.DataFrame(cm_rf,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])

sns.heatmap(cm_rf1, annot=True, cmap='coolwarm')

cm_rf1
#SVM (Support Vector Machine)

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
param_grid= {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_
predict_grid=grid.predict(X_test)
print(classification_report(y_test, predict_grid))
svm_score=grid.score(X_test,y_test)

svm_score
cm_svm=metrics.confusion_matrix(y_test, predict_grid,labels=[0,1])

cm_svm1=pd.DataFrame(cm_svm,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])

sns.heatmap(cm_svm1, annot=True, cmap='coolwarm')

cm_svm1
Result=pd.DataFrame({'Models':['Logistic','KNN','Navebayes','RandomForest','SVM'], 'Accuracy':[lr_score,knn1_score,naive_score,rf_score,svm_score]})
Result
test.head(2)
PassengerID=test['PassengerId']

test.drop(['PassengerId', 'Name','Ticket', 'Cabin'], axis=1,inplace=True)
test['Age']= test[['Age', 'Pclass']].apply(age_impute, axis=1)
test['Age'].isnull().any()
columns=['Sex', 'Pclass', 'Embarked']

test_dum=pd.get_dummies(test, columns=columns, drop_first=True)
test_dum.head()
test_dum.drop(['Fare'],axis=1, inplace=True)
test_dum.head()
s1=scaler.fit_transform(test_dum)

df2=pd.DataFrame(s1,columns=test_dum.columns)
df2.head()
df2.info()
knn_test=knn19.predict(df2)
combine=list(zip(PassengerID,knn_test))
output=pd.DataFrame(combine,columns=['PassengerId', 'Survived'])
output.head()
output.to_csv('Final_submission.csv',index=False)