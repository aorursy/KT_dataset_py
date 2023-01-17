#importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
df_train.info()
df_train.describe()
df_train.shape
df_train.isnull().sum()
df_test.head()
df_test.shape
df_test.info()
df_test.describe()
df_test.isnull().sum()
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Survived',data=df_train,palette='Set3')
plt.figure(figsize=(18,5))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(141)

sns.countplot(x='Survived',hue='Sex',data=df_train,palette='dark')

plt.subplot(142)

sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='RdBu_r')

plt.subplot(143)

sns.countplot(x='Survived',hue='Embarked',data=df_train,palette='Set3')


sns.boxplot(x='Survived',y='Age',data=df_train)

plt.figure(figsize=(18,6))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(131)

sns.countplot(x='SibSp',data=df_train)



plt.subplot(132)

df_train['Age'].hist(bins=30,color='darkred',alpha=0.7,grid=False)

plt.xlabel('Age')



plt.subplot(133)

df_train['Fare'].hist(color='green',bins=40,grid=False)

plt.xlabel('Fare')

sns.pairplot(df_train)
sns.violinplot(y='Age',data=df_train,color='m',linewidth=2)
plt.figure(figsize=(20,5))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(141)

sns.countplot(x='Survived',hue='Sex',data=df_train)

plt.subplot(142)

sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='dark')

plt.subplot(143)

sns.countplot(x='Survived',hue='Embarked',data=df_train,palette='Set3')
plt.figure(figsize=(10,5))

sns.countplot(x='Survived',hue='Parch',data=df_train)
df_train.isnull().sum()
#dropping Cabin Column

df_train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=df_train,palette='winter')
print("mean of AGE for Pclass=1  ",df_train[df_train['Pclass']==1]['Age'].mean())

print("mean of AGE for Pclass=2  ",df_train[df_train['Pclass']==2]['Age'].mean())

print("mean of AGE for Pclass=3  ",df_train[df_train['Pclass']==3]['Age'].mean())
#impute_age function impute mean_age with respect to corresponding Pclass



def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train.fillna(method='ffill',inplace=True)
df_train.info()
df_test.isnull().sum()
#dropping Cabin Column

df_test.drop('Cabin',axis=1,inplace=True)
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=df_test,palette='winter')
print("mean of AGE for Pclass=1  ",df_test[df_test['Pclass']==1]['Age'].mean())

print("mean of AGE for Pclass=2  ",df_test[df_test['Pclass']==2]['Age'].mean())

print("mean of AGE for Pclass=3  ",df_test[df_test['Pclass']==3]['Age'].mean())
#impute_age function impute mean_age with respect to corresponding Pclass



def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 40



        elif Pclass == 2:

            return 28



        else:

            return 24



    else:

        return Age
df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_test.fillna(method='ffill',inplace=True)
df_test.info()
#converting sex col and embarked col using pd.get_dummies() method

sex = pd.get_dummies(df_train['Sex'],prefix='Sex',prefix_sep='_',drop_first=True)



embark = pd.get_dummies(df_train['Embarked'],prefix='Embark',prefix_sep='_',drop_first=True)

df_train = pd.concat([df_train,sex,embark],axis=1)
df_train.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)
df_train.head()
#converting sex col and embarked col using pd.get_dummies() method

sex = pd.get_dummies(df_test['Sex'],prefix='Sex',prefix_sep='_',drop_first=True)



embark = pd.get_dummies(df_test['Embarked'],prefix='Embark',prefix_sep='_',drop_first=True)

df_test = pd.concat([df_test,sex,embark],axis=1)
df_test.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)
df_test.head()
X = df_train.drop('Survived',axis =1)

Y = df_train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.3, random_state =101)
from sklearn.linear_model import LogisticRegression 



logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)
logpredictions = logmodel.predict(X_test)
logpredictions
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(logpredictions,Y_test))

print(classification_report(logpredictions,Y_test))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
dtc_predictions = dtc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(dtc_predictions,Y_test))

print(classification_report(dtc_predictions,Y_test))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
rfc_predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(rfc_predictions,Y_test))

print(classification_report(rfc_predictions,Y_test))
from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train,Y_train)
svm_predictions = svc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(svm_predictions,Y_test))

print(classification_report(svm_predictions,Y_test))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,Y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(grid_predictions,Y_test))

print(classification_report(grid_predictions,Y_test))
final_predictions = rfc.predict(df_test)
final_predictions
df_test['Survived'] = final_predictions
df_submission = pd.DataFrame(df_test.drop(['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embark_Q','Embark_S'],axis=1))
df_submission.to_csv('Submission_file.csv',index=False)