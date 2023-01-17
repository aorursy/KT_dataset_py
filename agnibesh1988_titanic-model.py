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
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
%matplotlib inline
sns.set_style('darkgrid')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gs = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.info()
print('='*40)
test.info()
print('='*40)
gs.info()
def missing_summary(df):
  null = df.isnull().sum()
  not_null = df.notnull().sum()
  return pd.DataFrame({'Total_missing':null,'missing %': round((null/(null+not_null))*100,2),'Total_non_missing':not_null,'total_records':null+not_null})
# Summary of test & train 
print('training_data : \n',missing_summary(train),'\n','='*40,'\n','test data : \n',missing_summary(test))
train.corr()['Survived'].sort_values().plot(kind='bar')
plt.show()
plt.figure(figsize=(10,5))
sns.boxplot(x='Pclass' , y = 'Age', data=train)
plt.show()
plt.figure(figsize=(10,6))
sns.swarmplot(y='Age', x='Survived',hue='Sex', data=train,palette='viridis')
plt.show()
train.groupby(['Sex','Pclass'])['Survived'].mean().plot(figsize=(10,4))
plt.ylabel('Avg Survival rate')
plt.show()
train.groupby('Pclass')['Age'].mean()
for i in train['Pclass'].unique():
  train[train['Pclass'] == i] = train[train['Pclass'] == i].fillna({'Age': train[train['Pclass'] == i]['Age'].dropna().median()})
train.groupby('Pclass')['Age'].mean()
print('mean of age by :', test.groupby('Pclass')['Age'].mean(),'\n\n',
      '='*40,'\n\n'
      'mean of fare by :', test.groupby('Pclass')['Fare'].mean()
      )
for i in test['Pclass'].unique():
  test[test['Pclass'] == i] = test[test['Pclass'] == i].fillna({'Age': test[test['Pclass'] == i]['Age'].dropna().median()})
  test[test['Pclass'] == i] = test[test['Pclass'] == i].fillna({'Fare': test[test['Pclass'] == i]['Fare'].mean()})
print('mean of age by :', test.groupby('Pclass')['Age'].mean(),'\n\n',
      '='*40,'\n\n'
      'mean of fare by :', test.groupby('Pclass')['Fare'].mean()
      )
print('training_data : \n',missing_summary(train),'\n','='*40,'\n','test data : \n',missing_summary(test))
train['Title'] = train['Name'].str.split(',').str[1].str.split().str[0]
test['Title'] = test['Name'].str.split(',').str[1].str.split().str[0]
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {'Mr.':1,'Miss.':2,'Mrs.':3,'Master.':4,'Rev.':5,'Col.':6,'Ms.':7,'Dr.':8,'Dona.':9, 'Don.':9}
train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)
train.isnull().sum()
# Droping column which is not required
train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)
train['Embarked'].describe()
#Deleting the records of training dataset where Embarked is missing
train.dropna(axis=0,inplace=True)
# correlation with respect to target variable
train.corr()['Survived'].sort_values()
# we already classified name and we are droping columns from both dataset
train = train.drop(['Name','Ticket'],axis=1)
test = test.drop(['Name','Ticket'],axis=1)
print('test dataset:\n',test.dtypes)
print('='*40)
print('train dataset:\n',train.dtypes)
train['Title'] = train['Title'].astype('int64')
X = pd.get_dummies(data=train.drop('Survived',axis=1),columns=['Sex','Embarked','Pclass','Title'],drop_first=True)
y = train['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train.head()
test_sample = pd.get_dummies(data=test,columns=['Sex','Embarked','Pclass','Title'],drop_first=True)
test_sample.head()
def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train == True:
        '''
        performance of training data
        '''
 
        print("Train Result:")
        print("="*60)
        print("\n")
        print("Accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        print("\n\n")
        
    else:
        '''
         performance of test data
        '''
        print("Test Result:") 
        print("="*60)
        print("\n")
        print("Accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
        
scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
model_classifiers = {'K-Nearest_Neighbors': KNeighborsClassifier(),
                     'SVM'                : SVC(),
                     'LogisticRegression' : LogisticRegression(),
                     'Gaussian_Process'   : GaussianProcessClassifier(),
                     'Gradient_Boosting'  : GradientBoostingClassifier(),
                     'Decision_Tree'      : DecisionTreeClassifier(),
                     'Extra_Trees'        : ExtraTreesClassifier(),
                     'Random_Forest'      : RandomForestClassifier(),
                     'Neural_Net'         : MLPClassifier(alpha=1, max_iter=1000),
                     'AdaBoost'           : AdaBoostClassifier(),
                     'XGBoost'            : XGBClassifier(random_state=42)}
def model_compare(x_train_df,y_train_df,x_test_df,y_test_df,model_list):
  return pd.DataFrame({'Model Name'    : [ i for i in model_list.keys()],
                       'Training Score': [accuracy_score(y_train_df,i.fit(x_train_df,y_train_df).predict(x_train_df)) for i in model_list.values()],
                       'Test Score'    : [accuracy_score(y_test_df,i.fit(x_train_df,y_train_df).predict(x_test_df)) for i in model_list.values()]
                       }
                      )
model_compare(X_train,y_train,X_test,y_test,model_classifiers)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
print_score(logreg, X_train, y_train, X_test, y_test, train=True)
print_score(logreg, X_train, y_train, X_test, y_test, train=False)
svm = SVC()
svm.fit(X_train,y_train)
print_score(svm, X_train, y_train, X_test, y_test, train=True)
print_score(svm, X_train, y_train, X_test, y_test, train=False)
test_sample = scale.transform(test_sample)
Survived = svm.predict(test_sample)
predicted_test = pd.concat([test,pd.DataFrame({'Survived':Survived.tolist()})],axis=1)[['PassengerId','Survived']].sort_values('PassengerId')
print('Confusion Matrix:\n\n', confusion_matrix(gs.Survived,predicted_test.Survived),
      '\n\n','='*40,'\n'
      'Accuracy Score:\n\n',accuracy_score(gs.Survived,predicted_test.Survived))
predicted_test.to_csv('submission.csv', index=False)
