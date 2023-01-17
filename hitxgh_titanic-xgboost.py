# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

import seaborn as sns

from sklearn.preprocessing import StandardScaler



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train_df=train.drop(['PassengerId','Survived','Cabin','Name','Ticket'],axis=1)

test_df=test.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)

label=train['Survived']

test_df.info
#Data visualization 



#Determining variables

cols=['Pclass','Sex','Age','Parch','Embarked','Fare','SibSp']



#rows and columns of subgraphs 

n_rows=1

n_cols=7



#Draw

fig,axes=plt.subplots(n_rows,n_cols,figsize=(n_cols*5,n_rows*5))



for r in range(n_rows):

    for  c in range(n_cols):

        i=r*n_cols+c

        ax=axes.flatten()[i]

        sns.countplot(train[cols[i]],hue=train['Survived'],ax=ax)

        ax.set_title(cols[i])

        ax.legend(title='Survived',loc='upper right')

plt.tight_layout()#Make the subplot fill the space 
#male:1,female:0

train_df['Sex']=train_df['Sex'].map(lambda x:1 if (x=='male') else 0)

test_df['Sex']=test_df['Sex'].map(lambda x:1 if (x=='male') else 0)
#S:0,C:1,Q:2

train_df['Embarked'].fillna(train_df['Embarked'].mode(),inplace=True)

train_df['Embarked']=train_df['Embarked'].map(lambda x:0 if (x=='S') else(1 if (x=='C') else 2))



test_df['Embarked'].fillna(test_df['Embarked'].mode(),inplace=True)

test_df['Embarked']=test_df['Embarked'].map(lambda x:0 if (x=='S') else(1 if (x=='C') else 2))
train_df['Age'].fillna(train_df['Age'].mean(),inplace=True)

test_df['Age'].fillna(test_df['Age'].mean(),inplace=True)
train_df['Fare'].fillna(train_df['Fare'].mean(),inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
print(train_df.head(20))
train_df.info
#Data visualization 



#Determining variables

cols=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']



#rows and columns of subgraphs 

n_rows=1

n_cols=7



#Draw

fig,axes=plt.subplots(n_rows,n_cols,figsize=(n_cols*5,n_rows*5))



for r in range(n_rows):

    for  c in range(n_cols):

        i=r*n_cols+c

        ax=axes.flatten()[i]

        sns.countplot(train[cols[i]],hue=train['Survived'],ax=ax)

        ax.set_title(cols[i])

        ax.legend(title='Survived',loc='upper right')

plt.tight_layout()#Make the subplot fill the space 
ss=StandardScaler()#dataframe-->ndarray

train_df=ss.fit_transform(train_df)

test_df=ss.fit_transform(test_df)
from sklearn.model_selection import cross_val_score

lr=LogisticRegression(max_iter=20000)

lr.fit(train_df,label)

scores=cross_val_score(lr,train_df,label,cv=15,scoring='accuracy')

print(scores.mean())
from sklearn.svm import LinearSVC

svm=LinearSVC(max_iter=10000)

svm.fit(train_df,label)

scores_svm=cross_val_score(svm,train_df,label,cv=20,scoring='accuracy')

print(scores_svm.mean())
from xgboost import XGBClassifier

xgbc=XGBClassifier()

xgbc.fit(train_df,label)

scores_xgb=cross_val_score(xgbc,train_df,label,cv=20,scoring='accuracy')

print(scores_xgb.mean())
from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(xgbc,train_df, label,cmap=plt.cm.Blues,normalize=None)

disp.ax_.set_title('Confusion_matrix')

plt.show()
from sklearn.metrics import plot_roc_curve

plot_roc_curve(xgbc,train_df,label)

plt.show()


predictions=xgbc.predict(test_df)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")