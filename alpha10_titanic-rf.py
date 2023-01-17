# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.info()
plt.figure(figsize=(10,5))
sns.heatmap(train_df.isnull())
train_df.drop('Cabin',axis=1,inplace=True)
# Cabin column can be removed from the dataset as there are too many null values to be imputed
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train_df)
def impute_age(cols):
    Pclass = cols[0]
    Age = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 28
        else:
            return 25
    else:
        return Age
train_df['Age'] = train_df[['Pclass','Age']].apply(impute_age,axis=1)
plt.figure(figsize=(10,5))
sns.heatmap(train_df.isnull())
train_df.head()
sex = pd.get_dummies(train_df['Sex'],drop_first=True)
embark = pd.get_dummies(train_df['Embarked'],drop_first=True)

train_df.drop(['Name','Sex','Embarked','Ticket'],axis =1,inplace=True)
train_df.head()
train = pd.concat([train_df,sex,embark],axis=1)
train.head()
test_df = pd.read_csv('../input/test.csv')
test_df.info()
plt.figure(figsize=(10,5))
sns.heatmap(test_df.isnull())
test_df.drop('Cabin',axis=1,inplace=True)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train_df)
def impute_age_test(cols):
    Pclass = cols[0]
    Age = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 28
        else:
            return 27
    else:
        return Age
test_df['Age'] = test_df[['Pclass','Age']].apply(impute_age_test,axis=1)
test_df['Fare'].fillna(value=test_df['Fare'].mean(),inplace = True)
test_df.info()
sex = pd.get_dummies(test_df['Sex'],drop_first=True)
embark = pd.get_dummies(test_df['Embarked'],drop_first=True)

test_df.drop(['Name','Sex','Embarked','Ticket'],axis =1,inplace=True)
test = pd.concat([test_df,sex,embark],axis=1)
test.head()
test.info()
train.info()
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=1000,max_depth=8)
m.fit(X_train,y_train)
y_pred = m.predict(X_valid)
def display_confusion_matrix(sample_test, prediction, score=None):
    cm = metrics.confusion_matrix(sample_test, prediction)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if score:
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
    print(metrics.classification_report(sample_test, prediction))
from sklearn import metrics
score = metrics.accuracy_score(y_valid,y_pred)
display_confusion_matrix(y_valid,y_pred,score = score)

y_pred_test = m.predict(test)

result_df = test.copy()
result_df['Survived'] = y_pred_test
result_df.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)
