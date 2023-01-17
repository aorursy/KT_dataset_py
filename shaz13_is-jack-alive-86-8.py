#Standard imports 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('test.csv')
train.head()
plt.figure(figsize=(12,8))

sns.heatmap(train.isnull(),cmap='viridis')
def sin_bin(x):

    if x =='male':

        return 1

    else:

        return 0
train['Sex'] = train['Sex'].apply(sin_bin)
plt.figure(figsize=(12,8))

sns.distplot(train['Age'].dropna())
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow')
plt.figure(figsize=(12, 7))

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age

    

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
def emark_num(x):

    if x is 'C':

        return 1

    elif x is 'Q':

        return 2

    else:

        return 3

    

train['Embarked'] = train['Embarked'].apply(emark_num)
# trim 1

# train.drop(['PassengerId','Name'],axis=1, inplace=True)

train.drop(['Name','Ticket'],axis=1,inplace=True)
def is_cab(z):

    

    if isinstance(z, float):

        return 0

    else:

        return 1



train['Cabin'] = train['Cabin'].apply(is_cab)
# Checking the head for cleansed data

train.head()
# Checking the heatmap to verify the absence of null data

plt.figure(figsize=(12, 7))

sns.heatmap(train.isnull(), cmap='viridis')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.333, 

                                                    random_state=2014) 
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

print(accuracy_score(y_test,predictions)*100)
# accuracy_scores = []

# for i in range(1,3000):

#     from sklearn.model_selection import train_test_split



#     X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

#                                                         train['Survived'], test_size=0.333, 

#                                                         random_state=i) 

#     from sklearn.linear_model import LogisticRegression

#     logmodel = LogisticRegression()

#     logmodel.fit(X_train,y_train)

#     predictions = logmodel.predict(X_test)

#     from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

#     accuracy_scores.append(accuracy_score(y_test,predictions))
# max(accuracy_scores)