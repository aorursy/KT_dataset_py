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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

#train.head()

import seaborn as sns

#sns.heatmap(train.isnull())
#sns.set_style('white')

#sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.countplot(x='Survived',hue='Sex',data=train)
def convert_age(cols):

    Age=cols[0]

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

        
train['Age'] = train[['Age','Pclass']].apply(convert_age,axis=1)

train.drop('Cabin',axis=1,inplace=True)

test['Age'] = test[['Age','Pclass']].apply(convert_age,axis=1)

test.drop('Cabin',axis=1,inplace=True)
sex=pd.get_dummies(test['Sex'],drop_first=True)

embark=pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test=pd.concat([test,sex,embark],axis=1)

test.head(5)
#sns.heatmap(train.isnull())
sex=pd.get_dummies(train['Sex'],drop_first=True)

embark=pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train=pd.concat([train,sex,embark],axis=1)

train.head(5)
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.copy()

X_test.fillna(X_train.mean(), inplace=True)

X_train.shape, Y_train.shape, X_test.shape
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.4, random_state=101)

#from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

#logm = LogisticRegression()

logm = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=10)

logm.fit(X_train,Y_train)
#predictions = logm.predict(X_test)

y_train_pred = logm.predict(X_train)

y_test_pred = logm.predict(X_test)
acc_log = round(logm.score(X_train, Y_train) * 100, 2)

acc_log
#coeff_df = pd.DataFrame(train.columns.delete(0))

#coeff_df.columns = ['Feature']

#coeff_df["Correlation"] = pd.Series(logm.coef_[0])



#coeff_df.sort_values(by='Correlation', ascending=False)



# Calculate the accuracy

from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(Y_train, y_train_pred)

#test_accuracy = accuracy_score(y_test, y_test_pred)

print('The training accuracy is', train_accuracy)

#print('The test accuracy is', test_accuracy)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test_pred

    })
submission.to_csv("submission.csv", index=False)

submission.tail()