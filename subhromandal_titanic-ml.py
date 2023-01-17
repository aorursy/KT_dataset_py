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
train= pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
import seaborn as sns

sns.heatmap(train.isnull())
train_new=train.drop(['Cabin','Name','Ticket'],axis=1)
train_new.head(4)
sns.heatmap(train_new.isnull())
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

    
train_new['Age'] = train_new[['Age','Pclass']].apply(impute_age,axis=1)
train_new.head()
sex=pd.get_dummies(train_new['Sex'],drop_first=True)

embarked=pd.get_dummies(train_new['Embarked'],drop_first=True)
train_recent=pd.concat([train_new,sex,embarked],axis=1)
train1=train_recent.drop(['Sex','Embarked'],axis=1).head()
train1.head()
test=pd.read_csv('/kaggle/input/titanic/test.csv')

test.head(5)
sns.heatmap(test.isnull())
import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))

sns.boxplot(x='Pclass',y='Age',data=test,palette='winter',dodge=True,linewidth=2)
def impute_age1(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 42



        elif Pclass == 2:

            return 25



        else:

            return 22



    else:

        return Age
test['Age'] = test[['Age','Pclass']].apply(impute_age1,axis=1)
test.head(3)
sex=pd.get_dummies(test['Sex'],drop_first=True)

embarked=pd.get_dummies(test['Embarked'],drop_first=True)
test_recent=pd.concat([test,sex,embarked],axis=1)
test1=test_recent.drop(['Sex','Embarked','Name','Cabin','Ticket'],axis=1)

test1.head(3)
train1.dropna(inplace=True)
test1.dropna(inplace=True)
test1.head(3)
from sklearn.ensemble import RandomForestClassifier
train1.columns
x_train=train1.drop('Survived',axis=1)

y_train=train1['Survived']

x_test=test1
rfc=RandomForestClassifier(n_estimators=800,max_depth=5, random_state=1)
rfc.fit(x_train,y_train)
predictions=rfc.predict(x_test)
output = pd.DataFrame({'PassengerId': test1.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv',index=False)