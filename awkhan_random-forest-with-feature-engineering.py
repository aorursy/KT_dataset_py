# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
test=pd.read_csv('../input/test.csv')

train=pd.read_csv('../input/train.csv')

train.head()
print(test.shape, train.shape)
train.head()
m_1=train[train['Pclass']==1]['Age'].median()

m_2=train[train['Pclass']==2]['Age'].median()

m_3=train[train['Pclass']==3]['Age'].median()

fm_1=test[test['Pclass']==1]['Age'].median()

fm_2=test[test['Pclass']==2]['Age'].median()

fm_3=test[test['Pclass']==3]['Age'].median()







def imputeage(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass ==1:

            return m_1

        elif Pclass ==2:

            return m_2

        else:

            return m_3

        

    else:

        return Age





def imputeagef(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass ==1:

            return fm_1

        elif Pclass ==2:

            return fm_2

        else:

            return fm_3

        

    else:

        return Age
train['Age']=train[['Age','Pclass']].apply(imputeage,axis=1)

test['Age']=test[['Age','Pclass']].apply(imputeagef,axis=1)
from sklearn.preprocessing import LabelEncoder

sex_encoder=LabelEncoder()



sex_encoder.fit(list(train['Sex'].values) +list(test['Sex'].values))
train['Sex']=sex_encoder.transform(train['Sex'].values)

test['Sex']=sex_encoder.transform(test['Sex'].values)
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
test.head()
import re
def title_extract(name):

    x =re.search(', (.+?)\.', name)

    if x:

        return x.group(1)

    else:

        return ''
test['Title']=test['Name'].apply(title_extract)

train['Title']=train['Name'].apply(title_extract)
train.head()
train['Embarked']=train['Embarked'].fillna('Z')
fm=test['Fare'].median()

test['Fare']=test['Fare'].fillna(fm)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'Embarked']

dv=DictVectorizer()

dv.fit(train[feature_names].append(test[feature_names]).to_dict(orient='records'))

dv.feature_names_

rfor=RandomForestClassifier(random_state=40, n_estimators=300, max_depth=5)

X_train=dv.transform(train[feature_names].to_dict(orient='records'))

y_train=train['Survived']

X_test=dv.transform(test[feature_names].to_dict(orient='records'))

rfor.fit(X_train,y_train)
pred=rfor.predict(X_test)
pred.shape
my_predicitons=pd.DataFrame()

my_predicitons['PassengerId']=test['PassengerId']

my_predicitons['Survived']=pred

my_predicitons.head()
my_predicitons.to_csv('My_RF_titanic_predictions', index=False)