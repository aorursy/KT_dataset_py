# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.info()
sns.countplot(x='Survived',hue='Pclass',data=train_data,palette='rainbow');
import re

train_data['Title'] = train_data['Name'].apply(lambda x: (re.search('([a-zA-Z]+)\.', x)).group(1))

train_data['Title'].value_counts()


titles = {'Capt':       'Officer',

          'Col':        'Officer',

          'Major':      'Officer',

          'Jonkheer':   'Royalty',

          'Don':        'Royalty',

          'Sir' :       'Royalty',

          'Dr':         'Officer',

          'Rev':        'Officer',

          'Countess':   'Royalty',

          'Dona':       'Royalty',

          'Mme':        'Mrs',

          'Mlle':       'Miss',

          'Ms':         'Mrs',

          'Mr' :        'Mr',

          'Mrs' :       'Mrs',

          'Miss' :      'Miss',

          'Master' :    'Master',

          'Lady' :      'Royalty'

                    }



# train_data['Title'] = train_data['Title'].map(titles)

# train_data['Title'].value_counts()

#map method is leaving Officer and loyalty so i dumped the method



for key,value in titles.items():

    train_data.loc[train_data['Title'] == key, 'Title'] = value

    

#New frequencies.

train_data['Title'].value_counts()
# Had forgetten bout test data lol

test_data['Title'] = test_data['Name'].apply(lambda x: (re.search(' ([a-zA-Z]+)\.', x)).group(1))

for key,value in titles.items():

    test_data.loc[test_data['Title'] == key, 'Title'] = value
sns.countplot(x='Survived',hue='Title',data=train_data,palette='rainbow');
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis');
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train_data,palette='winter');
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
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)

test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)
# checking the heatmap again

sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis');
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)
train_data.dropna(inplace=True)

test_data.dropna(inplace=True)
train_data.head()
sex    = pd.get_dummies(train_data['Sex'],drop_first=True)

embark = pd.get_dummies(train_data['Embarked'],drop_first=True)

title  = pd.get_dummies(train_data['Title'],drop_first=True)



sex_1    = pd.get_dummies(test_data['Sex'],drop_first=True)

embark_1 = pd.get_dummies(test_data['Embarked'],drop_first=True)

title_1  = pd.get_dummies(test_data['Title'],drop_first=True)
train_data.drop(['Sex','Embarked','Name','Ticket','Title'],axis=1,inplace=True)

test_data.drop(['Sex','Embarked','Name','Ticket','Title'],axis=1,inplace=True)
train_data = pd.concat([train_data,sex,embark,title],axis=1)

test_data  = pd.concat([test_data,sex,embark,title],axis=1)



train_data.tail()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data.drop('Survived',axis=1), 

                                                    train_data['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(len(predictions))
test_data.info()
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)