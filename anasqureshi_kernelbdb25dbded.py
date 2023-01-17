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
train = pd.read_csv('../input/train.csv')

train.head(2)
test = pd.read_csv('../input/test.csv')

test.head(2)
def fill_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age 

        
train['Age']=train[['Age','Pclass']].apply(fill_age,axis=1)
test['Age']=test[['Age','Pclass']].apply(fill_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

train.head()
train.dropna(inplace=True)
test.drop('Cabin',axis=1,inplace=True)
test.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

sex.head()
sex1 = pd.get_dummies(test['Sex'],drop_first=True)

sex1.head()
embark = pd.get_dummies(train['Embarked'],drop_first=True)

embark.head()
embark1 = pd.get_dummies(test['Embarked'],drop_first=True)

embark1.head()
train = pd.concat([train,sex,embark],axis=1)
train.head()
test = pd.concat([test,sex1,embark1],axis=1)
test.head()
train.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
train.head()
test.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)
test.head()
X = train.drop('Survived',axis=1)

y = train['Survived']
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
train.dropna(axis=0,subset=['Survived'],inplace=True)
train.drop(['male','Q','S'],axis=1)
from sklearn.preprocessing import Imputer

imputer_1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_1 = imputer_1.fit(X.loc[:,['Fare']])

X.loc[:,['Fare']] = imputer_1.transform(X.loc[:,['Fare']])







imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)

imputer_1_test = imputer_1_test.fit(test.loc[:,['Fare']])

test.loc[:,['Fare']] = imputer_1_test.transform(test.loc[:,['Fare']])
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
test.drop(['male','S','Q'],axis=1)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
train.drop(['male','S','Q'],axis=1)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
imputer_1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_1 = imputer_1.fit(X.loc[:,['male']])

X.loc[:,['male']] = imputer_1.transform(X.loc[:,['male']])



imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)

imputer_1_test = imputer_1_test.fit(test.loc[:,['male']])

test.loc[:,['male']] = imputer_1_test.transform(test.loc[:,['male']])
imputer_1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_1 = imputer_1.fit(X.loc[:,['male']])

X.loc[:,['male']] = imputer_1.transform(X.loc[:,['male']])



imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)

imputer_1_test = imputer_1_test.fit(test.loc[:,['male']])

test.loc[:,['male']] = imputer_1_test.transform(test.loc[:,['male']])
imputer_1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_1 = imputer_1.fit(X.loc[:,['S']])

X.loc[:,['S']] = imputer_1.transform(X.loc[:,['S']])



imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)

imputer_1_test = imputer_1_test.fit(test.loc[:,['S']])

test.loc[:,['S']] = imputer_1_test.transform(test.loc[:,['S']])
imputer_1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_1 = imputer_1.fit(X.loc[:,['Q']])

X.loc[:,['Q']] = imputer_1.transform(X.loc[:,['Q']])



imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)

imputer_1_test = imputer_1_test.fit(test.loc[:,['Q']])

test.loc[:,['Q']] = imputer_1_test.transform(test.loc[:,['Q']])
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data

total = X.isnull().sum().sort_values(ascending=False)

percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
imp_fare = test.loc[(test.Embarked == 'S') & (test.Pclass == 3), "Fare"].mean() 



test.loc[test.Fare != test.Fare, "Fare"] = round(imp_fare, 2)

test.loc[(test.Name == "Storey, Mr. Thomas"),:]
my_submission = pd.DataFrame({'survived': model.predict(test),'passengerId':testing['PassengerId']},index =test.index)

my_submission
my_submission.to_csv('submission.csv', index=False)