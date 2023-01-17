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
train_data=pd.read_csv('../input/titanic/train.csv')

test_data=pd.read_csv('../input/titanic/test.csv')
test_data1=pd.read_csv('../input/titanic/test.csv')
train_data.head()
train_data.drop('Name',axis=1,inplace=True)
test_data.head()
test_data.drop('Name',inplace=True,axis=1)
test_data.head()
train_data.head()
train_data.drop('Ticket',axis=1,inplace=True)
test_data.head()
test_data.drop('Ticket',axis=1,inplace=True)
train_data.head()
train_data.describe()
train_data.dropna(inplace=True)
test_data.head()
len(test_data)
test_data.dropna(inplace=True)
train_data.head()
len(train_data)
test_data.head()
train_data.drop('Fare',axis=1,inplace=True)
test_data.drop('Fare',axis=1,inplace=True)
train_data.head()
test_data.head()
train_data.drop('Cabin',inplace=True,axis=1)
test_data.head()
train_data.head()
def convert(x):

    if x=='female':

        return 1

    else:

        return 0
test_data.head()
train_data['Sex']=train_data['Sex'].apply(convert)
test_data['Sex']=test_data['Sex'].apply(convert)
test_data.head()
train_data.head()
train_data['Embarked'].unique()
test_data['Embarked'].unique()
def convert1(x):

    if x=='C':

        return 1

    elif x=='S':

        return 2

    else:

        return 3
train_data['Embarked']=train_data['Embarked'].apply(convert1)
test_data.head()
test_data['Embarked']=test_data['Embarked'].apply(convert1)
test_data.head()
train_data.head()
train_data['Family_size']=train_data['SibSp']+train_data['Parch']
test_data['Family_size']=test_data['SibSp']+test_data['Parch']
test_data.head()
train_data.head()
train_data.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data.head()
train_data.head()
train_data['Age'].unique()
def age_conversion(x):

    if x in range(0,20):

        return 'A'

    elif x in range(21,40):

        return 'B'

    elif x in range(41,60):

        return 'C'

    elif x in range(61,80):

        return 'D'

    else:

        return 'E'
train_data['Age']=train_data['Age'].apply(age_conversion)
test_data.head()
test_data['Age']=test_data['Age'].apply(age_conversion)
test_data.head()
train_data.head()
def convert_age22(x):

    if x=='A':

        return 1

    elif x=='B':

        return 2

    elif x=='C':

        return 3

    elif x=='D':

        return 4

    elif x=='E':

        return 5
train_data['Age_categories']=train_data['Age'].apply(convert_age22)
test_data['Age_categories']=test_data['Age'].apply(convert_age22)
test_data.head()
train_data.head()
train_data.drop('Age',axis=1,inplace=True)
test_data.drop('Age',axis=1,inplace=True)
test_data.head()
train_data.head()
train_data.drop('PassengerId',inplace=True,axis=1)
# test_data.drop('PassengerId',inplace=True,axis=1)
test_data.head()
train_data.head()
test_data.head()
test_data.drop('Cabin',inplace=True,axis=1)
test_data.head()
test_data.drop('PassengerId',inplace=True,axis=1)
test11=test_data['PassengerId']
test11=test11.values
test11=np.array(test11)
test11
test_data.head()
test_data_values=test_data.values
test_data_values
X=train_data.drop('Survived',axis=1).values

Y=train_data['Survived'].values
X=np.array(X)

Y=np.array(Y)
Y.dtype
X.shape,Y.shape
from sklearn.ensemble import RandomForestClassifier
randfort=RandomForestClassifier(n_estimators=50,random_state=0)
randfort.fit(X,Y)
test_data_values.dtype
test_data_value=np.array(test_data_values)
predict=randfort.predict(test_data_value)
submiens=pd.DataFrame({'PassengerId':test11,'Survived':predict})
submiens.head()
submiens.to_csv("submission.csv",index=False)