# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

df=df_train.copy()

test=df_test.copy()

df_gen_sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
print(df_train.head())

df_train['Embarked'].value_counts()

#df_train['Cabin'].value_counts()
print(df_train.shape)

print(df_test.shape)

df_train.Cabin.unique()
df_train.info()

df_test.info()
df_train['Age']=df_train['Age'].fillna(df_train['Age'].mean())

df_train['Age'] = df_train['Age'].astype(int)

df_test['Age']=df_test['Age'].fillna(df_test['Age'].mean())

df_test['Age'] = df_test['Age'].astype(int)

#--------------------------------------------------------------------------

df_train['AgeBin'] = pd.qcut(df_train['Age'], 5)

df_test['AgeBin'] = pd.qcut(df_test['Age'], 5)



label = LabelEncoder()

df_train['AgeBin_Code'] = label.fit_transform(df_train['AgeBin'])

df_test['AgeBin_Code'] = label.fit_transform(df_test['AgeBin'])



df_train['AgeBin_Code'] = df_train['AgeBin_Code'][:891]

df_test['AgeBin_Code'] = df_test['AgeBin_Code'][:418]



df_train.drop(['Age','AgeBin'], 1, inplace=True)

df_test.drop(['Age','AgeBin'], 1, inplace=True)
df_test.shape
df_train['Fare']=df_train['Fare'].fillna(df_train['Fare'].mean())

df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].mean())

#-----------------------------------------------------------------------

df_train['FareBin'] = pd.qcut(df_train['Fare'], 5)

df_test['FareBin'] = pd.qcut(df_test['Fare'], 5)



label = LabelEncoder()

df_train['FareBin_Code'] = label.fit_transform(df_train['FareBin'])

df_test['FareBin_Code'] = label.fit_transform(df_test['FareBin'])



df_train['FareBin_Code'] = df_train['FareBin_Code'][:891]

df_test['FareBin_Code'] = df_test['FareBin_Code'][:418]



df_train.drop(['Fare','FareBin'], 1, inplace=True)

df_test.drop(['Fare','FareBin'], 1, inplace=True)
df_train.head()
df_train['Cabin']=df_train['Cabin'].fillna('S')

df_test['Cabin']=df_test['Cabin'].fillna('S')
df_train['Embarked']=df_train['Embarked'].fillna('S')

df_test['Embarked']=df_test['Embarked'].fillna('S')

#--------------------------------------------------------------------------------

dummy1_e1=pd.get_dummies(df_train['Embarked'])

dummy1_e1.columns=['Embarked_C','Embarked_Q','Embarked_S']

df_train=pd.concat([df_train,dummy1_e1],axis=1)

dummy1_e2=pd.get_dummies(df_test['Embarked'])

dummy1_e2.columns=['Embarked_C','Embarked_Q','Embarked_S']

df_test=pd.concat([df_test,dummy1_e2],axis=1)
dummy1_s1=pd.get_dummies(df_train['Sex'])

df_train=pd.concat([df_train,dummy1_s1],axis=1)

dummy1_s2=pd.get_dummies(df_test['Sex'])

df_test=pd.concat([df_test,dummy1_s2],axis=1)
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']+1

df_train['IsAlone'] = 0

df_train.loc[df_train['FamilySize'] == 1, 'IsAlone'] = 1

#------------------------------------------------------------------

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']+1

df_test['IsAlone'] = 0

df_test.loc[df_test['FamilySize'] == 1, 'IsAlone'] = 1
df_train['Name_len'] = df_train.Name.apply(len)

df_test['Name_len'] = df_test.Name.apply(len)
def get_title(string):

    if '.' in string:

        return string.split(',')[1].split('.')[0].strip()

    else:

        return 'N.F'

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Dona', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    

    

df_train['Ticket'] = df_train['Ticket'].apply(lambda x: len(x))

df_train['Title'] = df_train['Name'].apply(get_title)

df_train['Title'] = df_train.apply(replace_titles, axis = 1)

df_test['Ticket'] = df_test['Ticket'].apply(lambda x: len(x))

df_test['Title'] = df_test['Name'].apply(get_title)

df_test['Title'] = df_test.apply(replace_titles, axis = 1)

dummy1_t1=pd.get_dummies(df_train['Title'])

df_train=pd.concat([df_train,dummy1_t1],axis=1)

dummy1_t2=pd.get_dummies(df_test['Title'])

df_test=pd.concat([df_test,dummy1_t2],axis=1)
df_train.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
y=df_train.Survived

drop_elements = ['PassengerId', 'Name','Cabin','Embarked','Sex','Title']

df_train = df_train.drop(drop_elements, axis = 1)

df_test =  df_test.drop(drop_elements, axis = 1)

X=df_train.drop('Survived',axis=1).values

X_test=df_test.values
std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

#y=std_scaler.fit_transform(y)

X_test = std_scaler.transform(X_test)
lr=LogisticRegression(C = 1, penalty='l2', solver='liblinear')
lr.fit(X,y)
Y_pred=lr.predict(X_test)
lr.score(X_test,Y_pred)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)