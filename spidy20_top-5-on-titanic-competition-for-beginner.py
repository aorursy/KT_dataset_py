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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test['Survived'] = np.nan
data=train.append(test,ignore_index=True,sort=False)

##Add missing fare to passangerID 1044 and create new Fare_bin feature

data['Fare'].fillna(8, inplace = True) 
data['Ticket_count'] = data.Ticket.apply(lambda x: data[data['Ticket']==x].shape[0] )

data['Fare_tickect']= data.apply(lambda x: x.Fare/x.Ticket_count,axis=1 )

data['Fare_bin'] = pd.qcut(data['Fare_tickect'], 4,labels=('Fare_bin1','Fare_bin2','Fare_bin3','Fare_bin4'))
## New Title feature created 

data['Title']=data['Name'].str.split(', ').str[1].str.split('.').str[0]

data['Title'] = data['Title'].replace(['Ms','Mlle'], 'Miss')

data['Title'] = data['Title'].replace(['Mme'], 'Mrs')

data['Title'] = data['Title'].replace(['Dona','Dr','Rev','the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col'], 'Rare')
##Add missing Age bases on Pclass and create new Age_bin feature

def Cal_age(cols):

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



data['Age'] = data[['Age','Title']].apply(Cal_age,axis=1)

data['Age_bin'] = pd.cut(data['Age'].astype(int), 5, labels=('Age_bin1','Age_bin2','Age_bin3','Age_bin4','Age_bin5'))

## New Family size feature

def Cal_Family_bin(cols):

    FamilyZize = cols[0] +cols[1]

    if FamilyZize == 0:

        return 'Alone'

    elif 1 <= FamilyZize <= 3:

        return 'Family'

    elif FamilyZize >= 4:

        return 'Big_family'

    

data['Family_type'] = data[['SibSp','Parch']].apply(Cal_Family_bin,axis=1)



# Missing Embarked info.

data.loc[data['Embarked'].isnull(), 'Embarked'] = 'S'

data['Family_name']=data['Name'].str.split(', ').str[0] + '-' + data['Fare'].astype(str)



# Families with a female or child no survive.

list1=data[((data['Sex']=='female') | (data['Age']<14)) & (data['Survived']==0) ]['Family_name'].tolist()



# Families with male no child survive.

list2=data[(data['Sex']=='male') & (data['Age']>18) & (data['Survived']==1)]['Family_name'].tolist()
def FC_dead(row):

    if row['Family_name'] in list1:

        return 1

    else:

        return 0







def M_Alive(row):

    if row['Family_name'] in list2:

        return 1

    else:

        return 0



data['Family_wit_FC_dead']=data.apply(FC_dead, axis=1)

data['Family_wit_M_alive']=data.apply(M_Alive, axis=1)





data['Family_Name_size'] = data.Family_name.apply(lambda x: data[data['Family_name']==x].shape[0] )
Fare_bin = pd.get_dummies(data['Fare_bin'])

Pclass_bin = pd.get_dummies(data['Pclass'],prefix ='Class')

Title_bin = pd.get_dummies(data['Title'])

Sex_bin = pd.get_dummies(data['Sex'],drop_first=True,prefix ='Sex')

Age_bin = pd.get_dummies(data['Age_bin'])

Family_type = pd.get_dummies(data['Family_type'])

Embarked_bin = pd.get_dummies(data['Embarked'],prefix ='Embarked')

Family_wit_FC_dead=data['Family_wit_FC_dead'].astype(np.uint8)

Family_wit_M_alive=data['Family_wit_M_alive'].astype(np.uint8)
data_cleaned = pd.concat([data['Survived'],Fare_bin,Pclass_bin,Title_bin,Sex_bin,Age_bin,Family_type,Embarked_bin,Family_wit_FC_dead,Family_wit_M_alive],axis=1)



train_cleaned = data_cleaned[data['Survived'].notnull()]

test_cleaned = data_cleaned[data['Survived'].isnull()]



test_cleaned.drop('Survived',axis=1,inplace=True)

PassId =test['PassengerId']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,mean_absolute_error





X = train_cleaned.drop('Survived',axis=1)

y = train_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc.score(X_test,y_test)
from xgboost import XGBClassifier

classifier = XGBClassifier(colsample_bylevel= 0.9999999,

                    colsample_bytree = 0.9999999, 

                    gamma=0.99999,

                    max_depth= 5,

                    min_child_weight= 1,

                    n_estimators= 1000,

                    nthread= 4,

                    random_state= 2,

                    silent= True)

classifier.fit(X,y)

classifier.score(X_test,y_test)
xgb_predict = classifier.predict(test_cleaned).astype(np.uint8)


output_xgb = pd.DataFrame({ 'PassengerId' : PassId, 'Survived': xgb_predict })

output_xgb.to_csv('submission-xgb.csv', index=False)

output_xgb.head()