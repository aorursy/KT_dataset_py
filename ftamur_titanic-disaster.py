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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

import pandas as pd
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
y = train['Survived'].values

y = y.reshape((y.shape[0], 1))

print("y.shape: {}".format(y.shape))
test_y = np.full([test.shape[0], 1], fill_value=np.nan)
test_y[:10]
test_y.shape
train_test_y = np.concatenate((y, test_y), axis=0)
train_test_y.shape
train = train.drop(['PassengerId', 'Survived'], axis=1)

test = test.drop(['PassengerId'], axis=1)
train_test = pd.concat([train, test], axis=0, ignore_index=True)
train_test['Survived'] = train_test_y
train_test.head()
train_test.tail()
train_test.info()
# Encoding Pclass in to dummy variables



from sklearn.preprocessing import OneHotEncoder



pclass = train_test['Pclass'].values

pclass = pclass.reshape(pclass.shape[0], 1)

print("pclass.shape: {}".format(pclass.shape))



encoder = OneHotEncoder(handle_unknown='ignore')

encoded = encoder.fit_transform(pclass)

encoded = encoded.toarray().astype('int8')



train_test = pd.concat([train_test, pd.DataFrame(encoded, columns=["pclass_" + str(pclass) for pclass in range(3)])], axis=1)
train_test.head()
train_test['Name'].isna().sum() # no nan value
import re



def leave_title(name):

    return re.findall('[A-Za-z]*\.', name)[0]



assert leave_title('Braund, Mr. Owen Harris') == 'Mr.'

assert leave_title('Heikkinen, Miss. Laina') == 'Miss.'

assert leave_title('Nasser, Mrs. Nicholas (Adele Achem)') == 'Mrs.'
train_test['titles'] = train_test['Name'].apply(leave_title)
train_test.head()
unique_title = train_test['titles'].unique()
unique_title
plt.figure(figsize=(20, 30))



sns.countplot(x='titles', hue='Survived', data=train_test[:891])                                       
def title_to_num(title):

    

    title = title.strip('.')

    

    if title in ['Mr']:

        return 'Man'

    elif title in ['Mrs', 'Miss', 'Master', 'Dona']:

        return 'Woman'

    elif title in ['Don', 'Rev', 'Capt', 'Jonkheer']:

        return 'Unlucky'

    elif title in ['Dr', 'Major', 'Col']:

        return 'Half-Lucky'

    elif title in ['Mme', 'Ms', 'Lady', 'Sir', 'Mlle', 'Countess']:

        return 'Lucky'

    
train_test['titles'] = train_test['titles'].apply(title_to_num)
train_test.head()
train_test['titles'].isna().sum()
plt.figure(figsize=(20, 20))



sns.countplot(x='titles', hue='Survived', data=train_test[:891])                                       
train_test['Sex'].isna().sum()
sex = train_test['Sex'].values

sex = sex.reshape(sex.shape[0], 1)

print("sex.shape: {}".format(sex.shape))



encoder = OneHotEncoder(drop='first')

encoded = encoder.fit_transform(sex)

encoded = encoded.toarray().astype('int8')



train_test = pd.concat([train_test, pd.DataFrame(encoded, columns=["sex"])], axis=1)
train_test.head()
train_test['SibSp'].isna().sum() # no nan values
sibsp = train_test['SibSp']
sibsp.describe()
plt.figure(figsize=(10, 10))



sns.countplot(x=train_test['SibSp'], hue='Survived', data=train_test)
train_test['Parch'].isna().sum() # no nan values
parch = train_test['Parch']
parch.describe()
plt.figure(figsize=(10, 10))



sns.countplot(x=train_test['Parch'], hue='Survived', data=train_test)
train_test_family = train_test.copy()
train_test_family['family'] = train_test_family['SibSp'] + train_test_family['Parch'] 
plt.figure(figsize=(10, 10))



sns.countplot(x='family', hue='Survived', data=train_test_family)
def family_size(family):

    

    if family == 0:

        return "Alone"



    elif family < 4:

        return "Small"

    

    elif 3 < family < 7:

        return "Medium"

    

    else:

        return "Large"

train_test['family'] = train_test['SibSp'] + train_test['Parch'] 
train_test['family_size'] = train_test['family'].apply(family_size)
train_test.head()
sns.countplot(x='family_size', hue='Survived', data=train_test)
train_test['Ticket'].isna().sum()
type(train_test['Ticket'][0])
def ticket_to_int(ticket):

    

    try:

        return int(ticket)

    except:

        return ticket.split()[0].strip('.')

        
train_test['tickets'] = train_test['Ticket'].apply(ticket_to_int)
train_test.head()
train_test['tickets'].describe()
len(train_test['tickets'].unique())
from collections import Counter



ticket_groups = Counter(train_test['tickets'])
ticket_groups
ticket_groups.get('A/5')
ticket_groups.get(113783)
def ticket_group(ticket):

    return ticket_groups.get(ticket)
train_test['ticket_groups'] = train_test['tickets'].apply(ticket_group)
plt.figure(figsize=(30, 10))



sns.countplot(x='ticket_groups', hue='Survived', data=train_test)
train_test.groupby('tickets')['Fare'].mean()
train_test.head()
train_test['Fare'].isna().sum()
train_test[train_test['Fare'].isna()]
train_test.groupby('Pclass')['Fare'].mean()
train_test['Fare'] = train_test['Fare'].fillna(train_test.groupby('Pclass')['Fare'].transform('mean'))
train_test['Fare'].isna().sum()
train_test['Fare'].describe()
train_test_fare = train_test.copy()



train_test_fare['Fare'] //= 30
plt.figure(figsize=(20, 10))



sns.countplot(x='Fare', hue='Survived', data=train_test_fare)
train_test['fare'] = train_test['Fare'] // 30
train_test.head()
train_test['Cabin'].shape
train_test['Cabin'].isna().sum() # many nan variables
train_test['Cabin'].unique()
def cabin_to_letter(cabin):

    

    if type(cabin) != float:

        return cabin[0]
train_test_cabin = train_test.copy()



train_test_cabin['cabin'] = train_test_cabin['Cabin'].apply(cabin_to_letter)
sns.countplot(x='cabin', hue='Survived', data=train_test_cabin)
train_test_cabin.groupby('cabin')['Fare'].mean()
train_test_cabin.groupby('cabin')['Fare'].max()
train_test_cabin.groupby('cabin')['Fare'].min()
train_test_cabin.groupby('cabin')['Pclass'].value_counts()
from math import isnan



def nan_cabins(row):

    

    try:

        str(row['Cabin'])



        if row['Pclass'] == 1:

            

            if row['Fare'] > 110:

                row['cabin'] = 'B'

            elif row['Fare'] > 100:

                row['cabin'] = 'C'

            elif row['Fare'] > 50:

                row['cabin'] = 'D'

            elif row['Fare'] > 45:

                row['cabin'] = 'E'

            elif row['Fare'] > 35:

                row['cabin'] = 'A'

            else:

                row['cabin'] = 'T'

            

                

        elif row['Pclass'] == 2:

 

            if row['Fare'] > 50:

                row['cabin'] = 'D'

            elif row['Fare'] > 45:

                row['cabin'] = 'E'

            else:

                row['cabin'] = 'F'

            

        else:

            

            if row['Fare'] > 45:

                row['cabin'] = 'E'

            elif row['Fare'] > 15:

                row['cabin'] = 'F'        

            else:

                row['cabin'] = 'G'

    

        return row

    

    except:

        return row
train_test = train_test_cabin.apply(nan_cabins, axis=1)
train_test['cabin'].isna().sum()
sns.countplot(x='cabin', hue='Survived', data=train_test)
train_test.head()
train_test['Age'].isna().sum()
train['Age'].describe()
train_test_age = train_test.copy()

train_test_age['Age'] = train_test_age['Age'] // 10
sns.countplot(x='Age', hue='Survived', data=train_test_age[:891])
sns.countplot(x='Age', hue='Sex', data=train_test_age[:891])
fare_age = train_test.groupby('fare')['Age'].mean()
train_test_age.head()
def nan_ages(row):

        

    try:

        int(row['Age'])

        return row

    

    except:

        row['Age'] = int(fare_age[row['fare']])

        return row
train_test = train_test_age.apply(nan_ages, axis=1)
train_test['Age'].isna().sum()
train_test.head()
train_test['Embarked'].isna().sum()
train_test[train_test['Embarked'].isna()]
train_test.groupby('Embarked')['Pclass'].value_counts()
train_test.groupby('Embarked')['Pclass'].median()
def nan_embarked(row):

    

    if row['Embarked'] in ['C', 'Q', 'S']:

        return row

    else:

      

        if row['Pclass'] == 1:

            row['Embarked'] = 'C'

        elif row['Pclass'] == 2:

            row['Embarked'] = 'S'

        elif row['Pclass'] == 3:

            row['Embarked'] = 'Q'

            

        return row

            
train_test = train_test.apply(nan_embarked, axis=1)
train_test['Embarked'].isna().sum()
train_test.columns
train_test_final = train_test.drop(['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Survived', 'Fare', 'family', 'tickets', 'Ticket', 'Cabin'], axis=1)
train_test_final.head()
train_test_final.tail()
train_test_final.isna().sum()
# we will convert 'Embarked', 'titles', 'family_size', 'cabin' to dummy variables
from sklearn.preprocessing import OneHotEncoder



train_test_encoder = train_test_final.copy()



encoder = OneHotEncoder(handle_unknown='ignore')



for column in ['Embarked', 'titles', 'family_size', 'cabin']:

    

    column_array = train_test[column].values

    encoded = encoder.fit_transform(column_array.reshape(column_array.shape[0], 1))

    encoded = encoded.toarray().astype('int8')

    

    train_test_encoder = pd.concat([train_test_encoder, pd.DataFrame(encoded, columns=[column + str(i) for i in range(encoded.shape[1])])], axis=1)
train_test_encoder.head()
train_test_encoder.columns
train_test_encoder = train_test_encoder.drop(['Embarked', 'titles', 'cabin', 'family_size'], axis=1)
train_test_encoder.head()
train_test_encoder.info()
train_test_encoder.isna().sum()
train_processed = train_test_encoder[:891]

test_processed = train_test_encoder[891:]
train_processed.shape
test_processed.shape
import xgboost as xgb

from sklearn.model_selection import train_test_split
train_processed.drop(['Age', 'ticket_groups'], axis=1, inplace=True)

test_processed.drop(['Age', 'ticket_groups'], axis=1, inplace=True)
X = train_processed.values

X_submission = test_processed.values
print("X.shape: {}".format(X.shape))

print("y.shape: {}".format(y.shape))

print()

print("X_submission.shape: {}".format(X_submission.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12)
print("X_train.shape: {}".format(X_train.shape))

print("y_train.shape: {}".format(y_train.shape))

print()

print("X_test.shape: {}".format(X_train.shape))

print("y_test.shape: {}".format(y_test.shape))
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
xgb_model.score(X_test, y_test)
xgb_pred = xgb_model.predict(X_submission)
pred = pd.DataFrame(xgb_pred)

pred = pred.reset_index()



pred['index'] = pred['index'] + 892

pred = pred.rename(columns={'index': 'PassengerId', 0: 'Survived'})
pred.head()
pred.to_csv('submission.csv', index=False)