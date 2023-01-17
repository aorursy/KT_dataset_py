# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

%matplotlib inline 

from matplotlib import pyplot as plt

from matplotlib import style

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data.head()
train_data.describe()
train_data.dtypes
# Cleaning the data 

# first step: Separate title from name 

# ref: https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/



def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if str.find(big_string, substring) != -1:

            return substring

    print (big_string)

    return np.nan
title = []



for name in train_data.Name:

    title.append(name.split(',')[1].split('.')[0].strip())

    

    
train_data['Title']=train_data['Name'].map(lambda x: substrings_in_string(x, title))

train_data.Title.head(5)
#replacing all titles with mr, mrs, miss, master

# ref: https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

train_data.Title=train_data.apply(replace_titles, axis=1)
train_data.head(5)
# step 2 : Age

#sum of all null data

train_data.isnull().sum().sort_values(ascending=False)
train_data.Age.fillna(train_data.Age.mean(), inplace = True)



print("max:" , max(train_data.Age))

print("min:" , min(train_data.Age))
#Cabin letters, ref:https://www.kaggle.com/zlatankr/titanic-random-forest-82-78



train_data['cabin_letter'] = train_data['Cabin'].apply(lambda x: str(x)[0])

train_data['cabin_letter'].value_counts()

#Embarked



sns.countplot(train_data['Embarked'], hue=train_data['Survived'])
#ref: https://www.kaggle.com/saadmuhammad17/a-beginners-guide-to-data-science-top-3

#visualization 

sns.countplot(train_data['Pclass'], hue = train_data['Survived'])
sns.countplot(train_data['Sex'],hue = train_data['Survived'])
train_data['fx_age'] = sorted(train_data.Age.apply(lambda x: '0-20' if x < 20 

                                            else ('20-40' if (x >= 20 and x < 40) 

                                            else ('40-60' if (x >= 40 and x <=60) else '> 60'))))

sns.countplot(train_data['fx_age'],hue = train_data['Survived'])
df = train_data.drop(['PassengerId','Name','Ticket','Fare','Cabin','Age','Pclass'], axis = 1)

df.head()
# preparation to use random forest model



x = df.drop('Survived',axis = 1)

y  = df['Survived']
x = pd.get_dummies(x)
len(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
rdmf = RandomForestClassifier(n_estimators=20, criterion='entropy')

rdmf.fit(x_train, y_train)
accuracies = []



rdmf_s = rdmf.score(x_test, y_test)

rdmf_str = rdmf.score(x_train, y_train)

accuracies.append(rdmf_s)

print(rdmf_s)

print(rdmf_str)
train_data['label'] = 'treino'

test_data['label'] = 'teste'
concat_df = pd.concat([train_data , test_data])

concat_df.head()
concat_df['Title']=concat_df['Name'].map(lambda x: substrings_in_string(x, title))

concat_df.Title=concat_df.apply(replace_titles, axis=1)

concat_df.Age.fillna(concat_df.Age.mean(), inplace = True)

concat_df['cabin_letter'] = concat_df['Cabin'].apply(lambda x: str(x)[0])

concat_df['fx_age'] = sorted(concat_df.Age.apply(lambda x: '0-20' if x < 20 

                                            else ('20-40' if (x >= 20 and x < 40) 

                                            else ('40-60' if (x >= 40 and x <=60) else '> 60'))))

concat_df = concat_df.drop(['PassengerId','Name','Ticket','Fare','Cabin','Age','Pclass'], axis = 1)
test_data_titanic = pd.get_dummies(concat_df)

test_data_titanic.head()
teste_df = test_data_titanic[test_data_titanic['label_teste'] == 1]

teste_df
teste_df = teste_df.drop(['label_teste','label_treino','Survived'], axis=1)
y_pred = rdmf.predict(teste_df)

len(y_pred)
titanic_final = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived':y_pred})
titanic_final.to_csv('Titanic_final.csv', index=False)