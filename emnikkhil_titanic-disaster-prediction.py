# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col = 'PassengerId')

test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col = 'PassengerId')
train.head()
train.describe()
test.describe()
train.shape, test.shape
survived = train['Survived'].copy()



train = train.drop('Survived', axis = 1)
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
df = pd.concat([test, train])

training_index = train.index

testing_index = test.index



print(train.equals(df.loc[training_index, :]))

print(test.equals(df.loc[testing_index, :]))
del train

del test
df.head()
df.info()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



df['Name_length'] = df['Name'].apply(len)



df['IsAlone'] = 0



df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1



df['Title'] = 0



df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') 



df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',

                         'Rev','Capt','Sir','Don'], ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'], inplace = True)
df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()



df = df.drop('Name', axis=1)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])



df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)



df= df.drop(['Ticket', 'Cabin'], axis=1)
df['Title'] = pd.Categorical(df['Title'])

df['Title'] = df['Title'].cat.codes



df.head()
train = df.loc[training_index, :]

train['Survived'] = survived



train.head()
train.to_csv('final_training_set.csv', header = True, index = True)

df.loc[testing_index, :].to_csv('final_testing_set.csv', header = True, index = True)
final_train = pd.read_csv('./final_training_set.csv')

final_test = pd.read_csv('./final_testing_set.csv')
final_train.head()
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Name_length', 'IsAlone', 'Title']



selected_train = train[features]



labels = final_train.Survived
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(selected_train, labels, train_size=0.8, test_size=0.2, random_state=2)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
print(classifier.score(X_train, y_train))
print(classifier.score(X_test, y_test))
selected_test = final_test[features]

model = XGBClassifier()

model.fit(selected_train, labels)

submission = pd.DataFrame({"PassengerId": final_test.PassengerId, "Survived": model.predict(selected_test)})

submission.to_csv('submission.csv', index=False)