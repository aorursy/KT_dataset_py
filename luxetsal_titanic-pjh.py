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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df = train_data

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df = test_data

test_data.head()
combine = [train_df, test_df]

train_df.head(5)
train_df.info()
# Age column

# NaN -> 0으로 채우기



#new_train_data = train_data.copy()



mean = 0



train_df["Age"].fillna(value=mean, inplace=True)

print(train_df["Age"])



test_df["Age"].fillna(value=mean, inplace=True)

print(test_df["Age"])
# Fare column

# NaN -> 평균으로 채우기

# Class 1 평균 : 84.15

# Class 2 평균 : 20.66

# Class 3 평균 : 13.68



# test data에 1개 missing



test_df[test_df["Fare"].isnull()]

test_df["Fare"][152] = 13.68
# obtain Title from name (Mr, Mrs, Miss etc)

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)





for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'],'Royalty')

    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')

    dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')

    dataset.loc[(dataset.Sex == 'male')   & (dataset.Title == 'Dr'),'Title'] = 'Mr'

    dataset.loc[(dataset.Sex == 'female') & (dataset.Title == 'Dr'),'Title'] = 'Mrs'



#: count survived rate for different titles

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Title', 'Age']].groupby(['Title'], as_index=False).mean().sort_values(by='Age', ascending=False)
# Fill Missing Age with Title Mean

# Title 평균으로 빈 Age 채우기 



Officer_mean = 49.272727

Royalty_mean = 41.6

Mrs_mean = 35.909091

Mr_mean = 32.470223

Miss_mean = 21.845638

Master_mean = 4.574167



for dataset in combine:

    dataset.loc[(dataset.Title == 'Officer')   & (dataset.Age == 0.0),'Age'] = Officer_mean

    dataset.loc[(dataset.Title == 'Royalty')   & (dataset.Age == 0.0),'Age'] = Royalty_mean

    dataset.loc[(dataset.Title == 'Mrs')   & (dataset.Age == 0.0),'Age'] = Mrs_mean

    dataset.loc[(dataset.Title == 'Mr')   & (dataset.Age == 0.0),'Age'] = Mr_mean

    dataset.loc[(dataset.Title == 'Miss')   & (dataset.Age == 0.0),'Age'] = Miss_mean

    dataset.loc[(dataset.Title == 'Master')   & (dataset.Age == 0.0),'Age'] = Master_mean
# Covert 'Title' to numbers (Mr->1, Miss->2 ...)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty":5, "Officer": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



# Remove 'Name' and 'PassengerId' in training data, and 'Name' in testing data

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]



# if age < 16, set 'Sex' to Child

for dataset in combine:

    dataset.loc[(dataset.Age < 16) & (dataset.Sex == 'female'),'Sex'] = 'Girl'

    dataset.loc[(dataset.Age < 16) & (dataset.Sex == 'male'),'Sex'] = 'Boy'

    

# Covert 'Sex' to numbers (female:1, male:2)

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0, 'Boy': 2, 'Girl': 3} ).astype(int)

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Create family size from 'sibsq + parch + 1'

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#create another feature called IsAlone

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1

    dataset.loc[(dataset['FamilySize'] > 4),  'IsAlone'] = 2



train_df[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean()



combine = [train_df, test_df]

train_df.head()
train_data = train_df

test_data = test_df
train_data.head(20)
from sklearn.svm import SVC # support vector classification



y = train_data["Survived"]



features = ["Pclass", "Sex", "Age", "Fare", "Title", "Embarked", "IsAlone"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = SVC(kernel='rbf',gamma='auto')

model.fit(X, y)

predictions = model.predict(X_test)
model.score(X, y)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('Titanic_submission.csv', index=False)

print("Your submission was successfully saved!")