# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print(list(train_data))



train_data_manip = train_data.copy(deep=True)
sex_mapping={'male': 0, 'female': 1}

train_data_manip['Sex']=train_data_manip['Sex'].map(sex_mapping)

print(train_data_manip['Sex'])
embarked_mapping={'S': 0, 'C': 1, 'Q': 2}

train_data_manip['Embarked']=train_data_manip['Embarked'].fillna('S')

train_data_manip['Embarked']=train_data_manip['Embarked'].map(embarked_mapping)

print(train_data_manip['Embarked'])
cabin_mapping={'A': 0, 'B': 1, 'C': 2, 'D':3,'E':4,'F':5,'G':6,'T':7}



train_data_manip['Cabin']=train_data_manip['Cabin'].fillna('T')

train_data_manip['Cabin']=train_data_manip['Cabin'].str.slice(0,1).map(cabin_mapping)

print(train_data_manip['Cabin'])
train_data_manip.loc[train_data_manip.Age<11, 'Age'] = 0

train_data_manip.loc[(train_data_manip.Age>=11) & (train_data_manip.Age<25), 'Age'] = 1

train_data_manip.loc[(train_data_manip.Age>=25) & (train_data_manip.Age<45), 'Age'] = 2

train_data_manip.loc[(train_data_manip.Age>=45) & (train_data_manip.Age<60), 'Age'] = 3

train_data_manip.loc[(train_data_manip.Age>=60), 'Age'] = 4

print(train_data_manip['Age'])
train_data_m=train_data_manip.dropna()



X = train_data_m[["Pclass","Sex","Age","Fare","Cabin","Embarked"]].values

y = train_data_m[["Survived"]].values



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
model=RandomForestClassifier()

model.fit(X,y)

y_pred = model.predict(X_test)

acc_random_forest = round(model.score(X, y)*100, 2)

print(acc_random_forest)
test_data_manip = test_data.copy(deep=True)

test_data_manip['Sex']=test_data_manip['Sex'].map(sex_mapping)

test_data_manip['Embarked']=test_data_manip['Embarked'].fillna('S')

test_data_manip['Embarked']=test_data_manip['Embarked'].map(embarked_mapping)

test_data_manip['Cabin']=test_data_manip['Cabin'].fillna('T')

test_data_manip['Cabin']=test_data_manip['Cabin'].str.slice(0,1).map(cabin_mapping)

mean_age=test_data_manip['Age'].mean()

test_data_manip['Age'] = test_data_manip['Age'].fillna(mean_age)

test_data_manip.loc[test_data_manip.Age<11, 'Age'] = 0

test_data_manip.loc[(test_data_manip.Age>=11) & (test_data_manip.Age<25), 'Age'] = 1

test_data_manip.loc[(test_data_manip.Age>=25) & (test_data_manip.Age<45), 'Age'] = 2

test_data_manip.loc[(test_data_manip.Age>=45) & (test_data_manip.Age<60), 'Age'] = 3

test_data_manip.loc[(test_data_manip.Age>=60), 'Age'] = 4

mean_fare=test_data_manip['Fare'].mean()

test_data_manip['Fare'] = test_data_manip['Fare'].fillna(mean_fare)



Sex_nulos=test_data_manip['Sex'].isnull().values.any()

print('Sex '+str(Sex_nulos))

Pclass_nulos=test_data_manip['Pclass'].isnull().values.any()

print('Pclass '+str(Pclass_nulos))

Embarked_nulos=test_data_manip['Embarked'].isnull().values.any()

print('Embarked '+str(Embarked_nulos))

Cabin_nulos=test_data_manip['Cabin'].isnull().values.any()

print('Cabin '+str(Cabin_nulos))

Age_nulos=test_data_manip['Age'].isnull().values.any()

print('Age '+str(Age_nulos))

Fare_nulos=test_data_manip['Fare'].isnull().values.any()

print('Fare '+str(Fare_nulos))





print(test_data_manip)

print(test_data_manip.shape)
X_test_sub = test_data_manip[["Pclass","Sex","Age","Fare","Cabin","Embarked"]].values



y_pred = model.predict(X_test_sub)



output = pd.DataFrame({'PassengerId': test_data_manip.PassengerId, 'Survived': y_pred})

print(output)



output.to_csv('my_submission_v11.csv', index=False)

print("Your submission was successfully saved!")