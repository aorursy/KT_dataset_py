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

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.info()
train_data = train_data.drop(['Name','PassengerId', 'Cabin', 'Ticket'], axis=1)
data = [train_data, test_data]



for dataset in data:

    mean = train_data['Age'].mean()

    new_age = np.random.randint(mean)

    

    age_slice=dataset['Age'].copy()

    age_slice[np.isnan(age_slice)] = new_age

    dataset['Age'] = age_slice

    dataset['Age'] = train_data['Age'].astype(int)

    

train_data['Age'].isnull().sum()
train_data['Embarked'].describe()
common_value = 'S'

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

    

train_data['Embarked'].isnull().sum()
train_data.info()
data = [train_data, test_data]



for dataset in data:

    dataset['Fare'] = dataset ['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_data.info()
gender = {"male": 0, "female": 1}

data = [train_data, test_data]



for dataset in data:

    dataset['Sex']= dataset['Sex'].map(gender)

    

train_data.info()
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)

    

train_data.info()
data = [train_data, test_data]



for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    

    dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 12 ) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 30), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 60), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4



train_data['Age'].value_counts()
data = [train_data, test_data]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_data['Fare'].value_counts()
data = [train_data, test_data]

for dataset in data:

    dataset['Age_Class'] = dataset['Age']* dataset ['Pclass']

    

train_data.head(10)
X = train_data.drop('Survived', axis=1)

y = train_data['Survived']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.53, random_state=123)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logmodel=LogisticRegression()

logmodel.fit(X_train, y_train)

predictions= logmodel.predict(X_test)

logmodel.score(X_test, y_test)

pd.crosstab(y_test, predictions)



print ("My score is: {}".format(round(accuracy_score(predictions, y_test),4)))
predictions = logmodel.predict(X_train)

predictions



submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions})

submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)