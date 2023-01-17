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
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
gender_submission.head()
train
women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier

y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('hansp.csv', index=False)
print("Your submission was successfully saved!")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
plt.subplots(figsize=(10,10))
sns.countplot('Sex',hue='Survived',data=train)
plt.show()
sns.distplot(train['Age'].values, bins=range(0,16), kde=False)
sns.distplot(train['Age'].values, bins=range(16, 32), kde=False)
sns.distplot(train['Age'].values, bins=range(32, 48), kde=False)
sns.distplot(train['Age'].values, bins=range(48,64), kde=False)
sns.distplot(train['Age'].values, bins=range(64,82), kde=False, axlabel='Age')
train['Age_Category'] = pd.cut(train['Age'],bins=[0,16,32,48,64,81])
sns.countplot('Age_Category',hue='Survived',data=train)
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4
    
train.head()
train['Family'] = train['SibSp'] + train['Parch'] + 1
train['Alone'] = 0
train.loc[train['Family'] == 1, 'Alone'] = 1
train.head()
train['Survived'].replace("Yes", 1,inplace=True)
train['Survived'].replace("No", 0, inplace=True)
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]
sns.barplot(x='Pclass', y='Survived', data=train);
train['Sex'].replace("male", 0, inplace=True)
train['Sex'].replace("female", 1, inplace=True)
train.head()
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
train.head()
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
train = train.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'PassengerId', 'Age_Category', 'FareBand'], axis=1)
train['Age'] = train['Age'].fillna(2)
train['Age'] = train['Age'].astype(int)
train.head()
training, testing = train_test_split(train, test_size=0.2, random_state=0)

cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family', 'Alone']
tcols = np.append(['Survived'],cols)
df = training.loc[:,tcols].dropna()

X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:,tcols].dropna()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['Survived']])
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(X, y)
y_red_random_forest = RFC.predict(X_test)
random_forest = round(clf.score(X, y)*100, 2)
random_forest

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
