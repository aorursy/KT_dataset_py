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
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train
df_train = df_train.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'])
df_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'])
df_train = df_train.fillna(method='ffill')
survived_question = df_train[['Survived', 'Sex', 'Age']]
survived_people = survived_question[(survived_question['Survived'] == 1)]
survived_people
survived_men = survived_people[(survived_people['Sex'] == 'male')]
survived_women = survived_people[(survived_people['Sex'] == 'female')]
import matplotlib.pyplot as plt
%matplotlib notebook
plt.title('Distribution of male and female survivors in order of age')
plt.hist(survived_men['Age'], bins=20, alpha=0.5, label='Male Surviors')
plt.hist(survived_women['Age'], bins=20, alpha=0.3, label='Female Surviors')
plt.legend(loc='upper right')
print(f'Number of survived males: {len(survived_men)}')
print(f'Number of survived females: {len(survived_women)}')
from sklearn.model_selection import train_test_split
np.random.seed(2)
df_test = df_train.fillna(method='ffill')
X = df_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df_test[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
