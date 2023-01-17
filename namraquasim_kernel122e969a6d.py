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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
data=pd.read_csv("../input/titanic/gender_submission.csv")
data.head()
data.tail()
test_data=pd.read_csv("../input/titanic/test.csv")
train_data=pd.read_csv("../input/titanic/train.csv")

test_data.info()
train_data.describe()
sns.countplot(x='PassengerId',hue='Survived',data=train_data)
sns.countplot(x='Pclass',hue='Survived',data=train_data)
sns.countplot(x='Sex',hue='Survived',data=train_data)
sns.countplot(x='Age',hue='Survived',data=train_data)
sns.countplot(x='SibSp',hue='Survived',data=train_data)
sns.countplot(x='Parch',hue='Survived',data=train_data)
sns.countplot(x='Ticket',hue='Survived',data=train_data)
sns.countplot(x='Fare',hue='Survived',data=train_data)
sns.countplot(x='Cabin',hue='Survived',data=train_data)
sns.countplot(x='Embarked',hue='Survived',data=train_data)
x=train_data[['Pclass','Sex','Fare','Embarked','SibSp','Age','Parch']]
y=train_data['Survived']
x.info()
x
x['Embarked'].fillna('S',inplace=True)
x.info()
x['Age'].fillna(train_data['Age'].mean(),inplace=True)
x.info()
y
x1=test_data[['Pclass','Sex','Fare','Embarked','SibSp','Age','Parch']]
x1.info()
x1['Age'].fillna(test_data['Age'].mean(),inplace=True)
x1.info()
x1['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
x1.info()
x['Sex'].replace('male',0,inplace=True)
x['Sex'].replace('female',1,inplace=True)
x1['Sex'].replace('male',0,inplace=True)
x1['Sex'].replace('female',1,inplace=True)
x
x['Embarked'].replace('S',0,inplace=True)
x['Embarked'].replace('C',1,inplace=True)
x['Embarked'].replace('Q',2,inplace=True)

x1['Embarked'].replace('S',0,inplace=True)
x1['Embarked'].replace('C',1,inplace=True)
x1['Embarked'].replace('Q',2,inplace=True)
x1
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(criterion = "gini", 
                     min_samples_leaf = 1, 
                     min_samples_split = 10,   
                     n_estimators=100, 
                     max_features='auto', 
                     oob_score=True, 
                     random_state=1, 
                     n_jobs=-1)

model.fit(x, y)

prediction = model.predict(x1)

model_accuracy = model.score(x, y)
print(model_accuracy)

parametros = pd.DataFrame({'feature':x.columns,'Parametros':np.round(model.feature_importances_,3)})
parametros = parametros.sort_values('Parametros',ascending=False).set_index('feature')
parametros.plot.bar()
submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived' : prediction})

submission.to_csv('Submission.csv', index = False)