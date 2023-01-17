# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
train_data = train_data.drop(columns=['Name','Cabin', 'PassengerId'])
train_data.fillna(0,inplace=True)
test_data.fillna(0,inplace=True)
train_data.head()
train_data.describe()
def bar_chart(feature):
    survived = train_data[train_data['Survived'] == 1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(15,7))
    
bar_chart('Sex')
bar_chart('Pclass')

bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = train_data[features]
y = train_data['Survived']
x.head()
x.isnull().sum()
# Now let's enocde categorical values 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

x['Sex'] = LE.fit_transform(x['Sex'])
x['Embarked'] = LE.fit_transform(np.array(x['Embarked']).astype(str))
print(x)
y.isnull().sum()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state =0)
x_train
##Now we fit our model
from xgboost import XGBClassifier
classifier = XGBClassifier(colsample_bylevel= 0.9,
                    colsample_bytree = 0.8, 
                    gamma=0.99,
                    max_depth= 5,
                    min_child_weight= 1,
                    n_estimators= 10,
                    nthread= 4,
                    random_state= 2,
                    silent= True)
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)
##Now take the test data for prediction
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_x = test_data[features]
test_x
test_x.isnull().sum()
##Let's fill values
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].median())
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].median())
test_x.isnull().sum()

##Let's enocde categorical values
test_x['Sex'] = LE.fit_transform(test_x['Sex'])
test_x['Embarked'] = LE.fit_transform(test_x['Embarked'])
test_x.head()

prediction = classifier.predict(test_x)
prediction
output = pd.DataFrame({'PassengerId': test_data.PassengerId,'Survived': prediction})
output.to_csv('submission.csv', index=False)
output.head()
