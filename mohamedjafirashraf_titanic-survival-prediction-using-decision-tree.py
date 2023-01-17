# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
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
sns.countplot(train_data['Survived'],hue=train_data['Sex'])
Dead, lives = train_data.Survived.value_counts()
male, female = train_data.Sex.value_counts()
print("Percentage of Male :", round(male/(male+female)*100),'%' )
print("Percentage of Female :", round(female/(male+female)*100 ),'%')
sns.countplot(train_data['Survived'],hue=train_data['Pclass'])
sns.countplot(train_data['Survived'],hue=train_data['Parch'])
train_data.replace('male',1, inplace=True)
train_data.replace('female',0,inplace=True)
not_zero = ['Age']

for column in not_zero:
    mean = int(train_data[column].mean(skipna=True))
    train_data[column] = train_data[column].replace(np.NaN,mean)
X = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y=train_data.Survived
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)
pred=model.predict([[1,0,38.0,1,0,71.2833]])
if pred ==1:
    print("The person is Survived")
else:
    print("The person is not Survived")
pred
pred=model.predict([[1,1,55.0,1,0,80.233]])
if pred ==1:
    print("The person is Survived")
else:
    print("The person is not Survived")
pred
survivors = pd.DataFrame(y_pred, columns = ['Survived'])
len(survivors)
survivors.insert(0, 'PassengerId', test_data['PassengerId'], True)
survivors
survivors.to_csv('Submission.csv', index = False)