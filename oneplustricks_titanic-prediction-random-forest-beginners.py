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
x=[]

# changing the cabin and cabinas columns 
for i in range(len(train['Cabin'])):
    y = ord(str(train['Cabin'][i])[0])
   
    if y==110:
        train['cabinas'] = 99999
    
    else:
        train['cabinas'] = y 
for i in range(len(test['Cabin'])):
    y = ord(str(test['Cabin'][i])[0])
   
    if y==110:
        test['cabinas'] = 99999
    
    else:
        test['cabinas'] = y        
train_test_data = [train, test] # combining train and test dataset

# changing sex column to integers
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
# droppig irrelevant features for basic model    
features_drop = ['PassengerId','Name', 'Fare', 'Ticket','Cabin','Embarked']

train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)

train.fillna(0,inplace=True)
test.fillna(0,inplace=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

train = train.drop('Survived',axis=1)
test = test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=100,n_jobs=5)
clf.fit(X, y)
scores = cross_val_score(clf, X, y, cv=5)
print('scores_mean = ',scores.mean())
y_pred_random_forest = clf.predict(test)
test_copy = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.DataFrame({
        'PassengerId': test_copy["PassengerId"],
        'Survived': y_pred_random_forest
    })
submission.reset_index(drop=True, inplace=True) 
submission.to_csv('submission.csv',index=False)
