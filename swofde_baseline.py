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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

def process_cabin(x):
    x = str(x)
    if 'a' in x:
        return 11.0
    elif 'b' in x:
        return 12.0
    elif 'c' in x:
        return 13.0
    elif 'd' in x:
        return 14.0
    elif 'e' in x:
        return 15.0
    else: 
        return len(str(x))
    
def master(x):
    if 'Mast' in x:
        return True
    else: 
        return False
    
    
def transform(db):
    db['Age'].fillna(34.0, inplace=True) 
    db['Cabin'].fillna('', inplace=True)     
    db['Age_bin'] = pd.cut(db['Age'], [0., 1.0, 5.0, 18.0, 33.0, 45.0, 65.0, 120.0], 
                       labels=False)    
    db['Master'] = db['Name'].apply(master)
    db = db[['Sex', 'Ticket', 'Cabin', 'Pclass', 'Age_bin', 'SibSp',
       'Parch', 'Fare', 'Embarked','Master', 'Age']]
    db['Sex'] = db['Sex'].replace({'male':1,'female':2})
    db['Ticket'] = db['Ticket'].apply(len)

    db['Cabin'] = db['Cabin'].apply(process_cabin)
    db['Embarked'] = db['Embarked'].replace({'S':4,'Q':2,'C':3})

    return db 
    

X, y = transform(df), df['Survived']

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import statistics


N, results = 30, []

for i in range(N):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBClassifier(n_estimators=2000,max_depth=4,reg_lambda=0.1)
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_val)
    results.append(accuracy_score(y_val,y_prediction))

print(sum(results)/N, statistics.stdev(results))

test = pd.read_csv('/kaggle/input/titanic/test.csv')
y_prediction = model.predict(transform(test))
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission['Survived'] = y_prediction
submission.to_csv('my_results.csv',index=False)
