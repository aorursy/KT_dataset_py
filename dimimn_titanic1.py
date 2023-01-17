# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import xgboost as xgb
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('/kaggle/input/titanic/train.csv')


df['Sex']=df['Sex'].replace({'male':1,'female':0})
df['Age']=df['Age'].fillna(30)
df
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
collumns = ['PassengerId', 'Pclass', 'Sex', 'Age']
X = df[collumns]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = xgb.XGBClassifier(max_depyh=3, n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=['error'])
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
y_pred = pd.DataFrame(y_test,columns=['Survived'])
y_pred.to_csv('submission.csv', index=False)
df
