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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
db=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df.isnull().sum()
emb = df.Embarked.dropna().mode()[0]
emb
df['Age'].median()
df['Cabin'] = df['Cabin'].fillna('A')
df['Embarked'] = df['Embarked'].fillna('S')
df['Age'] = df['Age'].fillna(28)
df["Name"] = df["Name"].apply(len)
df['Sex'] = df['Sex'].replace({'male':0, 'female':1}) 
df["Ticket"] = df["Ticket"].apply(len)
df["Cabin"] = df["Cabin"].apply(str).apply(len)
df['Embarked'] = df['Embarked'].replace({'S':0, 'C':1, 'Q':2})
df
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

y = df['Survived']
x = df[df.columns[1:]]
best = 0 
average = 0
bestdepth=0
bestestim=0
bestlearn=0
total_for_average = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
for i in range(1,10):
    for j in range(1,20):
        model = xgb.XGBClassifier(max_depth=i*1, n_estimators=j*10, learning_rate=0.01*j)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        #print(accuracy_score(y_test, y_pred))
        total_for_average += 1
        average += accuracy_score(y_test, y_pred)
        if (accuracy_score(y_test, y_pred) > best): 
            best = accuracy_score(y_test, y_pred)
            bestdepth=i*1
            bestestim=j*10
            bestlearn=0.01*j
print("\nThe Best is", best)
print("\nThe Best things are", bestdepth, bestestim, bestlearn)
print("The Average is", average/total_for_average)
y = df['Survived']
x = df[df.columns[1:]]
best = 0 
average = 0
total_for_average = 0

models=[]
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    model = xgb.XGBClassifier(max_depth=3, n_estimators=180, learning_rate=0.18)
    model.fit(x_train, y_train)
    models.append(model)
    y_pred = model.predict(x_test)


print(accuracy_score(y_test, y_pred))
total_for_average += 1
average += accuracy_score(y_test, y_pred)