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
train_data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

train_data
train_data.drop(columns=['Name', 'Ticket','Cabin'], inplace=True)

train_data
age_mean = train_data.Age.mean()

age_mean
train_data.Age.fillna(int(age_mean), inplace=True)
train_data.Embarked.unique()
train_data.Embarked.value_counts()
from sklearn.preprocessing import LabelEncoder

sex_le =  LabelEncoder() 

train_data['Sex'] = sex_le.fit_transform(train_data['Sex'].astype(str))



embarked_le =  LabelEncoder() 

train_data['Embarked'] = embarked_le.fit_transform(train_data['Embarked'].astype(str))
train_data
y = train_data['Survived']

X = train_data.drop(columns= ['Survived'])

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
para = list(range(100, 1001, 100))

print(para)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

results = {}

for n in para:

    print('para=', n)

    model = RandomForestClassifier(n_estimators=n , random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accu = accuracy_score(y_true=y_test, y_pred=preds)

    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')

    print(classification_report(y_true=y_test, y_pred=preds))

    print('--------------------------')

    results[n] = f1
import matplotlib.pylab as plt

lists = sorted(results.items()) 

p, a = zip(*lists) 

plt.plot(p, a)

plt.show()
best_para = max(results, key=results.get)

print('best para', best_para)

print('value', results[best_para])

test_data = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

test_data
test_data.drop(columns=['Name', 'Ticket','Cabin'], inplace=True)

test_data
from sklearn.preprocessing import LabelEncoder

sex_le =  LabelEncoder() 

test_data['Sex'] = sex_le.fit_transform(test_data['Sex'].astype(str))

test_data['Embarked'] = embarked_le.transform(test_data['Embarked'].astype(str))
final_model = RandomForestClassifier(n_estimators=best_para)

final_model.fit(X, y)
test_data.Age.fillna(X.Age.mean(), inplace=True)

test_data.Fare.fillna(X.Fare.mean(), inplace=True)



test_data.isna().sum()
preds = final_model.predict(test_data)

print(preds.shape)

print(test_data.shape)
preds[:5]
test_out = pd.DataFrame({

    'PassengerId': test_data.index, 

    'Survived': preds

})

test_out.to_csv('submission.csv', index=False)

print('Done')