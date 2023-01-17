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
titanic_train_filepath= '../input/titanic/train.csv'

titanic_test_filepath='../input/titanic/test.csv'

df_train=pd.read_csv(titanic_train_filepath, index_col='PassengerId')
df_train
df_train.columns
df_train.isna().sum()
df_train.Embarked.value_counts()
df_train.Embarked.fillna('S', inplace=True)

df_train.Embarked.value_counts()
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)

df_train.Age.isna().sum()
df_train.isna().sum()
df_train.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

df_train.shape
from sklearn.preprocessing import LabelEncoder



lebeled_Sex = LabelEncoder()

labeled_Embarked = LabelEncoder()



df_train['Sex'] = lebeled_Sex.fit_transform(df_train['Sex'])

df_train['Embarked'] = labeled_Embarked.fit_transform(df_train['Embarked'])
y = df_train.Survived

X = df_train.drop(['Survived'], axis=1)



from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)
para = list(range(100, 1001, 100))

print(para)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

results = {}

for n in para:

    print('para=', n)

    model = RandomForestClassifier(n_estimators=n)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    accu = accuracy_score(y_true=y_valid, y_pred=preds)

    f1 = f1_score(y_true=y_valid, y_pred=preds, average='micro')

    print(classification_report(y_true=y_valid, y_pred=preds))

    print('--------------------------')

    results[n] = f1
import matplotlib.pylab as plt



# sorted by key, return a list of tuples

lists = sorted(results.items()) 

p, a = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(p, a)

plt.show()
best_para = max(results, key=results.get)

print('best para', best_para)

print('value', results[best_para])
df_test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

df_test
df_test.isna().sum()
df_test['Age'].fillna(df_train['Age'].mean(), inplace=True)

df_test.Age.isna().sum()
df_test['Fare'].fillna(df_train['Fare'].mean(), inplace=True)

df_test.Fare.isna().sum()
df_test.isna().sum()
df_test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

print(df_test.shape)
from sklearn.preprocessing import LabelEncoder



df_test['Sex'] = lebeled_Sex.transform(df_test['Sex'])

df_test['Embarked'] = labeled_Embarked.transform(df_test['Embarked'])
final_model = RandomForestClassifier(n_estimators=best_para)

final_model.fit(X, y)
preds = final_model.predict(df_test)

df_sub = pd.DataFrame(data={

    'PassengerId': df_test.index,

    'Survived': preds

})
df_sub.Survived.value_counts()
df_sub.to_csv('submission.csv', index=False)