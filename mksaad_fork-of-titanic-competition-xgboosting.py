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
df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

df
for col in df.columns:

    print(col, len(df[col].unique()))
df.drop(columns=['Name', 'Ticket'], inplace=True)
df
target_col = 'Survived'

print(df[target_col].unique())

print('-----------------------')

print(df[target_col].value_counts())

print('-----------------------')

print(df.dtypes)

print('-----------------------')

print(df.isna().sum())
df.drop(columns=['Cabin'], inplace=True)

df
age_mean = df.Age.mean()

age_mean
df.Age.fillna(int(age_mean), inplace=True)
print(df.isna().sum())
df.Embarked.mode()
df.Embarked.fillna(df.Embarked.mode(), inplace=True)
df
df.Embarked.value_counts()
from sklearn.preprocessing import LabelEncoder



# for col in df.columns:

#     if df[col].dtype == 'object':

#         print('apply label encoding on', col)

#         df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# df



sex_le =  LabelEncoder() 

df['Sex'] = sex_le.fit_transform(df['Sex'].astype(str))



embarked_le =  LabelEncoder() 

df['Embarked'] = embarked_le.fit_transform(df['Embarked'].astype(str))
y = df[target_col]

X = df.drop(columns=[target_col])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
para = list(range(100, 1001, 100))

print(para)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBRegressor, XGBClassifier



results = {}

for n in para:

    print('para=', n)

    # model = RandomForestClassifier(n_estimators=n)

    model = XGBClassifier(n_estimators=n, learning_rate=0.05, n_jobs=4)

    model.fit(X_train, y_train, early_stopping_rounds=20,

              eval_set=[(X_test, y_test)], 

             verbose=False)

    preds = model.predict(X_test)

    accu = accuracy_score(y_true=y_test, y_pred=preds)

    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')

    print(classification_report(y_true=y_test, y_pred=preds))

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
test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

test_df
test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

test_df
test_df['Sex'] = sex_le.transform(test_df['Sex'].astype(str))

test_df['Embarked'] = embarked_le.transform(test_df['Embarked'].astype(str))
test_df
final_model = XGBClassifier(n_estimators=best_para, learning_rate=0.05, n_jobs=4)

final_model.fit(X, y)
test_df.isna().sum()
test_df.Age.fillna(int(age_mean), inplace=True)

test_df.Fare.fillna(int(df.Fare.mean()), inplace=True)

df
preds = final_model.predict(test_df)
preds[:5]
sub_df = pd.DataFrame(data={

    'PassengerId': test_df.index,

    'Survived': preds

})
sub_df.Survived.value_counts()
sub_df.to_csv('submission.csv', index=False)