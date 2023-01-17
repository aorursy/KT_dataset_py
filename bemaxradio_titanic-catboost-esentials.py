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
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
drop_cols = ['Cabin', 'Ticket']

train_df = train_df.drop(columns=drop_cols)

test_df = test_df.drop(columns=drop_cols)
train_df['Embarked'].dropna().value_counts()
embarked_median = 'S'

train_df['Embarked'].fillna(embarked_median, inplace=True)

test_df['Embarked'].fillna(embarked_median, inplace=True)
for feat in train_df.drop(columns=['Name']):

    print(feat, f'Unique vals = {train_df[feat].nunique()}')

    train_df[feat].hist(label=feat)

    plt.show()
age_fillna = -1

train_df['Age'].fillna(age_fillna, inplace=True)

test_df['Age'].fillna(age_fillna, inplace=True)
y = train_df.pop('Survived')

X = train_df.drop(columns=['PassengerId', 'Name'])
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
try:

    from catboost import CatBoostClassifier

except:

    !pip install catboost

    from catboost import CatBoostClassifier
model = CatBoostClassifier(

    boosting_type='Ordered',

    learning_rate=0.19,

    l2_leaf_reg=0.07,

    max_depth=5,

    n_estimators=85,

    custom_metric=['Accuracy'],

    class_weights=[1, 1.3],

    silent=True

    )
cat_features = [feat for feat in train_df if train_df[feat].nunique() < 10]
model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, cat_features=cat_features);
from sklearn.metrics import classification_report

y_pred = model.predict(X_val)

print(classification_report(y_val, y_pred))
model.fit(X, y, cat_features=cat_features, );
X_test = test_df.drop(columns=['PassengerId', 'Name'])

X_test
y_test_pred = pd.Series(model.predict(X_test), name='Survived')

output = pd.concat([test_df['PassengerId'], y_test_pred], axis=1)
output.to_csv('prediction_results.csv', index=False)