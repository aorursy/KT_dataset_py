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

print ("done")
train_data_initial = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

train_data_initial.head()
train_data_initial.info()
train_data_initial.drop(columns = ['Name','Ticket','Cabin'], inplace=True)
train_data = train_data_initial.copy()
train_data['Age'].fillna(value = train_data['Age'].mean(), inplace = True)

train_data['Embarked'].fillna(value = train_data['Embarked'].value_counts().idxmax(), inplace = True)
s = (train_data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder

label_train_data = train_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_train_data[col] = label_encoder.fit_transform(train_data[col])
label_train_data.info()
target_col = 'Survived'

y = label_train_data[target_col]

X = label_train_data.drop(columns=[target_col])

X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

preds_dict = []

for n_estimators in range (100,1001,100):

    for max_depth in range (6, 70,10):

        for max_leaf_nodes in range (5, 500, 50):

            parameters = {'n_estimators': n_estimators,

                              'max_depth': max_depth, 

                              'max_leaf_nodes': max_leaf_nodes

                             }

            model = RandomForestClassifier(**parameters)

            model.fit(X_train, y_train)

            preds = model.predict(X_valid)

            prediction = {}

            prediction['n_estimators'] = n_estimators

            prediction['max_depth'] = max_depth

            prediction['max_leaf_nodes'] = max_leaf_nodes

            prediction['accuracy_score'] = accuracy_score(y_true=y_valid, y_pred=preds)

            preds_dict.append(prediction)

print (preds_dict)
#max(preds_dict, prediction.keys)

count = 0

indexIs = 0

maxValue = preds_dict[0]['accuracy_score']

for i in preds_dict:

    if maxValue < i['accuracy_score']:

        print(count , ' :', i['accuracy_score'])

        maxValue = i['accuracy_score']

        indexIs = count

    count = count + 1

print(count ,': Max Val is :',maxValue)

print(preds_dict[indexIs])
test_data = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
test_data.drop(columns=['Name', 'Ticket','Cabin'], inplace=True)

test_data
test_data.info()
test_data['Age'].fillna(value = train_data['Age'].mean(), inplace = True)

test_data['Fare'].fillna(value = train_data['Fare'].mean(), inplace = True)
s = (test_data.dtypes == 'object')

object_cols_test = list(s[s].index)

print("Categorical variables:")

print(object_cols_test)
from sklearn.preprocessing import LabelEncoder

label_test_data = test_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols_test:

    label_train_data[col] = label_encoder.fit(train_data_initial[col].astype(str))

    label_test_data[col] = label_encoder.transform(test_data[col].astype(str))
label_test_data.info()
final_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=105, max_depth = 6, random_state= 0)

final_model.fit(X, y)

preds = final_model.predict(label_test_data)
sub_df = pd.DataFrame(data={

    'PassengerId': label_test_data.index,

    'Survived': preds

})

sub_df.to_csv('submission.csv', index=False)

print ('done')