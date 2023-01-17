import pandas as pd

import numpy as np

from sklearn import svm



data = pd.read_csv('../input/train.csv')
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

data = data.dropna()
data['Gender'] = data['Sex'].map({'female': 0, 'male':1}).astype(int)
data['Embarked'].unique()

data['Port'] = data['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
data = data.drop(['Sex', 'Embarked'], axis=1)
cols = data.columns.tolist()

cols = [cols[1]] + cols[0:1] + cols[2:]

data = data[cols]
train_data = data.values
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(train_data[0:,2:], train_data[0:,0])
df_test = pd.read_csv('../input/test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)



df_test = df_test.dropna()



df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})

df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})



df_test = df_test.drop(['Sex', 'Embarked'], axis=1)



test_data = df_test.values
data.head(10)
df_test.head(10)
output = rbf_svc.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.head(10)