# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df  = pd.read_csv('../input/train.csv')
df.head(10)
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df.info()
df = df.dropna()
df.info()
df['Sex'].unique()
df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
df['Embarked'].unique()
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()

print(cols)
cols = [cols[1]] + cols[0:1] + cols[2:]

df = df[cols]
df.head(10)
df.info()
train_data = df.values
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:, 2:], train_data[0:, 0])
df_test = pd.read_csv('../input/test.csv')
df_test.head(5)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)



df_test = df_test.dropna()

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})

df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)



test_data = df_test.values
output = model.predict(test_data[:,1:])
result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.head(10)