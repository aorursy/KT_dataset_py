# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/xente-challenge/training.csv')

df.head()
df['FraudResult'].value_counts().plot(kind='bar');
for name in df.columns:

    if 'Id' in name:

        k = name.find('Id')

        df[name[:4]+'_id'] = df[name].map(lambda s: s[k+3:]).astype(int)

# df['tr_id'] = df['TransactionId'].map(lambda s: s[len("TransactionId_"):]).astype(int)
df.head()
df_1 = df.drop([name for name in df.columns if 'Id' in name], axis=1)

df_1.head()
df_2 = df_1.drop([name for name in df.columns if 'Code' in name], axis=1)

df_2.head()
df_3 = pd.get_dummies(df_2, columns=['ProductCategory'])

df_3.head()
df['ProductCategory'].value_counts()
df_3['amount is positive'] = (df_3['Amount'] > 0).astype(int)

df_3['amount != value'] = (np.abs(df_3['Amount']) != df_3['Value']).astype(int)

df_3.head()
df_4 = df_3.drop('Amount', axis=1)

df_4.head()
df_4['time'] = pd.to_datetime(df_4['TransactionStartTime'], format="%Y-%m-%dT%H:%M:%SZ")

df_4.head()
df_5 = df_4.drop('TransactionStartTime', axis=1)

df_5.head()
df_5.info()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=19)

X = df_5.drop(['time', 'FraudResult'], axis=1)

y = df_5['FraudResult']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=19)
tree.fit(X_train, y_train)
# картинка

from sklearn.tree import export_graphviz

tree_dot = export_graphviz(tree)

print(tree_dot)
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
(len(y_valid) - sum(y_valid))/len(y_valid)
y_pred
y_valid
k = 0

for i in range(len(y_valid)):

    if y_valid.values[i] == 1 and y_pred[i] == 1:

        k += 1

print(k)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, y_pred)
from sklearn.metrics import f1_score

f1_score(y_valid, y_pred)