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

df_1 = df.drop([name for name in df.columns if 'Id' in name], axis = 1)

df_1.head()
df['FraudResult'].value_counts()#.plot(kind = 'bar')

for name in df.columns:

    if 'Id' in name:

        k = name.find('Id')

        df[name[:4]+'_id'] = df[name].map(lambda s: s[k+3]).astype(int)

df.head()
df_1 = df.drop([name for name in df.columns if 'Id' in name], axis = 1)

df_1.head()
df_2 = df_1.drop([name for name in df.columns if 'Code' in name], axis = 1)

df_2.head()
df_3 = pd.get_dummies(df_2, columns = ['ProductCategory'])

df_3.head()

df_3['amount is positive'] = (df_3['Amount']>0).astype(int)

df_3['amount != value'] = (np.abs(df_3['Amount']) != df_3['Value']).astype(int)

df_3.head()
df_4 = df_3.drop(['Amount'], axis =1)

df_4.columns
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz

tree= DecisionTreeClassifier(max_depth = 4, random_state = 19)

X = df_4.drop(['TransactionStartTime','FraudResult'], axis =1)

y = df_4['FraudResult']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state =19)

tree.fit(X_train, y_train)

tree_dot = export_graphviz(tree)

print(tree_dot)
y_pred = tree.predict(X_valid)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, y_pred)