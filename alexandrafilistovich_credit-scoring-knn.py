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
import pandas as pd

df_train = pd.read_csv('/kaggle/input/creditscoringds/df_train.csv', sep=',')

df_test = pd.read_csv('/kaggle/input/creditscoringds/df_test.csv', sep=',')

df_test = df_test.drop(['ProductCategory_ticket'], axis=1)



tr_id = pd.read_csv('/kaggle/input/mydata/Test.csv', sep=',')

tr_id.head()
df_test.head()
y_test = tr_id[['TransactionId']]

y_test['IsDefaulted'] = 1

y_test.head()
X_test = df_test

X_test.head()
df_train.head()
'''from sklearn.utils import shuffle

df_train = shuffle(df_train)

df_train.head()'''
df_te = df_test.drop(['TransactionStatus'], axis=1)

df_te.head()
df_tr = df_train



df_tr = df_train.drop(np.where(df_train['TransactionStatus'] == 0)[0])

df_tr = df_tr.drop(['TransactionStatus'], axis=1)

df_tr.head()
df_tr_X = df_tr.drop('IsDefaulted', axis=1)

df_tr_X.head()
df_tr_y = df_tr[['IsDefaulted']]

df_tr_y.head()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



neighbors = KNeighborsClassifier(n_neighbors=5)

X = df_tr_X.values

y = df_tr_y.values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                     test_size=0.3, 

                                                     random_state=17)

neighbors.fit(X_train, y_train)

print('Accuracy: \n', neighbors.score(X_valid, y_valid))
y_pred = neighbors.predict(X_valid)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, y_pred)
neighbors.fit(X, y)
y_x = neighbors.predict(df_te)

for i in range(y_test.shape[0]):

    y_test.at[i,'IsDefaulted'] = y_x[i]

y_test.head()
y_test.to_csv('ANSWER_knn.csv',index=False)