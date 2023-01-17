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
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import ADASYN
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv', header=0, delimiter = ',')

pd.set_option('display.max_columns', None)

print(df.shape)
df.head()
df = df.drop(['id'], axis=1)

df.head()
df['target'].value_counts()

df.groupby('target').mean()
Y_train = df['target']

temp = [i for i in df if i not in Y_train]

X_train = df[temp]

X_train = X_train.drop(['target'], axis = 1)
X_resampled, Y_resampled = ADASYN().fit_sample(X_train, Y_train)

X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)

X_resampled.shape
logreg = LogisticRegression(max_iter=500)

Y_resampled = np.ravel(Y_resampled)
logreg.fit(X_resampled, Y_resampled)
X_resampled.head()
Y_resampled
X_test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

X_test.head()
id_list = X_test['id']

X_test = X_test.drop(['id'], axis=1)

X_test.head()
X_test.shape
predictions = logreg.predict_proba(X_test)

df_temp = pd.DataFrame(predictions)

df_temp.head()
Y_pred = df_temp.drop([0], axis =1)

df_test = X_test

df_test['target']= Y_pred
df_test.head()
id_list.shape
id_list1=pd.DataFrame(id_list)

id_list1
id_list1.shape
Y_pred
Y_pred.shape
df_new = pd.concat([pd.DataFrame(id_list1), pd.DataFrame(Y_pred)], axis=1)

df_new.head()
df_new.columns.values[1] = 'target'

df_new.head()
df_new.to_csv('submission.csv',index=False)