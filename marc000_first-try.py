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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.info()
X_train = df_train.copy(deep=False)

del X_train['loss']

del X_train['id']

y_train = pd.DataFrame(df_train['loss'])



X_test = df_test.copy(deep=False)

del X_test['id']
X_train_d = pd.get_dummies(X_train)

X_test_d = pd.get_dummies(X_test)
X_train_d.head()
X_train_d.info()
from sklearn.decomposition import PCA

pca = PCA(n_components=12)
pca.fit(X_train_d)

X_train = pd.DataFrame(pca.transform(X_train_d))
pca.fit(X_test_d)

X_test = pd.DataFrame(pca.transform(X_test_d))
X_train.info()
X_train.head()
y_train.head()
from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
df = pd.DataFrame(y_pred)
df.head()
df['id'] = df_test['id']
df.columns = ['loss', 'id']
df = df[['id', 'loss']]
df.head()
df.to_csv('submission.csv', index=False)