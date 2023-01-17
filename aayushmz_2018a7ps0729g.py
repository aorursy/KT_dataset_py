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
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
df= pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

df_test =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
correlations = []

for column in df.columns:

    correlations.append((abs(df[column].corr(df['target'])), column))

correlations.sort(reverse=True)

for i, (cor, col) in enumerate(correlations):

    print(i, col, cor)
X_train = df.drop(['id', 'col_45','col_31','col_47','col_41','col_64','col_46','col_85','col_44','col_54','col_81','col_29','col_71','col_87','col_27','col_50','col_21', 'target'], 1)

y_train = df['target']
X_test = df_test.drop(['id', 'col_45','col_31','col_47','col_41','col_64','col_46','col_85','col_44','col_54','col_81','col_29','col_71','col_87','col_27','col_50','col_21'], 1)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
for i in range(len(y_pred)):

    if y_pred[i] < 0:

        y_pred[i] = 0
submission = pd.DataFrame()
submission['id'] = df_test['id']

submission['target'] = y_pred
submission.to_csv('submission.csv', index=False)