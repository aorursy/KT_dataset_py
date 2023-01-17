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
df_train=pd.read_csv('/kaggle/input/breast-cancer/train.csv')

df_test=pd.read_csv('/kaggle/input/breast-cancer/test.csv')

df_sub=pd.read_csv('/kaggle/input/breast-cancer/sample-submission.csv')
df_sub
y=df_train['class']

x=df_train.drop(['class'],axis=1)

x.drop(['Id'],inplace=True,axis=1)

x.info()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_train.describe()
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d"% (x_test.shape[0], (y_test != y_pred).sum()))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test) 

y_test.head()
df_test.drop(['Id'],inplace=True,axis=1)

y_submission=clf.predict(df_test)

y_submission
dataset = pd.DataFrame({'Id':df_sub['Id'], 'class': y_submission[:]})

dataset
dataset.to_csv('my_submission.csv',index=False)

print("Success!!")