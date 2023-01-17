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
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

df

#سام p .

#e غير سام 
df.dtypes
from sklearn.preprocessing import LabelEncoder

df.apply(LabelEncoder().fit_transform)

#لتحويلهم الى ارقام 
df['class'].unique()
df.columns
df['population'].unique()
df.describe()
df.head()
y = df.odor
X=df.drop(columns ='odor')
X.describe()
X.head()
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,train_size = 0.8 , test_size = 0.2 ,random_state=1)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

X, y = df(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

df = GaussianNB()

y_pred = df.fit(X_train, y_train).predict(X_test)