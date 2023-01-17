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
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df = df.drop(['sl_no'], axis = 1)

df.head()
df.info()
df.fillna(0, inplace=True)
df = df.replace(['F','M'], [0,1])

df = df.replace(['Others','Central'], [0,1])

df = df.replace(['Placed','Not Placed'], [0,1])

df = df.replace(['Mkt&HR','Mkt&Fin'], [0,1])

df = df.replace(['Sci&Tech','Comm&Mgmt','Others'], [0,1,2])

df = df.replace(['Commerce','Science','Arts'], [0,1,2])

df = df.replace(['Yes','No'], [0,1])
df.hsc_s.unique()
df.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X = df.drop(['status'],axis= 1)

y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X,y)
scalee = StandardScaler()

scalee.fit(X_train)

X_train = scalee.transform(X_train)

X_test = scalee.transform(X_test)
model = KNeighborsClassifier(n_neighbors = 5)

model.fit(X_train,y_train)
model.score(X_test,y_test)