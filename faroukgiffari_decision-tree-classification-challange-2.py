# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv')
df1.head()
df1.isnull().values.any()
df1.describe()
df1.columns
df1['user_type'] = df1['user_type'].astype('category')

df1['user_type'] = df1['user_type'].cat.codes

df1['trip_duration_seconds'] = df1['trip_duration_seconds'].astype('int')

df1['from_station_id'] = df1['from_station_id'].astype('int')

df1['to_station_id'] = df1['to_station_id'].astype('int')

df1.dtypes
df1.head()
df1.shape
Duration = df1['trip_duration_seconds']

Duration
X = df1[['trip_duration_seconds','from_station_id','to_station_id']] .values  

X[0:5]
y= df1[['user_type']].values
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split( X, y, test_size=0.1, random_state=2)

print ('Train set:', X_trainset.shape,  y_trainset.shape)

print ('Test set:', X_testset.shape,  y_testset.shape)
Kelastree = DecisionTreeClassifier(criterion="", max_depth = 4)

Kelastree
Kelastree.fit('X_trainset','y_trainset')
predKelas = Kelas.predict(X_testset)
from sklearn import metrics

import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predKelas))