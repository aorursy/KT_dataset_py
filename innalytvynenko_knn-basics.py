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
df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', sep = ';')

df.head()
df_1 = df.drop('id', axis = 1)

df_1['age_years'] = df_1['age'] / 365.25



df_2 = df_1.drop('age', axis = 1)

df_2.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



y = df_2['cardio']

df_3 = df_2.drop('cardio', axis = 1)
X = df_3

X_new = scaler.fit_transform(X)

X_new
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 10)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_valid, y_pred))