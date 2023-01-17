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
df_train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
df_train.head(5)
df_train.hist()
df_train.describe()
df_train['Sex'].value_counts()
df_train['SmokingStatus'].value_counts()
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df_train['Sex'] = df_train['Sex'].replace({'Male':0,'Female':1})
df_train['SmokingStatus'] = df_train['SmokingStatus'].replace({'Ex-smoker':0,'Never smoked':1,'Currently smokes':2})
df_train = df_train.sort_values(['Patient','Weeks'])
df = df_train.drop_duplicates(subset = ['Patient'],keep='first')
X = df_train[['Age','Sex','SmokingStatus']]
y = df_train[['FVC']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 42)
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test,y_test)
forest = RandomForestRegressor(max_depth=2, random_state=0)
forest.fit(X_train,y_train['FVC'])
forest.score(X_test,y_test)
!pip install rfpimp
from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(forest, X_train, y_train, r2)
perm_imp_rfpimp
df_train[df_train['Sex']==0]['FVC'].hist()
df_train[df_train['Sex']==1]['FVC'].hist()
df_train.plot.scatter(x='Age',y='FVC')