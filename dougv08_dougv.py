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

import numpy as np 

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/kaggle/input/equipfailstest/equip_failures_training_set.csv')
df.describe()
cols = df.columns

for col in cols:

    df[col] = pd.to_numeric(df[col], errors='coerce')
df.isna().sum().sum()
df.fillna(-1, inplace=True)
df.isna().sum().sum()
df.target.plot(kind='hist')
split=0.2

df_fail = df[df.target == 1]

df_normal = df[df.target == 0].sample(n=int(df_fail.shape[0]/split), random_state=42)

bal = df_fail.append(df_normal)
bal.target.plot(kind='hist')
X, y = bal.iloc[:,2:] , bal['target']

total, test = df.iloc[:,2:] , df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()

scaler.fit(X)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

total = scaler.transform(total)
params_grid = {

    'n_neighbors' : list(range(2,20,1)),

    'weights' : ['uniform', 'distance']

}
clf = GridSearchCV(KNeighborsClassifier(), params_grid, scoring='f1', cv=5)
clf.fit(X_train,y_train)
clf.best_params_
y_pred = clf.predict(X_test)
print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
pred = clf.predict(total)
print(f'The total accuracy score: {accuracy_score(test, pred)}')

print(f'The total F1 Score is: {f1_score(test,pred)}')
from joblib import dump, load
dump(clf, 'gridsearch_knn.joblib')

dump(scaler, 'scaler.joblib')