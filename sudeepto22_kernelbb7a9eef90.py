# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')

df = pd.merge(X_train,y_train,how='outer',on='series_id')
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df['surface_code'] = label.fit_transform(df.surface)
surface_dict = df[['surface','surface_code']].drop_duplicates().sort_values(by='surface_code').set_index('surface_code').to_dict()['surface']
df.drop(['row_id','series_id','measurement_number','group_id','surface'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split as tts

X = df.drop(['surface_code'], axis=1)

y = df['surface_code']

X_train, X_test, y_train, y_test = tts(X,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

y_pred = dtree.predict(X_test)

print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,y_train)

y_pred = random_forest.predict(X_test)

print(accuracy_score(y_test,y_pred))
X_test = pd.read_csv('../input/X_test.csv')

X_test_series_id = X_test['series_id']

X_test.drop(['row_id','series_id','measurement_number'], axis=1, inplace=True)
y_pred = random_forest.predict(X_test)
df_submission = pd.DataFrame(list(zip(list(X_test_series_id),list(y_pred))), columns=['series_id','surface'])
from scipy import stats

df_submission = df_submission.groupby('series_id').agg(lambda x: stats.mode(x)[0]).reset_index()

df_submission.surface = df_submission.surface.map(lambda x: surface_dict[x])

df_submission.to_csv('submission.csv', index=False)