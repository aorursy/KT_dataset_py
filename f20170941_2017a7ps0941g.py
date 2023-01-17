import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
dataset = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

dataset_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
dataset.head()
dataset.isnull().sum()

dataset = dataset.fillna(dataset.mean())

dataset_test = dataset_test.fillna(dataset_test.mean())
labelencoder = LabelEncoder()

labelencoder.fit(dataset['type'])

dataset['type'] = labelencoder.transform(dataset['type'])



X_train = dataset.drop(columns=['id','rating'])

y_train = dataset['rating']



X_test = dataset_test.drop(columns=['id'])

labelencoder.fit(X_test['type'])

X_test['type'] = labelencoder.transform(X_test['type'])


from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score



clf = ExtraTreesRegressor(n_estimators=700)



clf.fit(X_train, y_train)

y_final = clf.predict(X_test)





df_final = pd.DataFrame({'id': dataset_test['id'], 'rating': np.round(y_final)})

df_final.to_csv('submission.csv', index=False)