from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
load_path = './'
X_train = pd.read_csv(load_path+'train.csv')
X_test = pd.read_csv(load_path+'test.csv')
X_train.shape, X_test.shape
X_train.head()
X_test.head()
y_train = X_train['target']
del X_train['target']
X_train.shape
X_train.head()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
from sklearn.linear_model import RidgeClassifier
cls = RidgeClassifier()
cls.fit(X_train, y_train)
cls.score(X_val, y_val)
from sklearn.svm import SVC
cls = SVC(gamma='auto')
cls.fit(X_train, y_train)
cls.score(X_val, y_val)
from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(n_estimators=100)
cls.fit(X_train, y_train)
cls.score(X_val, y_val)
y_pred = cls.predict(X_test)
df_op = pd.DataFrame(data=y_pred, columns=['target'])
df_op.index = pd.Series(range(df_op.shape[0]),name='id')
df_op.head()
df_op.to_csv('submission.csv')