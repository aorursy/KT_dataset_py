import pandas as pd

from fastai.tabular import *

from sklearn.ensemble import RandomForestClassifier
path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')
df.head()
df.dtypes
# fill missing

df.isnull().sum()
df['missing_education-num'] = df['education-num'].isnull().map({True: 1, False:0})

df['missing_occupation'] = df['occupation'].isnull().map({True: 1, False:0})

df.head()
val = df['education-num'].median()

df['education-num'] = df['education-num'].fillna(val)
val = 'no_occupation'

df['occupation'] = df['occupation'].fillna(val)
df.head()
for col in df.select_dtypes(include=['object']):

    df[col] = df[col].astype('category')
for col in df.select_dtypes(include=['category']):

    df[col] = df[col].cat.codes
df.head()
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
X = df.drop('salary', axis = 1)

y = df['salary']
len(df)
n_valid = int(len(df) * 0.2)  # same as Kaggle's test set size

n_trn = len(df)-n_valid



X_train, X_valid = split_vals(X, n_trn)

y_train, y_valid = split_vals(y, n_trn)



X_train.shape, y_train.shape, X_valid.shape
clf = RandomForestClassifier(n_jobs=-1, max_features=0.5, n_estimators=20)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_valid)
from sklearn.metrics import accuracy_score
accuracy_score(y_valid, predictions)