from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

import pandas as pd

import numpy as np

from sklearn.preprocessing import scale

from sklearn.utils import shuffle
df = pd.read_csv("../input/creditcard.csv")
df = shuffle(df, random_state = 123)
df.columns
import scipy.stats as sts

sts.describe(df['Time'])
from matplotlib import pyplot as plt

%matplotlib inline

df['Time'].hist()
df_copy = df

y = df_copy["Class"]

df_copy.drop(["Class"], axis = 1, inplace = True)

columns = df_copy.columns

X = scale(df_copy)
LR = LogisticRegression(random_state = 123, n_jobs = -1)

LR.fit(X, y)

scores = cross_val_score(LR, X, y, cv = 5)

df0 = pd.DataFrame(list(zip(columns, abs(LR.coef_[0]))))

df0.columns = ['feature name', 'coef_']

df0.sort_values(by = 'coef_', ascending = False)
print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))
sts.describe(df['V26'])
df['V26'].hist()
columns = df_copy.columns

X = scale(df_copy)



df_copy.loc[df_copy['V26'] < -2, 'V26_strange'] = df_copy['V26']

df_copy.loc[df_copy['V26'] > 2, 'V26_strange'] = df_copy['V26']

df.loc[df['V26'] < -2, 'V26'] = 0

df.loc[df['V26'] > 2, 'V26'] = 0

df_copy['V26_strange'].fillna(0, inplace = True)



LR = LogisticRegression(random_state = 123, n_jobs = -1)

LR.fit(X, y)

scores = cross_val_score(LR, X, y, cv = 5)

df0 = pd.DataFrame(list(zip(columns, abs(LR.coef_[0]))))

df0.columns = ['feature name', 'coef_']

df0.sort_values(by = 'coef_', ascending = False)
print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))
sts.describe(df['V18'])
df['V18'].hist()
columns = df_copy.columns

X = scale(df_copy)



df_copy.loc[df_copy['V18'] < -2, 'V18_strange'] = df_copy['V18']

df_copy.loc[df_copy['V18'] > 2, 'V18_strange'] = df_copy['V18']

df.loc[df['V18'] < -2, 'V18'] = 0

df.loc[df['V18'] > 2, 'V18'] = 0

df_copy['V18_strange'].fillna(0, inplace = True)



LR = LogisticRegression(random_state = 123, n_jobs = -1)

LR.fit(X, y)

scores = cross_val_score(LR, X, y, cv = 5)

df0 = pd.DataFrame(list(zip(columns, abs(LR.coef_[0]))))

df0.columns = ['feature name', 'coef_']

df0.sort_values(by = 'coef_', ascending = False)
print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))
sts.describe(df['V19'])
df['V19'].hist()
columns = df_copy.columns

X = scale(df_copy)



df_copy.loc[df_copy['V19'] < -2, 'V19_strange'] = df_copy['V19']

df_copy.loc[df_copy['V19'] > 2, 'V19_strange'] = df_copy['V19']

df.loc[df['V19'] < -2, 'V19'] = 0

df.loc[df['V19'] > 2, 'V19'] = 0

df_copy['V19_strange'].fillna(0, inplace = True)



LR = LogisticRegression(random_state = 123, n_jobs = -1)

LR.fit(X, y)

scores = cross_val_score(LR, X, y, cv = 5)

df0 = pd.DataFrame(list(zip(columns, abs(LR.coef_[0]))))

df0.columns = ['feature name', 'coef_']

df0.sort_values(by = 'coef_', ascending = False)
print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))