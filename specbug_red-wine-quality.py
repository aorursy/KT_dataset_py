import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.metrics import accuracy_score

%matplotlib inline
# Imputer to fill in missing values
# Classified the continous sequence into 'good' & 'bad' categories

df = pd.read_csv('../input/winequality-red.csv')
imputer = SimpleImputer()
imputer.fit_transform(df)
bins = (1, 6.5, 12)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)
df['quality'] = df['quality'].cat.codes
X = df.drop(['quality'], axis=1)
y = df[['quality']]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1511)

#SVC generates an accuracy of around 0.91

t0 = time()
clf = SVC(C=1, kernel='rbf')
clf.fit(train_X, train_y)
print('Training time', round(time() - t0, 3), 's')
pred = clf.predict(test_X)
pred = pred.tolist()
for i in range(len(pred)):
  pred[i] = round(pred[i])
print('Score', accuracy_score(test_y, pred))
# Random Forest generates an accuracy of around 0.93

t00 = time()
clf1 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_depth=100)
clf1.fit(train_X, train_y)
print('Training time', round(time() - t00, 3), 's')
pred1 = clf1.predict(test_X)
pred1 = pred1.tolist()
for i in range(len(pred1)):
  pred[i] = round(pred1[i])
print('Score', accuracy_score(test_y, pred1))
