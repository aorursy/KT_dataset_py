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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import graphviz
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('../input/eenbastats-neelvora/TeamBoxScore1617.csv')
df.head()
df.isnull().sum().max()
df = df.dropna(axis=0)
df.shape
df['teamRslt'] = df['teamRslt'].replace('Win',1)
df['teamRslt'] = df['teamRslt'].replace('Loss',0)
df['teamRslt'] = df['teamRslt'].astype('int')

df['teamLoc'] = df['teamLoc'].replace('Home',1)
df['teamLoc'] = df['teamLoc'].replace('Away',0)
df['teamLoc'] = df['teamLoc'].astype('int')

corrmat = df.corr()
f, ax = plt.subplots(figsize=(50,50))
sns.heatmap(corrmat, vmax=.8, square=True)
k = 12
cols = corrmat.nlargest(k, 'teamRslt')['teamRslt'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'pace')['pace'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'PreGame AVG T3PA')['PreGame AVG T3PA'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'PreGame AVG T3P%')['PreGame AVG T3P%'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'AVG TOrtg')['AVG TOrtg'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'AVG TDrtg')['AVG TDrtg'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'AVG TFG%')['AVG TFG%'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k, 'Pregame AVG TFGA')['Pregame AVG TFGA'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k,'AVG TAST/TO')['AVG TAST/TO'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k,'AVG TAST/TO')['AVG TAST/TO'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k,'AVG Oortg')['AVG Oortg'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k,'AVG Odrtg')['AVG Odrtg'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
k = 12
cols = corrmat.nlargest(k,'teamLoc')['teamLoc'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
features_selected = ["PreGame AVG T3PA", "PreGame AVG T3P%", "AVG TOrtg", "AVG TDrtg", "AVG TAST/TO", "AVG Oortg", "AVG Odrtg", "pace", "teamLoc"]
x = df[features_selected]
y = df["teamRslt"]
x.head()
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.4, random_state=2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
kncm = KNeighborsClassifier(n_neighbors=5)
kncm.fit(x_train, y_train)
pred = kncm.predict(x_test)
print(metrics.accuracy_score(y_test, pred))
print(kncm.predict_proba(x_test))
lsvcm = LinearSVC(random_state=2)
lsvcm.fit(x_train, y_train)
print(lsvcm.coef_)
print(lsvcm.intercept_)
pred = (lsvcm.predict(x_test))
print(metrics.accuracy_score(y_test, pred))
rfcm = RandomForestClassifier()
rfcm.fit(x_train, y_train)
pred = rfcm.predict(x_test)
print(metrics.accuracy_score(y_test, pred))
lrm = LogisticRegression()
lrm.fit(x_train, y_train)
pred = lrm.predict(x_test)
print (pred)
print(metrics.accuracy_score(y_test, pred))
gbcm = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)
gbcm.score(x_test, y_test)
df2 = pd.read_csv('../input/eenbastats-neelvora/TeamBoxScore1718.csv')
df2.head()
new_features_selected = ["PreGame AVG T3PA", "PreGame AVG T3P%", "AVG TOrtg", "AVG TDrtg", "AVG TAST/TO", "AVG Oortg", "AVG Odrtg", "pace", "teamLoc"]
df2['teamRslt'] = df2['teamRslt'].replace('Win',1)
df2['teamRslt'] = df2['teamRslt'].replace('Loss',0)
df2['teamRslt'] = df2['teamRslt'].astype('int')

df2['teamLoc'] = df2['teamLoc'].replace('Home',1)
df2['teamLoc'] = df2['teamLoc'].replace('Away',0)
df2['teamLoc'] = df2['teamLoc'].astype('int')

df2.isnull().sum().max()
df2 = df2.dropna(axis=0)
df2.shape

z = df2[new_features_selected]
Y = df2["teamRslt"]
z.head()

lrm = LogisticRegression()
lrm.fit(x, y)
pred = lrm.predict(z)
print (pred)
print(metrics.accuracy_score(Y, pred))