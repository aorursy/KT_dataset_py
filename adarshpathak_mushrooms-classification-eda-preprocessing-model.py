import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DATA_PATH = '/kaggle/input/mushroom-classification/'
file_path = os.path.join(DATA_PATH,'mushrooms.csv')
pd.set_option('display.max_columns',30)
df = pd.read_csv(file_path)
print(f'shape of csv file: {df.shape}')
df.head()
df.columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat']
for i in df.columns:

    print(f'{i} -> {df[i].unique()}')
for i in df.columns:

    if df[i].dtype == 'object':

        df[i] = pd.factorize(df[i])[0]
df.groupby(['cap-shape'])['target'].value_counts()
pd.crosstab(df['cap-shape'],df['target'])
fig = px.violin(df,

          x = df['cap-shape'],

          y=df['target'])

fig.show()
fig = px.violin(df,

          x = df['cap-surface'],

          y=df['target'])

fig.show()
from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.feature_selection import mutual_info_classif
y = df.target
df.drop('target',axis =1,inplace=True)
x = df
vrt = VarianceThreshold(threshold=0.01)

vrt.fit(x,y)
sum(vrt.get_support())
X = vrt.transform(df)
chi2_selector = SelectKBest(chi2, k=11)

X_kbest = chi2_selector.fit_transform(X, y)
X_kbest.shape
mut_feat = mutual_info_classif(X_kbest,y)
mut_feat
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X_kbest,y,test_size=0.15,random_state=1)
lr = LogisticRegression(max_iter=200)

lr.fit(X_train,y_train)
lr.score(X_train,y_train)
cross_val_score(lr,X_train,y_train,cv=5)
lr.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_features=9,max_depth=5,n_estimators=10)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
cross_val_score(rf,X_train,y_train,cv=5)
rf.feature_importances_
rf.score(X_test,y_test)
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,auc
y_pred = rf.predict(X_test)
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title(f'tpr vs fpr plot with auc: {roc_auc_score(y_test,y_pred)}')

plt.show()