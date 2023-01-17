
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import itertools as it

%matplotlib inline

sns.set_style("dark")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)
df_test['target'] = np.nan

df = pd.concat([df_train, df_test])

print(df.shape)
df_train.head()
df.head()
df.tail()
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
df_test['target'] = np.nan
df = pd.concat([df_train, df_test])
df['sex'] = df['sex'].replace(' Male', 0)
df['sex'] = df['sex'].replace(' Female', 1)
new_df = df
#generating features on target groupbied by every feature
for i in df.columns:
    if (i not in ['target', 'fnlwgt', 'uid']):
        df_tmp = df[df['target'].notna()].groupby([i])['target'].agg(
            ['mean', 'std']).rename(columns={
                'mean': f'target_mean_{i}',
                'std': f'target_std_{i}'
            }).fillna(0.0).reset_index()
        new_df = pd.merge(new_df, df_tmp, how='left', on=[i])
        new_df[f'target_mean_{i}'] = new_df[f'target_mean_{i}'].fillna(0.0)
        new_df[f'target_std_{i}'] = new_df[f'target_std_{i}'].fillna(0.0)
df = new_df
df = df.drop(columns=['uid', 'fnlwgt'])

df = pd.get_dummies(
    df, columns=[c for c in df_train.columns if df_train[c].dtype == 'object'])
df.head()
df_test.info()
df_train.isnull().sum()
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
our_x_train = df[df['target'].notna()].drop(columns=['target'])
our_y_train = df[df['target'].notna()]['target']
our_x_test = df[df['target'].isna()].drop(columns=['target'])
our_y_test = df[df['target'].isna()]['target']
X_train, X_test, y_train, y_test = train_test_split(our_x_train,
                                                    our_y_train,
                                                    test_size=0.2,
                                                    random_state=42)
plt.figure(figsize=(15, 10))

#critetion
plt.subplot(3, 3, 1)
feature_param = ['gini', 'entropy']
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(criterion=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Criterion')
plt.grid()

#max_depth
plt.subplot(3, 3, 2)
max_depth_check = range(1, 30)
scores = []
for depth in max_depth_check:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(max_depth_check, scores, '.-')
plt.axis('tight')
plt.title('Depth')
plt.grid()

#Splitter
plt.subplot(3, 3, 3)
feature_param = ['best', 'random']
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(splitter=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Splitter')
plt.grid()

#Min Samples Leaf
plt.subplot(3, 3, 4)
feature_param = range(2, 21)
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(min_samples_leaf=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Min Samples Leaf')
plt.grid()

#Min Samples Split
plt.subplot(3, 3, 5)
feature_param = range(2, 21)
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(min_samples_split=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Min Samples Split')
plt.grid()

#max_features
plt.subplot(3, 3, 6)
feature_param = range(1, 13)
scores = []
for feature in feature_param:
    clf = DecisionTreeClassifier(max_features=feature)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
plt.plot(feature_param, scores, '.-')
plt.axis('tight')
plt.title('Max Features')
plt.grid()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
#model for DT
clf = DecisionTreeClassifier(criterion='gini',
                             splitter='best',
                             max_depth=8,
                             min_samples_split=42,
                             min_samples_leaf=17)

clf.fit(X_train, y_train)
log_loss(y_test, clf.predict_proba(X_test)[:, 1])
#Final prediction
dt_predictions = clf.predict_proba(our_x_test)[:, 1]
dt_predictions
#Let's create a dataset for Knn model
cats = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]

nums = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week'
]
df_for_knn = pd.concat([df_train, df_test])
for i in range(1, 4):
    print(i)
    for g in it.combinations(cats, i):
        df_for_knn = pd.concat([
            df_for_knn,
            df_for_knn.groupby(list(g))[nums].transform('mean').rename(
                columns=dict([(s, ':'.join(g) + '__' + s + '__mean')
                              for s in nums]))
        ],
                               axis=1)
df_for_knn.drop(columns=cats, inplace=True)
cols = [c for c in df_for_knn.columns if c != 'uid' and c != 'target']
df_for_knn[cols] = StandardScaler().fit_transform(df_for_knn[cols])
df_m = df_for_knn[cols].corr()
cor = {}
for c in cols:
    cor[c] = set(df_m.loc[c][df_m.loc[c] > 0.5].index) - {c}

len(cor)
for c in cols:
    if c not in cor:
        continue
    for s in cor[c]:
        if s in cor:
            cor.pop(s)
cols = list(cor.keys())

len(cols)
#Make knn datasets
our_knn_x_train = df_for_knn.loc[df_for_knn['target'].notna()][cols]
our_knn_y_train = df_for_knn.loc[df_for_knn['target'].notna()]['target']
our_knn_x_test = df_for_knn.loc[df_for_knn['target'].isna()][cols]
our_knn_y_test = df_for_knn.loc[df_for_knn['target'].isna()]['target']
#knn CV
knn_x_train, knn_x_test, knn_y_train, knn_y_test = train_test_split(
    our_knn_x_train, our_knn_y_train, test_size=0.2, random_state=42)
#Knn model
model_knn = KNeighborsClassifier(weights='distance',
                                 algorithm='auto',
                                 leaf_size=30,
                                 n_neighbors=100,
                                 p=2,
                                 metric='minkowski',
                                 n_jobs=-1)
model_knn = model_knn.fit(knn_x_train, knn_y_train)
model_knn.predict_proba(knn_x_test)[:, 1]
#Check log_loss of knn
print(log_loss(knn_y_test, model_knn.predict_proba(knn_x_test)[:, 1]))
#Predicting whole dataset
knn_predictions = model_knn.predict_proba(our_knn_x_test)[:, 1]
#assumption to check log_loss
list_of_coefs, lst_of_best = [], []

for i in list(np.arange(0.7, 0.9, 0.01)):
    for j in list(np.arange(0.1, 0.3, 0.01)):
        if round(i, 3) + round(j, 3) == 1:
            list_of_coefs.append((round(i, 3), round(j, 3)))
            
list_of_coefs = list_of_coefs[::-1]
for i, j in list_of_coefs:
    search_best_error = log_loss(
        y_test,
        clf.predict_proba(X_test)[:, 1]) * i + log_loss(
            knn_y_test,
            model_knn.predict_proba(knn_x_test)[:, 1]) * j
    print('coefs:', i, j, 'log_loss: ', search_best_error)
    lst_of_best.append(search_best_error)
#create random samples for trees
import random
randomlist = []
for i in range(0,1000):
    n = random.randint(15000,len(our_x_train))
    randomlist.append(n)
#calculate score for every random sample
predictions = []
for i in randomlist:
    Xb = our_x_train.sample(n = i, replace=True)
    Yb = our_y_train[Xb.index]
    model = DecisionTreeClassifier(max_depth=random.randint(7,11),
                             min_samples_split=42,
                             min_samples_leaf=17)
    model.fit(Xb,Yb)
    predictions.append(model.predict_proba(our_x_test)[:,1])
#take mean
final_score_of_tree = np.array([i.mean() for i in np.array(predictions).T])
# log_loss(y_test,final_score_of_tree)
#create random samples for knns
randomlist_knn = []
for i in range(0,100):
    n = random.randint(15000,len(our_knn_x_train))
    randomlist_knn.append(n)
#calculate score for every random sample of knn
predictions_knn = []
for i in randomlist_knn:
    Xb = our_knn_x_train.sample(n = i, replace=True)
    Yb = our_knn_y_train[Xb.index]
    model_knn = KNeighborsClassifier(weights='distance',
                                 algorithm='auto',
                                 leaf_size=30,
                                 n_neighbors=random.randint(90,110),
                                 p=2,
                                 metric='minkowski',
                                 n_jobs=-1)
    model.fit(Xb,Yb)
    predictions_knn.append(model.predict_proba(our_knn_x_test)[:,1])
#take mean
final_score_of_knn = np.array([i.mean() for i in np.array(predictions_knn).T])
# log_loss(knn_y_test,final_score_of_knn)
#Lets make final prediction using 0.8*tree+0.2knn
p = 0.82 * final_score_of_tree + 0.18 * final_score_of_knn
sns.distplot(p)
df_submit = pd.DataFrame({
    'uid': new_df.loc[new_df['target'].isna()]['uid'],
    'target': p
})
df_submit.to_csv('/kaggle/working/submit.csv', index=False)

!head /kaggle/working/submit.csv
