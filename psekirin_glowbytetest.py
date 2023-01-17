# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/data"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
df = pd.read_csv('../input/data/data.csv')
df.head()
idgrp = df.groupby(['id', 'cycle'])

cnted = idgrp.aggregate('count')

cnted[cnted['p02'] > 1]
df_make_sense = df.drop(['p00', 'p01', 'p07', 'p09', 'p10', 'p16'], 1)

df_make_sense.describe()

#df_make_sense.mean()[2:-2].sum()-df.mean()['p05'] - df.mean()['p20']
corrs = df_make_sense[df_make_sense.columns].corr()

factors = corrs.columns

for factor in factors:

    for idx, corr in zip(factors, corrs[factor]):

        if abs(corr) > 0.90 and factor != idx:

            print(factor, idx, corr)
df_make_sense = df_make_sense.drop(['p20'], 1)
fig, splts = plt.subplots(16, 1, figsize=(16, 16*10))

splts_names = ['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p17', 'p18', 'p19', 's1', 's2']

for cur_splt, cur_col in zip(splts, splts_names):

    for cur_id in range(80):

        x_vals = df_make_sense[df_make_sense['id'] == cur_id]['cycle']

        y_vals = df_make_sense[df_make_sense['id'] == cur_id][cur_col]

        cur_splt.set_title(cur_col)

        cur_splt.plot(x_vals, y_vals, marker='.')

plt.show()
window_size = 10

windowed_vals = [i for i in range(1, window_size)]

df_rolled = df_make_sense[['p02', 'p03', 'p04', 'p05', 'p06', 'p08', 'p11', 'p12', 'p13', 'p14', 'p15', 'p18', 'p19']].rolling(window_size).mean()

for cur_col in ['p02', 'p03', 'p04', 'p05', 'p06', 'p08', 'p11', 'p12', 'p13', 'p14', 'p15', 'p18', 'p19']:

    df_rolled.loc[df_make_sense['cycle'].isin(windowed_vals), cur_col] = df_make_sense.loc[df_make_sense['cycle'].isin(windowed_vals), cur_col]

df_rolled.insert(0, 'id', df_make_sense['id'])

df_rolled.insert(1, 'cycle', df_make_sense['cycle'])

df_rolled.head(20)





splts_names = ['p02', 'p03', 'p04', 'p05', 'p06', 'p08', 'p11', 'p12', 'p13', 'p14', 'p15', 'p18', 'p19']

fig, splts = plt.subplots(len(splts_names), 1, figsize=(16, 10*len(splts_names)))

for cur_splt, cur_col in zip(splts, splts_names):

    for cur_id in range(80):

        x_vals = df_rolled[df_rolled['id'] == cur_id]['cycle']

        y_vals = df_rolled[df_rolled['id'] == cur_id][cur_col]

        cur_splt.set_title(cur_col)

        cur_splt.plot(x_vals, y_vals, marker='.')

plt.show()
max_cycle = df_make_sense.groupby('id').aggregate('max')['cycle']

last_cycle = []

for k in range(1, 81):

    for i in range(max_cycle[k] - 1):

        last_cycle.append(0)

    last_cycle.append(1)

df_rolled['s1'] = df_make_sense['s1']

df_rolled['s2'] = df_make_sense['s2']

df_rolled['last_cycle'] = last_cycle
columns_to_draw = ['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 's1', 's2']

amount_of_cols = len(columns_to_draw)

fig, splts = plt.subplots(amount_of_cols, amount_of_cols, figsize=(6*amount_of_cols, amount_of_cols*6))

for cur_splt_row, cur_row in zip(splts, columns_to_draw):

    for cur_splt, cur_col in zip(cur_splt_row, columns_to_draw):

        cur_splt.set_title(cur_row + '_' + cur_col)

        cur_splt.scatter(df_rolled[df_rolled['last_cycle'] == 0][cur_col], df_rolled[df_rolled['last_cycle'] == 0][cur_row], marker='.', color='blue')

        cur_splt.scatter(df_rolled[df_rolled['last_cycle'] == 1][cur_col], df_rolled[df_rolled['last_cycle'] == 1][cur_row], marker='.', color='red')

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=1)

X = df_rolled[['p02', 'p03', 'p04', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15', 'p18', 'p19']]

X_r = pca.fit(X).transform(X)

print('score', pca.get_params())

#plt.scatter(X_r.T[0], X_r.T[1])

#plt.show()



df_pca = df_rolled[:]

df_pca.insert(15, 'pca1', X_r[:])

#df_pca.insert(16, 'pca2', X_r.T[1][:])

#df_pca['pca2'] = X_r.T[1][:]

print(X_r[:5])

df_pca.head()

columns_to_draw = ['pca1', 'p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 's1', 's2']

amount_of_cols = len(columns_to_draw)

fig, splts = plt.subplots(amount_of_cols, amount_of_cols, figsize=(6*amount_of_cols, amount_of_cols*6))

for cur_splt_row, cur_row in zip(splts, columns_to_draw):

    for cur_splt, cur_col in zip(cur_splt_row, columns_to_draw):

        cur_splt.set_title(cur_row + '_' + cur_col)

        cur_splt.scatter(df_pca[df_pca['last_cycle'] == 0][cur_col], df_pca[df_pca['last_cycle'] == 0][cur_row], marker='.', color='blue')

        cur_splt.scatter(df_pca[df_pca['last_cycle'] == 1][cur_col], df_pca[df_pca['last_cycle'] == 1][cur_row], marker='.', color='red')

plt.show()
from sklearn import preprocessing
df_pca.columns
std_scaler = preprocessing.StandardScaler()

df_scaled = pd.DataFrame(std_scaler.fit_transform(df_pca[df_pca.columns[2:-1]]))

df_scaled = df_scaled.rename(lambda x: df_pca.columns[2:-1][x], axis='columns')

df_scaled.insert(0, 'id', df_rolled['id'])

df_scaled.insert(1, 'cycle', df_rolled['cycle'])

df_scaled['last_cycle'] = df_rolled['last_cycle']

df_scaled.head()
columns_to_draw = ['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 'pca1', 's1', 's2']

amount_of_cols = len(columns_to_draw)

fig, splts = plt.subplots(amount_of_cols, 1, figsize=(16, amount_of_cols*10))

for cur_splt, cur_col in zip(splts, columns_to_draw):

    for cur_id in range(81):

        x_vals = df_scaled[df_scaled['id'] == cur_id]['cycle']

        y_vals = df_scaled[df_make_sense['id'] == cur_id][cur_col]

        cur_splt.set_title(cur_col)

        cur_splt.plot(x_vals, y_vals, marker='.')

plt.show()
from sklearn.model_selection import train_test_split

X = df_scaled[df_scaled.columns[2:-1]]

#X = X.drop('s1', 's2'], axis=1)

Y = df_scaled[df_scaled.columns[-1:]]

X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn import tree

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



clf = tree.DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=3, min_weight_fraction_leaf=0.001, max_depth=5)

clf = clf.fit(X_train, Y_train)

predicted = clf.predict(X_test)

print(precision_score(Y_test, predicted), recall_score(Y_test, predicted), f1_score(Y_test, predicted))

print(roc_auc_score(Y_test, predicted))
pred = clf.predict(df_scaled[['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 'pca1', 's1', 's2']])

act = df_scaled['last_cycle']

dist_cycle = []

for idx, row in df[(pred == 1) & (act == 0)].iterrows():

    dist_cycle.append(max_cycle[row['id']] - row['cycle'])

plt.boxplot(dist_cycle)

plt.show()

import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None, 

                         feature_names=X_train.columns,  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
X_test.head()
for i in np.arange(2, 3, 0.025):

    predicted = X_test['pca1']>=i

    pr = precision_score(Y_test, predicted)

    re = recall_score(Y_test, predicted)

    f1 = f1_score(Y_test, predicted)

    ra = roc_auc_score(Y_test, predicted)

    plt.scatter(i, pr, c='red', marker='.')

    plt.scatter(i, re, c='orange', marker='.')

    plt.scatter(i, f1, c='blue', marker='.')

    plt.scatter(i, ra, c='green', marker='^')

plt.show()
predicted = X_test['pca1']>=2.1

print(precision_score(Y_test, predicted), recall_score(Y_test, predicted), f1_score(Y_test, predicted))

print(roc_auc_score(Y_test, predicted))
clf = tree.DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=3, min_weight_fraction_leaf=0.001, max_depth=5)

clf = clf.fit(X_train[['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 's1', 's2']], Y_train)



predicted = clf.predict(X_test[['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 's1', 's2']])

print(precision_score(Y_test, predicted), recall_score(Y_test, predicted), f1_score(Y_test, predicted))

print(roc_auc_score(Y_test, predicted))



import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None, 

                         feature_names=['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 's1', 's2'],  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
from sklearn.metrics import roc_curve

plt.figure(figsize=(6, 6))

fpr, tpr, thr = roc_curve(Y_test, predicted)

plt.plot(fpr, tpr, label='Test data roc fold')

fpr, tpr, thr = roc_curve(Y_train, clf.predict(X_train[['p02', 'p03', 'p04', 'p05', 'p06','p08', 'p11', 'p12', 'p13', 'p14', 'p15',  'p18', 'p19', 's1', 's2']]))

plt.plot(fpr, tpr, label='Train data roc fold')

plt.plot([0,1], [0, 1], label='Random')

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")

plt.legend()
df_shifted = df_scaled[['id', 'cycle', 'pca1', 's1', 's2']]

df_shifted['pca1_sqrt'] = df_scaled['pca1'].apply(lambda x: (x + 2) ** (1/3))

for cur_col in ['pca1']:

    lag_col = df_shifted[cur_col].shift(-1)

    for i in lag_col[df_scaled['last_cycle'] == 1].index:

        lag_col.set_value(i, None)

    df_shifted[cur_col + '_next'] = lag_col

    df_shifted[cur_col + '_next_diff'] = lag_col - df_shifted[cur_col]

df_shifted['last_cycle'] = df_scaled['last_cycle']

df_shifted.head()
columns_to_draw = ['pca1', 'pca1_next', 'pca1_sqrt', 's1', 's2']

amount_of_cols = len(columns_to_draw)

fig, splts = plt.subplots(amount_of_cols, amount_of_cols, figsize=(6*amount_of_cols, amount_of_cols*6))

for cur_splt_row, cur_row in zip(splts, columns_to_draw):

    for cur_splt, cur_col in zip(cur_splt_row, columns_to_draw):

        cur_splt.set_title(cur_row + '_' + cur_col)

        cur_splt.scatter(df_shifted[df_shifted['last_cycle'] == 0][cur_col], df_shifted[df_shifted['last_cycle'] == 0][cur_row], marker='.', color='blue')

        cur_splt.scatter(df_shifted[df_shifted['last_cycle'] == 1][cur_col], df_shifted[df_shifted['last_cycle'] == 1][cur_row], marker='.', color='red')

        cur_splt.plot([-3, 3], [-3, 3])

plt.show()
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit (df_shifted[['pca1']], df_shifted['pca1_next'].fillna(0))

#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

print(reg.coef_, reg.score(df_shifted[['pca1']], df_shifted['pca1_next'].fillna(0)))
columns_to_draw = ['pca1_sqrt','pca1', 's1', 's2']

amount_of_cols = len(columns_to_draw)

fig, splts = plt.subplots(amount_of_cols, 1, figsize=(16, amount_of_cols*10))

for cur_splt, cur_col in zip(splts, columns_to_draw):

    for cur_id in range(2):

        x_vals = df_shifted[df_shifted['id'] == cur_id]['cycle']

        y_vals = df_shifted[df_shifted['id'] == cur_id][cur_col]

        cur_splt.set_title(cur_col)

        cur_splt.plot(x_vals, y_vals, marker='.')

plt.show()