# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)

target_train = df_train['target']
print(target_train.shape)

df_train.head()
df_test['target'] = np.nan

df = pd.concat([df_train, df_test])

df1 = df

print(df.shape)
print(df1.shape)
df.head()
df.tail()
df.dtypes
df_tmp = df.loc[
    df['target'].notna()
].groupby(
    ['education']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean', 'std': 'target_std'}
).fillna(0.0).reset_index()

df_tmp.head()
df = pd.merge(
    df,
    df_tmp,
    how='left',
    on=['education']
)

df.shape
df['target_mean'] = df['target_mean'].fillna(0.0)
df['target_std'] = df['target_std'].fillna(0.0)
pd.get_dummies(df['workclass'])
df = pd.get_dummies(
    df, 
    columns=[c for c in df_train.columns if df_train[c].dtype == 'object']
)
df.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

y = df[df['target'].notna()]['target']
X_train, X_holdout, y_train, y_holdout = train_test_split(
    df[df['target'].notna()].values, y, test_size=0.3, random_state=20)

tree = DecisionTreeClassifier(max_depth=10, random_state=20)
knn = KNeighborsClassifier(n_neighbors=10)
tree.fit(X_train, y_train)

#for knn scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
knn.fit(X_train_scaled, y_train)
from sklearn.metrics import accuracy_score
tree_pred = tree.predict(X_holdout)
accuracy_score(y_holdout, tree_pred)

knn_pred = knn.predict(X_holdout_scaled)
accuracy_score(y_holdout, knn_pred)
tree_params = {
    'max_depth': range(1, 12),
    'max_features': range(4, 20),
}
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)

tree_grid.fit(X_train, y_train)
print(tree_grid.best_params_,tree_grid.best_score_)
accuracy_score(y_holdout, tree_grid.predict(X_holdout))
from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': range(1, 15)}

knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)

knn_grid.fit(X_train, y_train)

knn_grid.best_params_, knn_grid.best_score_
accuracy_score(y_holdout, knn_grid.predict(X_holdout))
df1.head()
cats = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

nums = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
import itertools as it
for i in range(1, 4):
    print(i)
    for g in it.combinations(cats, i):
        df1 = pd.concat(
            [
                df1, 
                df1.groupby(list(g))[nums].transform('mean').rename(
                    columns=dict([(s, ':'.join(g) + '__' + s + '__mean') for s in nums])
                )
            ], 
            axis=1
        )
df1.drop(columns=cats, inplace=True)
df1.shape
cols = [c for c in df1.columns if c != 'uid' and c != 'target']
from sklearn.preprocessing import StandardScaler

df1[cols] = StandardScaler().fit_transform(df1[cols])
df_m = df1[cols].corr()
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
cols

model_knn = KNeighborsClassifier(
    n_neighbors=100,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1
)

model_knn = model_knn.fit(df1.loc[df1['target'].notna()][cols], df1.loc[df1['target'].notna()]['target'])
p_knn = model_knn.predict_proba(df1.loc[df1['target'].isna()][cols])
p_train_knn = model_knn.predict_proba(df1.loc[df1['target'].notna()][cols])
print(p_train_knn)
#Finally
model_tree = DecisionTreeClassifier(criterion='gini',
                               splitter='best',
                               max_depth=11,
                               max_features= 19,
                               min_samples_split=42,
                               min_samples_leaf=17)

model_tree = model_tree.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])
p_tree = model_tree.predict_proba(
    df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
p_train_tree = model_tree.predict_proba(df.loc[df['target'].notna()].drop(columns=['target']))[:, 1]
model_tree.predict_proba(
    df.loc[df['target'].isna()].drop(columns=['target']).head())
%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

df_train_log = pd.DataFrame({
    'uid': df.loc[df['target'].notna()]['uid'],
    'DT': p_train_tree
    })
df_test_log = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'DT': p_tree
    })
df_train_log1 = pd.DataFrame({
    'uid': df.loc[df['target'].notna()]['uid'],
    'KNN': p_train_knn[:, 1]
    })
df_test_log1 = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'KNN': p_knn[:, 1]
    })
df_train_log1.head()
df_train_log.head()
from sklearn.metrics import log_loss
log_tree = log_loss(target_train,df_train_log['DT'])
print('log_loss1=', log_tree)

log_knn = log_loss(target_train,df_train_log1['KNN'])
print('log_loss2=', log_knn)
df_train_log['target'] = df_train_log['DT']*0.7 + df_train_log1['KNN']*0.3
df_train_log.head()
log_knn_dt = log_loss(target_train,df_train_log['target'])
print('log_loss2=', log_knn_dt)
df_test_log['target'] = df_test_log['DT']*0.7 + df_test_log1['KNN']*0.3
df_test_log.head()

sns.distplot(p)
df_final = df_test_log.drop(columns=['DT'])
df_final.head()
print(df_final.shape)
df_final.to_csv('/kaggle/working/submit.csv', index=False)
!head /kaggle/working/submit.csv



