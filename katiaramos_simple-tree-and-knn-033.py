# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()



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
df_train.head()
df_test['target'] = np.nan



df_test.head()
df = pd.concat([df_train, df_test])

df.head()

#print(df.shape)
# Save one copy of the data to test in KNN model

df_knn = df
#df.dtypes
#Data analyzing

Numeric_features = [

    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',

    'hours-per-week', 'target'

]



Categorical_features = [

    'workclass', 'education', 'marital-status', 'occupation', 'relationshop',

    'race', 'sex', 'native-country'

]
#sns.countplot(df['target'], label="Count")
#sns.countplot(df['race'], label="Count")
married = [i for i in df['marital-status'].unique() if i[:8] == ' Married']

alone = [i for i in df['marital-status'].unique() if i not in married]



df['marital-status'] = df['marital-status'].replace(married, 1)

df['marital-status'] = df['marital-status'].replace(alone, 0)
df_tmp = df.loc[df['target'].notna()].groupby(['education'])['target'].agg(

    ['mean','std']).rename(columns={

    'mean': 'target_mean',    

    'std': 'target_std'

    }).fillna(0.0).reset_index()
df = pd.merge(df, df_tmp, how='left', on=['education'])
df['target_std'] = df['target_std'].fillna(0.0)

df['target_mean'] = df['target_mean'].fillna(0.0)
df.head()
#df['race'].unique()
#white = [i for i in df['race'].unique() if i[:5] == 'White']

#other = [i for i in df['race'].unique() if i not in white]



#df['race'] = df['race'].replace(white, 1)

#df['race'] = df['race'].replace(other, 0)



#df.head()
#Check correlation

plt.figure(figsize=(12, 4))

sns.heatmap(df[Numeric_features].corr(), annot=True, cmap='Blues')
df = pd.get_dummies(

    df, 

    columns=[c for c in df_train.columns if df_train[c].dtype == 'object']

)
df.head()
# dt - train

dt_x_train = df.loc[df['target'].notna()].drop(columns=['target'])

dt_y_train = df.loc[df['target'].notna()]['target']



# dt - test

dt_x_test = df.loc[df['target'].isna()].drop(columns=['target'])
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold

from sklearn.metrics import fbeta_score, make_scorer, log_loss, accuracy_score



model_dt = DecisionTreeClassifier(criterion='gini',splitter='best',random_state=42)

parameter_grid = {

    'max_depth': range(5, 15),

    'max_features': range(1, 9),

    'min_samples_split': [35, 40, 45, 50],

    'min_samples_leaf': [5, 10, 15, 20],

}

LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)



grid_search = GridSearchCV(model_dt, param_grid=parameter_grid, cv=5, scoring=LogLoss)



grid_search.fit(dt_x_train, dt_y_train)
print(f'Best score: {grid_search.best_score_}')

print(f'Best parameters: {grid_search.best_params_}')
model_dt = DecisionTreeClassifier(

    criterion='gini',

    splitter='best',

    random_state=42,

    max_depth=grid_search.best_params_['max_depth'],

    max_features=grid_search.best_params_['max_features'],

    min_samples_split=grid_search.best_params_['min_samples_split'],

    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],

)



model_dt = model_dt.fit(dt_x_train, dt_y_train)



p_dt = model_dt.predict_proba(dt_x_test)
cats = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']



nums = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
import itertools as it
for i in range(1, 4):

    print(i)

    for g in it.combinations(cats, i):

        df_knn = pd.concat(

            [

                df_knn, 

                df_knn.groupby(list(g))[nums].transform('mean').rename(

                    columns=dict([(s, ':'.join(g) + '__' + s + '__mean') for s in nums])

                )

            ], 

            axis=1

        )
df_knn.drop(columns=cats, inplace=True)

df_knn.shape
cols = [c for c in df_knn.columns if c != 'uid' and c != 'target']
from sklearn.preprocessing import StandardScaler



df_knn[cols] = StandardScaler().fit_transform(df_knn[cols])
df_m = df_knn[cols].corr()
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
# knn - train

knn_x_train = df_knn.loc[df_knn['target'].notna()][cols]

knn_y_train = df_knn.loc[df_knn['target'].notna()]['target']



# knn - test

knn_x_test = df_knn.loc[df_knn['target'].isna()][cols]
from sklearn.neighbors import KNeighborsClassifier



model_knn = KNeighborsClassifier(

    n_neighbors=100,

    weights='distance',

    algorithm='auto',

    leaf_size=30,

    p=2,

    metric='minkowski',

    n_jobs=-1

)



# Fit

model_knn = model_knn.fit(knn_x_train, knn_y_train)



# Predict

p_knn = model_knn.predict_proba(knn_x_test)
p_knn
sns.distplot(p_knn[:, 1])
sns.distplot(p_dt[:, 1])
w_dt=.75

p = w_dt*p_dt[:, 1] + (1-w_dt)*p_knn[:, 1]
p
%matplotlib inline

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")
sns.distplot(p)
df_submit = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': p

})
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
#!head /kaggle/working/submit.csv