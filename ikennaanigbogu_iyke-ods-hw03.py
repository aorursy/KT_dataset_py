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
df_train.head()
df_test['target'] = np.nan



df = pd.concat([df_train, df_test])



print(df.shape)
df.head()
df.tail()
df.dtypes
cats = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']



nums = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
import itertools as it
for i in range(1, 4):

    print(i)

    for g in it.combinations(cats, i):

        df = pd.concat(

            [

                df, 

                df.groupby(list(g))[nums].transform('mean').rename(

                    columns=dict([(s, ':'.join(g) + '__' + s + '__mean') for s in nums])

                )

            ], 

            axis=1

        )
df.drop(columns=cats, inplace=True)

df.shape
cols = [c for c in df.columns if c != 'uid' and c != 'target']
from sklearn.preprocessing import StandardScaler, MinMaxScaler



df[cols] = MinMaxScaler().fit_transform(df[cols])
df_m = df[cols].corr()
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
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier
'''

X=df.loc[df['target'].notna()][cols]

y=df.loc[df['target'].notna()]['target']

# range of k we want to try

k_range = range(150, 200,3)

# empty list to store scores

k_scores = []



# 1. we will loop through reasonable values of k

for k in k_range:

    print(k)

    # 2. run KNeighborsClassifier with k neighbours

    knn = KNeighborsClassifier(n_neighbors=k)

    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

    # 4. append mean of scores for k neighbors to k_scores list

    k_scores.append(scores.mean())



# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

# plt.plot(x_axis, y_axis)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-validated accuracy')

'''
'''

X=df.loc[df['target'].notna()][cols]

y=df.loc[df['target'].notna()]['target']



#List Hyperparameters to tune

leaf_size = list(range(20,50))

n_neighbors = [100,148,151,153]

p=[1,2]

weights = ['distance','uniform']

algorithm = ['auto','brute']

metric = ['minkowski']

n_jobs = [-1]

#convert to dictionary

hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p, weights=weights, 

                       algorithm=algorithm, metric=metric,n_jobs=n_jobs)



knn = KNeighborsClassifier()

gridsearch = RandomizedSearchCV(knn, hyperparameters, cv=10)

gridsearch.fit(X,y)

print("Best parameter for RandomSearch:\n")

print(gridsearch.best_params_)

'''
'''

X=df.loc[df['target'].notna()][cols]

y=df.loc[df['target'].notna()]['target']



parameters = {'criterion': ['gini','entropy'],

             'max_depth': range (1,100,3)}



dtc = DecisionTreeClassifier()

gridsearch = GridSearchCV(dtc, param_grid=parameters, cv=10)

gridsearch.fit(X,y)

print("Best parameter for GridSearch:\n")

print(gridsearch.best_params_)

'''
'''

X=df.loc[df['target'].notna()][cols]

y=df.loc[df['target'].notna()]['target']

# range of k we want to try

d_range = range(10,50,1)

# empty list to store scores

d_scores = []

# 1. we will loop through reasonable values of d

for d in d_range:

    print(d)

    # 2. run DecisionTreeClassifier with different max_depth

    dtc = DecisionTreeClassifier(criterion='entropy',max_depth=d)

    # 3. obtain cross_val_score for DecisionTreeClassifier with different max_depth

    scores = cross_val_score(dtc, X, y, cv=10, scoring='accuracy')

    # 4. append mean of scores for max_depth to d_scores list

    d_scores.append(scores.mean())



# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

# plt.plot(x_axis, y_axis)

plt.plot(d_range, d_scores)

plt.xlabel('Value of d for max_depth')

plt.ylabel('Cross-validated accuracy')

'''
knn_model = KNeighborsClassifier(

    n_neighbors=153,

    weights='distance',

    algorithm='auto',

    leaf_size=36,

    p=1,

    metric='minkowski',

    n_jobs=-1

)



dtc_model = DecisionTreeClassifier(

    criterion='entropy',

    max_depth=9,

    splitter='best',

    max_features=6,

    min_samples_split=45,

    min_samples_leaf=10)



knn_model = knn_model.fit(df.loc[df['target'].notna()][cols], df.loc[df['target'].notna()]['target'])

dtc_model = dtc_model.fit(df.loc[df['target'].notna()][cols], df.loc[df['target'].notna()]['target'])
knn_p = knn_model.predict_proba(df.loc[df['target'].isna()][cols])

dtc_p = dtc_model.predict_proba(df.loc[df['target'].isna()][cols])

finalpred=(knn_p*0.25+dtc_p*0.75)
%matplotlib inline

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")
sns.distplot(knn_p[:, 1])
sns.distplot(dtc_p[:, 1])
sns.distplot(finalpred[:, 1])
from sklearn.ensemble import VotingClassifier

vot_model = VotingClassifier(estimators=[('knn', knn_model), ('dt', dtc_model)], voting='soft')



vot_model.fit(df.loc[df['target'].notna()][cols], df.loc[df['target'].notna()]['target'])

vot_p = vot_model.predict_proba(df.loc[df['target'].isna()][cols])
sns.distplot(vot_p[:, 1])
df_submit_knn = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': knn_p[:, 1]

})



df_submit_dtc = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': dtc_p[:, 1]

})



df_submit_vot = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': vot_p[:, 1]

})



df_submit_mix = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': finalpred[:, 1]

})
df_submit_knn.to_csv('/kaggle/working/submit_knn.csv', index=False)

df_submit_dtc.to_csv('/kaggle/working/submit_dtc.csv', index=False)

df_submit_vot.to_csv('/kaggle/working/submit_vop.csv', index=False)

df_submit_mix.to_csv('/kaggle/working/submit_mix.csv', index=False)
!head /kaggle/working/submit.csv