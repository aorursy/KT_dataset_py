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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set();
df= pd.read_csv('../input/bank-marketing-dataset/bank.csv')
df.head(10)
df.info()
df.describe().T
df['deposit'].value_counts(normalize=True)
# df['deposit'].hist()
df['deposit'].value_counts().plot(kind='bar', label='deposit')

plt.ylabel('count')

plt.xlabel('deposit')
df['education'].value_counts(normalize=True)
df['job'].value_counts(normalize=True)
df['poutcome'].value_counts(normalize=True)
sns.boxplot(df['balance'])
df['balance'].hist(bins=20);
sns.boxplot(df['duration'])
df['duration'].hist(bins=30);
def outliers_indices(feature):

    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_dur=outliers_indices('duration')

wrong_bal=outliers_indices('balance')

out=set(wrong_bal|wrong_dur)

len(out)
df['deposit'].value_counts(normalize=True)
df['deposit'].hist()
df.drop(out, inplace=True)
sns.pairplot(df);
df['deposit']=df['deposit'].map({'no': 0,'yes': 1})

#Можно трактовать и как ранговый (значит, можно вычислять, например, коэфициент Спирмена) и как категориальный

# df['deposit']
sns.heatmap(df.corr(method='spearman'));
plt.figure(figsize=(25, 250))

sns.countplot(y='duration', hue='deposit', data=df);
df.groupby('deposit')['duration'].mean()
df.groupby('deposit')['balance'].mean().plot(kind='bar')

plt.ylabel('balance')
df.groupby('deposit')['balance'].mean()
from scipy.stats import pointbiserialr

pointbiserialr(df['balance'],df['deposit'])
pointbiserialr(df['duration'],df['deposit'])
sns.countplot(y='previous', hue='deposit', data=df[(df['previous']<5)]);
plt.figure(figsize=(15, 15))

sns.countplot(y='previous', hue='deposit', data=df[(df['previous']>5)]);
sns.countplot(y='pdays', hue='deposit', data=df[df['pdays']<0]);
plt.figure(figsize=(15, 100))

sns.countplot(y='pdays', hue='deposit', data=df[df['pdays']>0]);
# df[(df['deposit']==1)]['month'].value_counts()

df['deposit']
pointbiserialr(df['previous'],df['deposit'])
pointbiserialr(df['pdays'],df['deposit'])
df['duration']=df['duration']/60
df['default']=df['default'].map({'no':0,'yes':1})

df['housing']=df['housing'].map({'no':0,'yes':1})

df['loan']=df['loan'].map({'no':0,'yes':1})

df.info()
# dummy df

ddf = pd.get_dummies(df, columns=['job', 'education', 'marital', 'contact', 'poutcome', 'month'])

ddf.info()
ddf.head().T
from sklearn.model_selection import train_test_split

X=ddf.drop('deposit',axis=1)

y=ddf['deposit']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)
bank_dataframe = pd.DataFrame(X_train, columns=X.columns)

grr = pd.plotting.scatter_matrix(bank_dataframe, 

                                 c=y_train, 

                                 figsize=(200, 200), 

                                 marker='o',

                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_valid, y_valid)
y_pred = knn.predict(X_valid)

y_pred
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

# Поскольку итоговый результат однозначен, данная метрика качества вполне подходит

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51),'metric':['minkowski','chebyshev']}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
pd.DataFrame(knn_grid.cv_results_).T
knn_grid.score(X_valid, y_valid)
y_pred = knn_grid.predict(X_valid)

accuracy_score(y_valid, y_pred)
best_knn = KNeighborsClassifier(n_neighbors=9)

y_pred = best_knn.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
results_df = pd.DataFrame(knn_grid.cv_results_)

plt.plot(results_df[(results_df['param_metric']=='minkowski')]['param_n_neighbors'], results_df[(results_df['param_metric']=='minkowski')]['mean_test_score'])

plt.xlabel('n_neighbors')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
knn = KNeighborsClassifier(n_neighbors=9,weights='distance')

knn_params = {'p': np.linspace(1, 10,num=200)}

knn_grid = GridSearchCV(knn,

                        knn_params,

                        scoring='accuracy',

                        cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
knn_grid.score(X_valid, y_valid)
y_pred = knn_grid.predict(X_valid)

accuracy_score(y_valid, y_pred)
best_knn = KNeighborsClassifier(n_neighbors=13,metric='manhattan',weights='distance')

#При манхэттенской метрике число соседей 13 более эффективно

y_pred = best_knn.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
scores = cross_val_score(best_knn, X, y, 

                         cv=kf, scoring='accuracy')

scores.mean()
from sklearn.neighbors import RadiusNeighborsClassifier

rnc=RadiusNeighborsClassifier(weights='distance')

rnc_params = {'radius': np.linspace(500, 5000,num=50)}

rnc_grid = GridSearchCV(rnc,

                        rnc_params,

                        scoring='accuracy',

                        cv=kf)

rnc_grid.fit(X_train, y_train)
rnc_grid.best_estimator_

rnc_grid.best_score_

rnc_grid.best_params_

# rnc_grid.score(X_valid, y_valid)

from sklearn.neighbors import RadiusNeighborsClassifier

rnc=RadiusNeighborsClassifier(weights='distance', radius=1000)

rnc_params = {'p': np.linspace(1, 10,num=10)}

rnc_grid = GridSearchCV(rnc,

                        rnc_params,

                        scoring='accuracy',

                        cv=kf)

rnc_grid.fit(X_train, y_train)
rnc_grid.best_estimator_

rnc_grid.best_score_

rnc_grid.best_params_

pd.DataFrame(rnc_grid.cv_results_).T
# rnc_grid.score(X_valid, y_valid)
from sklearn.neighbors import NearestCentroid

nc=NearestCentroid()

nc_params = {'metric':['manhattan','chebyshev']}

nc_grid = GridSearchCV(nc,

                        nc_params,

                        scoring='accuracy',

                        cv=kf)

nc_grid.fit(X_train, y_train)
nc_grid.best_estimator_

nc_grid.best_score_

nc_grid.best_params_

pd.DataFrame(nc_grid.cv_results_).T

nc_grid.score(X_valid, y_valid)

best_nc = NearestCentroid(metric='chebyshev')

y_pred = best_nc.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)