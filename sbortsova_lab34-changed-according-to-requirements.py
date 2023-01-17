import pandas as pd

from os.path import join as pjoin

import os

import warnings

import numpy as np

from scipy  import sparse

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

import sklearn.linear_model as lm

warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

#графики в svg выглядят более четкими

%config InlineBackend.figure_format = 'svg' 



#увеличим дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = (5, 4)

rcParams["xtick.labelsize"] = 7

import pandas as pd

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

os.getcwd()

os.listdir("../input/")
PATH_TO_DATA = '../input/lab34classificationtable/'
df_train = pd.read_csv(pjoin(PATH_TO_DATA, 'train.csv'))

df_train.head()
df_test = pd.read_csv(pjoin(PATH_TO_DATA, 'test.csv'))

df_test.head()
df_train.y.hist()
df_train['marital'].value_counts().plot(kind='bar', label='marital')

plt.legend()
features = list(set(df_train.columns))



df_train[features].hist(figsize=(12,6));
_, axes = plt.subplots(1, 2, sharey=True, figsize=(12,6))



sns.countplot(x='marital', hue='y', data=df_train, ax=axes[0]);

chart = sns.countplot(x='education', hue='y', data=df_train, ax=axes[1]);

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
_, axes = plt.subplots(1, 2, sharey=True, figsize=(12,6))



sns.countplot(x='contact', hue='y', data=df_train, ax=axes[0]);

chart = sns.countplot(x='job', hue='y', data=df_train, ax=axes[1]);

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
_, axes = plt.subplots(1, 2, sharey=True, figsize=(12,6))



sns.countplot(x='housing', hue='y', data=df_train, ax=axes[0]);

sns.countplot(x='loan', hue='y', data=df_train, ax=axes[1]);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(12,6))



sns.countplot(x='previous', hue='y', data=df_train, ax=axes[0]);

sns.countplot(x='campaign', hue='y', data=df_train, ax=axes[1]);
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix);
df_train.y.unique()
df_train['contact_date'].astype('datetime64[ns]')

df_train['birth_date'].astype('datetime64[ns]')
df_test['contact_date'].astype('datetime64[ns]')

df_test['birth_date'].astype('datetime64[ns]')
df_train ['age'] = (pd.to_datetime(df_train['contact_date'])- pd.to_datetime(df_train['birth_date']))

df_train['age'].astype('int64')

df_train['pdays'].astype('int64')
df_test ['age'] = (pd.to_datetime(df['contact_date'])- pd.to_datetime(df['birth_date']))

df_test['age'].astype('int64')

df_test['pdays'].astype('int64')
categorical_columns = [c for c in df_train.columns if df_train[c].dtype.name == 'object']

numerical_columns   = [c for c in df_train.columns if df_train[c].dtype.name != 'object']

numerical_columns.remove('Unnamed: 0')

categorical_columns

numerical_columns
df_train[categorical_columns].describe()
for c in categorical_columns:

    print(c, df_train[c].unique())
data_describe = df_train.describe(include=[object])

binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]

nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]

print("Бинарные", binary_columns)

print("Небинарные", nonbinary_columns)
df_train.at[df_train['contact'] == 'cellular', 'contact'] = 0

df_train.at[df_train['contact'] == 'telephone', 'contact'] = 1

df_train['contact'].describe()
df_test.at[df_test['contact'] == 'cellular', 'contact'] = 0

df_test.at[df_test['contact'] == 'telephone', 'contact'] = 1

df_test['contact'].describe()
nonbinary_columns.remove('contact_date')

nonbinary_columns.remove('birth_date')

nonbinary_columns.remove('default')

numerical_columns.remove('y')
data_nonbinary = pd.get_dummies(df_train[nonbinary_columns])

print(nonbinary_columns)

print(data_nonbinary.columns)
data_nonbinary_test = pd.get_dummies(df_test[nonbinary_columns])
data_numerical = df_train[numerical_columns]

data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()

data_numerical.describe()
data_numerical_test = df_test[numerical_columns]

data_numerical_test = (data_numerical_test - data_numerical_test.mean()) / data_numerical_test.std()

data_numerical_test.describe()
data_train = pd.concat((data_numerical, df_train[binary_columns], data_nonbinary), axis=1)

data_train = pd.DataFrame(data_train, dtype=float)

print(data_train.shape)

print(data_train.columns)
data_test = pd.concat((data_numerical_test, df_test[binary_columns], data_nonbinary_test), axis=1)

data_test = pd.DataFrame(data_test, dtype=float)

print(data_train.shape)

print(data_train.columns)
y_train = df_train['y']

X_train=data_train

feature_names = X_train.columns

print('features_name', feature_names)
X_test=data_test

feature_names = X_test.columns

print('features_name', feature_names)
X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_train, y_train, test_size = 0.3, random_state = 11)



N_train, _ = X_train_drop.shape 

N_test,  _ = X_test_drop.shape 

print(N_train, N_test)
linear = lm.LinearRegression()



linear.fit(X_train_drop, y_train_drop)

linear.score(X_train_drop, y_train_drop)

print('Coefficient: \n', linear.coef_)

print('Intercept: \n', linear.intercept_)

print('R² Value: \n', linear.score(X_train_drop, y_train_drop))
logistic = lm.LogisticRegression()

logistic.fit(X_train_drop, y_train_drop)

logistic.score(X_train_drop, y_train_drop)

print('Coefficient: \n', logistic.coef_)

print('Intercept: \n', logistic.intercept_)

print('R² Value: \n', logistic.score(X_train_drop, y_train_drop))
n_neighbors_array = [1, 3, 5, 7, 10, 15]

knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})

grid.fit(X_train_drop, y_train_drop)



best_cv_err = 1 - grid.best_score_

best_n_neighbors = grid.best_estimator_.n_neighbors

print('Лучшая ошибка', best_cv_err, 'для лучшего количества соседей', best_n_neighbors)
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)

knn.fit(X_train_drop, y_train_drop)

print('Accuracy: \n', knn.score(X_train_drop, y_train_drop))
from sklearn import ensemble

rf = ensemble.RandomForestClassifier(n_estimators=70, random_state=11)

rf.fit(X_train_drop, y_train_drop)

print('Accuracy: \n', rf.score(X_train_drop, y_train_drop))
importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]



print("Feature importances:")

for f, idx in enumerate(indices):

    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
d_first = 10

plt.figure(figsize=(8, 8))

plt.title("Feature importances")

plt.bar(range(d_first), importances[indices[:d_first]], align='center')

plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)

plt.xlim([-1, d_first]);
best_features = indices[:10]

best_features_names = feature_names[best_features]

print(best_features_names)
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)

gbt.fit(X_train_drop[best_features_names], y_train_drop)

print('Accuracy: \n', gbt.score(X_train_drop[best_features_names], y_train_drop))
models = [logistic, knn, rf, gbt]
from sklearn.metrics import roc_curve, auc

import pylab as pl

plt.figure(figsize=(10, 10)) 

for model in models:

    if model==gbt:

        pred_scr = model.predict_proba(X_train_drop[best_features_names])

    else:

        pred_scr = model.predict_proba(X_train_drop)

    fpr, tpr, thresholds = roc_curve(y_train_drop, pred_scr[:, 1])

    roc_auc  = auc(fpr, tpr)

    pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (str(model), roc_auc))

    



pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

pl.xlim([0, 1])

pl.ylim([0, 1])

pl.xlabel('False Positive Rate')

pl.ylabel('True Positive Rate')

pl.title('Receiver operating characteristic example')

pl.legend(loc="lower right")

pl.show()
rfc = ensemble.RandomForestClassifier(n_estimators=70, random_state=11)

rfc.fit(X_train, y_train)



res = rfc.predict_proba(X_test)[:, 1]

result = pd.DataFrame({'id': df_test.index, 'y': res})

print('result Y', res, result)

result.to_csv('res_baseline.csv', index=False)