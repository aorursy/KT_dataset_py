import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, scale, PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFECV, chi2, SelectKBest
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
%matplotlib inline
sklearn.utils.check_random_state(42)
train = pd.read_csv('../input/train_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')

test = pd.read_csv('../input/test_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')
train.info()
train.head()
train.describe()
train.isnull().any(axis=0)
pd.get_dummies(train['workclass'])
for col in train.columns:
    if train[col].dtype.name == 'object':
        train[col].value_counts().plot(title=col, kind='pie', label='')
        plt.show()
    else:
        train[col].hist(bins=15).set_title(col)
        plt.show()
print('Zero value frequencies:')
print('capital.gain = {0:.2f}%'.format(sum(train['capital.gain'] == 0) / train.shape[0] * 100))
print('capital.loss = {0:.2f}%'.format(sum(train['capital.loss'] == 0) / train.shape[0] * 100))
all_features = list(train.columns)
all_features.remove('Id')
all_features.remove('income')
numeric_features = [f for f in all_features if train[f].dtype != 'object']
train[numeric_features] = train[numeric_features].astype('float64')
from scipy.stats import pearsonr

# Convert 'income' to numeric (True=1, False=0)
income_num = train['income'] == '>50K'

# print('Correlações e p-values entre algumas features e income:')
# for k in ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']:
#     print(k, pearsonr(train[k], income_num))
    
def dict_stats(a):
    corr, pval = pearsonr(a, income_num)
    fcorr, fpval = pearsonr(a >= a.mean(), income_num)
    return {'correlation [x]': corr, 'p-value [x]': pval, 'stdev [x]': a.std(),
            'correlation [f(x)]': fcorr, 'p-value [f(x)]': fpval, 'stdev [f(x)]': (a >= a.mean()).std()}
stats = train[numeric_features].apply(dict_stats, result_type='expand').transpose()
stats[['correlation [x]', 'correlation [f(x)]']].plot(kind='bar', title='Correlations', figsize=(15, 5))
stats[['p-value [x]', 'p-value [f(x)]']].plot(kind='bar', title='P-values (logarithmic scale)', figsize=(15, 8))
stats[['stdev [x]', 'stdev [f(x)]']].plot(kind='bar', title='Standard deviations (logarithmic scale)', figsize=(15, 5))
plt.yscale('log')
from sklearn.ensemble import ExtraTreesClassifier

# Preprocess the data (standardize, one-hot encode, add new features)
features = all_features.copy()
features.remove('fnlwgt')
features.remove('education')
xtrain_post = train.copy()[features]

xtrain_post = pd.get_dummies(xtrain_post)
for feat in numeric_features:
    if feat in features:
        xtrain_post[feat] = scale(xtrain_post[[feat]])
        xtrain_post[feat + '_f'] = xtrain_post[feat] >= xtrain_post[feat].mean()

et = ExtraTreesClassifier(n_estimators=10)
rfecv = RFECV(estimator=et, step=3, scoring='accuracy', cv=3, n_jobs=-1)
rfecv.fit(xtrain_post, train['income'])
xtrain_post.columns[rfecv.support_].tolist()
base_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
def preprocess(data):
    ds = data.copy()[base_features]
    ds['education.num_f'] = ds['education.num'] >= data['education.num'].mean()
    ds['capital.gain_f'] = ds['capital.gain'] >= data['capital.gain'].mean()
    ds['marital.status_Married-civ-spouse'] = data['marital.status'] == 'Married-civ-spouse'
    ds['relationship_Husband'] = data['relationship'] == 'Husband'
    return ds.astype('float64')

xtrain = preprocess(train)
ytrain = train['income']
xtest = preprocess(test)
xtrain.head()
# First test
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
gs = GridSearchCV(rfc, {
    'n_estimators': [10, 25, 40, 55],
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 5, 9, 13, 17],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
gs = GridSearchCV(rfc, {
    'n_estimators': [20, 25, 30],
    'criterion': ['gini', 'entropy'],
    'max_depth': [11, 12, 13, 14, 15],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
rfc = RandomForestClassifier(n_estimators=25, criterion='gini', max_depth=11)
cross_val_score(rfc, xtrain, ytrain, cv=5).mean()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ab = AdaBoostClassifier(DecisionTreeClassifier())
gs = GridSearchCV(ab, {
    'n_estimators': [20, 35, 50, 65],
    'base_estimator__max_depth': [1, 2, 3],
    'base_estimator__criterion': ['entropy', 'gini'],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
gs = GridSearchCV(ab, {
    'n_estimators': [30, 35, 40],
    'base_estimator__max_depth': [2, 3, 4],
    'base_estimator__criterion': ['entropy'],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, criterion='entropy'), n_estimators=30)
cross_val_score(ab, xtrain, ytrain, cv=5).mean()
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([
    ('pt', PowerTransformer()),
    ('ss', StandardScaler()),
    ('lr', LogisticRegression(dual=False, solver='liblinear')),
])
gs = GridSearchCV(pipe, {
    'lr__penalty': ['l1', 'l2'],
    'lr__C': [.5, 1, 1.5],
    'lr__intercept_scaling': [.5, 1, 1.5],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
gs = GridSearchCV(pipe, {
    'lr__penalty': ['l1'],
    'lr__C': [.25, .5, .75],
    'lr__intercept_scaling': [.25, .5, .75],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
pipe = Pipeline([
    ('pt', PowerTransformer()),
    ('ss', StandardScaler()),
    ('lr', LogisticRegression(C=.5, intercept_scaling=.5, penalty='l1', dual=False, solver='liblinear')),
])
cross_val_score(pipe, xtrain, ytrain, cv=5, n_jobs=-1).mean()
from sklearn.svm import SVC
pipe = Pipeline([
    ('pt', PowerTransformer()),
    ('ss', StandardScaler()),
    ('svc', SVC()),
])
gs = GridSearchCV(pipe, {
    'svc__C': [.7, 1, 1.3],
    'svc__kernel': ['poly', 'rbf', 'sigmoid'],
}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
pipe = Pipeline([
    ('pt', PowerTransformer()),
    ('ss', StandardScaler()),
    ('svc', SVC(C=1, kernel='rbf')),
])
cross_val_score(pipe, xtrain, ytrain, cv=5, n_jobs=-1).mean()
from sklearn.neighbors import KNeighborsClassifier

gs = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [5, 10, 15, 20, 25, 30, 35]}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
gs = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [13, 14, 15, 16, 17]}, cv=3, n_jobs=-1)
gs.fit(xtrain, ytrain)
gs.best_params_
cross_val_score(KNeighborsClassifier(16), xtrain, ytrain, cv=5, n_jobs=-1).mean()
def make_submission(model, out):
    model.fit(xtrain, ytrain)
    ytest = model.predict(xtest)
    df=pd.DataFrame({'Id': test['Id'], 'income': ytest})
    df.to_csv(out, index=False)
    print('{0}: accuracy={1}'.format(out, cross_val_score(model, xtrain, ytrain, cv=5, n_jobs=-1).mean()))

    
rfc = RandomForestClassifier(n_estimators=25, criterion='gini', max_depth=11)
make_submission(rfc, 'sub_rfc.csv')

ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, criterion='entropy'), n_estimators=30)
make_submission(ab, 'sub_abc.csv')

pipe_lr = Pipeline([
    ('pt', PowerTransformer()),
    ('ss', StandardScaler()),
    ('lr', LogisticRegression(C=.5, intercept_scaling=.5, penalty='l1', dual=False, solver='liblinear')),
])
make_submission(pipe_lr, 'sub_lr.csv')

pipe_svc = Pipeline([
    ('pt', PowerTransformer()),
    ('ss', StandardScaler()),
    ('svc', SVC(C=1, kernel='rbf')),
])
make_submission(pipe_svc, 'sub_svc.csv')

knn = KNeighborsClassifier(16)
make_submission(knn, 'sub_knn.csv')
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier([
    ('rfc', rfc),
    ('abc', ab),
    ('lr', pipe_lr),
    ('svc', pipe_svc),
    ('knn', knn),
])
make_submission(vc, 'sub_vc.csv')