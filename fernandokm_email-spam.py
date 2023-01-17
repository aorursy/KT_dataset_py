import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, StandardScaler
from sklearn.metrics import fbeta_score, make_scorer, roc_curve
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
%matplotlib inline

from IPython import display
import os

train = pd.read_csv('../input/spamemail/train_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')

test = pd.read_csv('../input/spamemail/test_features.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')
train.info()
train.head()
train.describe()
all_features = list(train.columns)
all_features.remove('Id')
all_features.remove('ham')

# Convert all features to float64 in order to avoid warnings about type conversion:
train[all_features] = train[all_features].astype('float64')
test[all_features] = test[all_features].astype('float64')
def dict_pearsonr(a):
    corr, pval = pearsonr(a, train['ham'])
    fcorr, fpval = pearsonr(a >= a.mean(), train['ham'])
    return {'correlation [x]': corr, 'p-value [x]': pval, 'stdev [x]': a.std(),
            'correlation [f(x)]': fcorr, 'p-value [f(x)]': fpval, 'stdev [f(x)]': (a >= a.mean()).std()}
stats = train[all_features].apply(dict_pearsonr, result_type='expand').transpose()
stats[['correlation [x]', 'correlation [f(x)]']].plot(kind='bar', title='Correlations', figsize=(15, 5))
stats[['p-value [x]', 'p-value [f(x)]']].plot(kind='bar', title='P-values', figsize=(15, 8))
stats[['stdev [x]', 'stdev [f(x)]']].plot(kind='bar', title='Standard deviations (logarithmic scale)', figsize=(15, 5))
plt.yscale('log')
fbeta3 = make_scorer(fbeta_score, beta=3)
train_f = (train[all_features] >= train[all_features].mean()).astype('float64')
features = [f for f in all_features if stats['p-value [x]'][f] <= .01]
features_f = [f for f in all_features if stats['p-value [f(x)]'][f] <= .01]
print('Removed from features:', set(all_features) - set(features))
print('Removed from features_f:', set(all_features) - set(features_f))
knn = KNeighborsClassifier()
gs = GridSearchCV(knn, {'p': [1, 2], 'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train[features], train['ham'])
print(f'without f   score={gs.best_score_:.3f}  k={gs.best_params_["n_neighbors"]}  p={gs.best_params_["p"]}')
      
gs.fit(train_f[features], train['ham'])
print(f'with f      score={gs.best_score_:.3f}  k={gs.best_params_["n_neighbors"]}  p={gs.best_params_["p"]}')
gs = GridSearchCV(knn, {'p': [1], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train[features], train['ham'])
print(f'without f   score={gs.best_score_:.3f}  k={gs.best_params_["n_neighbors"]}  p={gs.best_params_["p"]}')

gs = GridSearchCV(knn, {'p': [1], 'n_neighbors': [38, 39, 40, 41, 42, 43]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train_f[features], train['ham'])
print(f'with f      score={gs.best_score_:.3f}  k={gs.best_params_["n_neighbors"]}  p={gs.best_params_["p"]}')
pipe = Pipeline([
    ('pt', PowerTransformer()),
    ('nb', GaussianNB()),
])
score = cross_val_score(pipe, train[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'gaussian+power without f   score={score:.3f}')
score = cross_val_score(pipe, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'gaussian+power with f      score={score:.3f}')

nb = GaussianNB()
score = cross_val_score(nb, train[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'gaussian       without f   score={score:.3f}')
score = cross_val_score(nb, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'gaussian       with f      score={score:.3f}')

nb = BernoulliNB()
score = cross_val_score(nb, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'bernoulli      with f      score={score:.3f}')

nb = MultinomialNB()
score = cross_val_score(nb, train[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'multinomial    without f   score={score:.3f}')
score = cross_val_score(nb, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'multinomial    with f      score={score:.3f}')

nb = ComplementNB()
score = cross_val_score(nb, train[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'complement     without f   score={score:.3f}')
score = cross_val_score(nb, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'complement     with f      score={score:.3f}')
pipe = Pipeline([
    ('ss', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier())
])

gs = GridSearchCV(pipe, {'knn__p': [1, 2], 'knn__n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train[features], train['ham'])
print(f'without f   score={gs.best_score_:.3f}  k={gs.best_params_["knn__n_neighbors"]}  p={gs.best_params_["knn__p"]}')

gs.fit(train_f[features], train['ham'])
print(f'with f      score={gs.best_score_:.3f}  k={gs.best_params_["knn__n_neighbors"]}  p={gs.best_params_["knn__p"]}')
gs = GridSearchCV(pipe, {'knn__p': [1], 'knn__n_neighbors': [33, 34, 35, 36, 37]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train[features], train['ham'])
print(f'without f   score={gs.best_score_:.3f}  k={gs.best_params_["knn__n_neighbors"]}  p={gs.best_params_["knn__p"]}')

gs = GridSearchCV(pipe, {'knn__p': [1], 'knn__n_neighbors': [43, 44, 45, 46, 47]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train_f[features], train['ham'])
print(f'with f      score={gs.best_score_:.3f}  k={gs.best_params_["knn__n_neighbors"]}  p={gs.best_params_["knn__p"]}')
pipe = Pipeline([
    ('ss', StandardScaler(with_mean=False)),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier()),
])

gs = GridSearchCV(pipe, {'knn__p': [1, 2], 'knn__n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}, scoring=fbeta3, cv=3, n_jobs=-1)
gs.fit(train[features], train['ham'])
print(f'without f   score={gs.best_score_:.3f}  k={gs.best_params_["knn__n_neighbors"]}  p={gs.best_params_["knn__p"]}')

gs.fit(train_f[features], train['ham'])
print(f'with f      score={gs.best_score_:.3f}  k={gs.best_params_["knn__n_neighbors"]}  p={gs.best_params_["knn__p"]}')
pipe = Pipeline([
    ('ss', StandardScaler(with_mean=False)),
    ('pca', PCA()),
    ('nb', GaussianNB()),
])

score = cross_val_score(pipe, train[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'without f   score={score:.3f}')

score = cross_val_score(pipe, train[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
print(f'with f      score={score:.3f}')
pipe = Pipeline([
    ('bin', KBinsDiscretizer(encode='ordinal')),
    ('nb', MultinomialNB()),
])

gs = GridSearchCV(pipe, {'bin__n_bins': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'bin__strategy': ['uniform', 'quantile']}, scoring=fbeta3, cv=3, n_jobs=-1)

gs.fit(train[features], train['ham'])
print(f'without f   score={gs.best_score_:.3f}  n_bins={gs.best_params_["bin__n_bins"]}  strategy={gs.best_params_["bin__strategy"]}')

gs.fit(train_f[features], train['ham'])
print(f'with f      score={gs.best_score_:.3f}  n_bins={gs.best_params_["bin__n_bins"]}  strategy={gs.best_params_["bin__strategy"]}')
for n in [5, 10, 15, 20, 25, 30]:
    best_features = sorted(features, key=lambda f: stats['correlation [x]'][f], reverse=True)[:n]
    best_features_f = sorted(features, key=lambda f: stats['correlation [f(x)]'][f], reverse=True)[:n]
    nb = BernoulliNB()
    score = cross_val_score(nb, train_f[best_features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'bernoulli   with f      score={score:.3f}  n={n}')

    nb = MultinomialNB()
    score = cross_val_score(nb, train[best_features_f], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'multinomial without f   score={score:.3f}  n={n}')
    score = cross_val_score(nb, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'multinomial with f      score={score:.3f}  n={n}')
for n in [5, 10, 15, 20, 25, 30]:
    best_features = sorted(features, key=lambda f: stats['correlation [x]'][f], reverse=True)[:n]
    best_features_f = sorted(features, key=lambda f: stats['correlation [f(x)]'][f], reverse=True)[:n]
    knn = KNeighborsClassifier(n_neighbors=35, p=1, n_jobs=-1)
    score = cross_val_score(knn, train[best_features_f], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'knn without f   score={score:.3f}  k=35  p=1  n={n}')
    
    knn = KNeighborsClassifier(n_neighbors=39, p=1, n_jobs=-1)
    score = cross_val_score(knn, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'knn with f      score={score:.3f}  k=35  p=1  n={n}')
for n in [5, 10, 15, 20, 25, 30]:
    best_features = sorted(features, key=lambda f: stats['p-value [x]'][f])[:n]
    best_features_f = sorted(features, key=lambda f: stats['p-value [f(x)]'][f])[:n]
    nb = BernoulliNB()
    score = cross_val_score(nb, train_f[best_features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'bernoulli   with f      score={score:.3f}  n={n}')

    nb = MultinomialNB()
    score = cross_val_score(nb, train[best_features_f], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'multinomial without f   score={score:.3f}  n={n}')
    score = cross_val_score(nb, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'multinomial with f      score={score:.3f}  n={n}')
for n in [5, 10, 15, 20, 25, 30]:
    best_features = sorted(features, key=lambda f: stats['p-value [x]'][f])[:n]
    best_features_f = sorted(features, key=lambda f: stats['p-value [f(x)]'][f])[:n]
    knn = KNeighborsClassifier(n_neighbors=35, p=1, n_jobs=-1)
    score = cross_val_score(knn, train[best_features_f], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'knn without f   score={score:.3f}  k=35  p=1  n={n}')
    
    knn = KNeighborsClassifier(n_neighbors=39, p=1, n_jobs=-1)
    score = cross_val_score(knn, train_f[features], train['ham'], scoring=fbeta3, cv=3, n_jobs=-1).mean()
    print(f'knn with f      score={score:.3f}  k=35  p=1  n={n}')
def generate_submission(classifier, xtrain, _features, out_file):
    xtrain = xtrain[_features]
    ytrain = train['ham']
    xtest = test[_features]
    classifier.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    df = pd.DataFrame({'Id': test['Id'], 'ham': ypred})
    df.to_csv(out_file, index=False)
    
    score = cross_val_score(classifier, xtrain, ytrain, scoring=fbeta3, cv=10, n_jobs=-1).mean()
    print('{0}: score={1:.3f} features={2}\n'.format(out_file, score, _features))
generate_submission(KNeighborsClassifier(n_neighbors=39, p=1), train_f, features, 'sub1.csv')

pipe = Pipeline([
    ('ss', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier(n_neighbors=35, p=1)),
])
generate_submission(pipe, train, features, 'sub2.csv')

pipe = Pipeline([
    ('bin', KBinsDiscretizer(n_bins=5, strategy='quantile', encode='ordinal')),
    ('nb', MultinomialNB()),
])
generate_submission(pipe, train_f, features, 'sub3.csv')
from scipy import interp

pipe = Pipeline([
    ('ss', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier(n_neighbors=35, p=1)),
])

cv = StratifiedKFold(n_splits=5)

X = train[features].values
y = train['ham'].values

# Each element of tprs refers to the true positive rates of an interation of the cross validation.
# The true positive rates in tprs correspond to the false positive rates in mean_fpr.
# I.e. the first trp corresponds to fpr=0.00. The second one, to fpr=0.01.
tprs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for i, (itrain, itest) in enumerate(cv.split(X, y)):
    pipe.fit(X[itrain], y[itrain])
    proba = pipe.predict_proba(X[itest])
    fpr, tpr, thresholds = roc_curve(y[itest], proba[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label=f'ROC fold {i}')

    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC',
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.show()