import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_recall_fscore_support, make_scorer, fbeta_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

%matplotlib inline
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
# This dataset is really clean, let's double check

df.info()
sns.countplot(x='quality', data=df)
df['high_quality'] = (df['quality'] >= 7).astype('int64')

df['high_quality'].value_counts() / df['high_quality'].count()
df['high_quality'].value_counts().plot.pie(explode=[0, 0.1], figsize=(7,7))
corr = df.drop(columns='quality').corr()

idx = corr['high_quality'].abs().sort_values(ascending=False).index[:5]

idx_features = idx.drop('high_quality')

sns.heatmap(corr.loc[idx, idx])
_, ax = plt.subplots(1, 4, figsize=(20, 5))

for i, var in enumerate(idx_features):

    sns.boxplot(x='high_quality', y=var, data=df, ax=ax.flatten()[i])

sns.despine()
def scale(X_train, X_test):

    # Note that we only use the training data to fit the scaler, so that

    # no information from the test set leaks into our model

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test



def polynomials(X_train, X_test, degree=2):

    pol = PolynomialFeatures(degree)

    return pol.fit_transform(X_train), pol.transform(X_test)



def split(X, y):

    return train_test_split(X, y, test_size=0.3, random_state=1)
f1_beta = 0.25



def _compute_all_scores(model, X, y):

    return precision_recall_fscore_support(y, model.predict(X), average='binary', beta=f1_beta)[:-1]



def train(X_train, X_test, y_train, y_test, model):

    model.fit(X_train, y_train)

    prec, rec, f1 = _compute_all_scores(model, X_train, y_train)

    print('TRAIN: prec={:.4f}, recall={:.4f}, f1={:.4f}'.format(prec, rec, f1))

    prec, rec, f1 = _compute_all_scores(model, X_test, y_test)

    print('TEST : prec={:.4f}, recall={:.4f}, f1={:.4f}'.format(prec, rec, f1))

    

scorer = make_scorer(fbeta_score, beta=f1_beta)
# These are our features. X_subset are the ones that we identified as the most relevant.

X = df.drop(columns=['quality', 'high_quality'])

X_subset = X[idx_features].copy()

y = df['high_quality']
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

train(X_train, X_test, y_train, y_test, LogisticRegression(random_state=0))
X_train, X_test, y_train, y_test = split(X_subset, y)

X_train, X_test = scale(X_train, X_test)

X_train, X_test = polynomials(X_train, X_test)

train(X_train, X_test, y_train, y_test, LogisticRegression(random_state=0))
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

X_train, X_test = polynomials(X_train, X_test)

train(X_train, X_test, y_train, y_test, LogisticRegressionCV(random_state=0, max_iter=500, scoring=scorer))
X_train, X_test, y_train, y_test = split(X_subset, y)

X_train, X_test = scale(X_train, X_test)

X_train, X_test = polynomials(X_train, X_test)

train(X_train, X_test, y_train, y_test, LogisticRegressionCV(random_state=0, max_iter=500, scoring=scorer))
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

train(X_train, X_test, y_train, y_test, DecisionTreeClassifier(random_state=0, max_leaf_nodes=30))
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

train(X_train, X_test, y_train, y_test, RandomForestClassifier(random_state=0))
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

model = GridSearchCV(RandomForestClassifier(random_state=0), {

    'n_estimators': [100, 200, 300],

    'max_features': [1, 2, 3],

    'max_depth' : [4, 6, 8]

}, scoring=scorer)

train(X_train, X_test, y_train, y_test, model)
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

train(X_train, X_test, y_train, y_test, SVC(random_state=0))
X_train, X_test, y_train, y_test = split(X, y)

X_train, X_test = scale(X_train, X_test)

X_train, X_test = polynomials(X_train, X_test)

train(X_train, X_test, y_train, y_test, SVC(random_state=0, kernel='linear'))