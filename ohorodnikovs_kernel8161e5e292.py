import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import VotingClassifier

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
columns = pd.Index(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',

          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',

          'label'])

train = pd.read_csv('../input/adult-incomes-in-the-united-states/adult.data', header=None, names=columns)

test = pd.read_csv(

    '../input/adult-incomes-in-the-united-states/adult.test',

        header=None,

        names=columns,

        skiprows=1

)

train.shape, test.shape
train
test
train.isna().sum()
test.isna().sum()
train.dtypes
y_ = (test['label'].str.contains('>') * 1)

X_test, X_eval, y_test, y_eval = train_test_split(

    test.drop('label', axis=1),

    y_,

    test_size=5000,

    random_state=42,

    stratify=y_

)



X_train = train.drop('label', axis=1)

y_train = (train['label'].str.contains('>') * 1)

X_train.shape, X_eval.shape
num_features = X_train.dtypes[X_train.dtypes == 'int64'].index

cat_features = X_train.columns.difference(num_features)

discrete_features = pd.Index(['capital-gain', 'capital-loss'])
X_train[num_features].hist(figsize=(20, 15))
pipeline = Pipeline([

    ('extract', ColumnTransformer([

        ('discrete', KBinsDiscretizer(n_bins=3, strategy='kmeans'), discrete_features),

        ('one_hot', OneHotEncoder(), cat_features),

        ('scale', StandardScaler(), num_features.difference(discrete_features))

    ], remainder='passthrough')),

    #('svd', TruncatedSVD(n_components=50)),

    #('scale', StandardScaler())

])
pipeline.fit(X_train)

pipeline.transform(X_train)
svc_l_param = {

    'C': np.logspace(-3, 2, 6)

}

svc_l = GridSearchCV(

    LinearSVC(dual=False, class_weight='balanced'),

    svc_l_param,

    scoring='accuracy',

    n_jobs=-1,

    cv=10,

)
%%time

svc_l.fit(pipeline.transform(X_train), y_train)
svc_l.best_score_
svc_l.best_estimator_
svc_l.score(pipeline.transform(X_train), y_train)
%%time

log_reg = LogisticRegressionCV(

    cv=10,

    scoring='accuracy',

    class_weight='balanced',

    #solver='saga',

    max_iter=1000,

    n_jobs=-1,

    random_state=42

).fit(pipeline.transform(X_train), y_train)
log_reg.score(pipeline.transform(X_train), y_train)
XX_train, X_cv, yy_train, y_cv = train_test_split(

    X_train,

    y_train,

    test_size=0.3,

    random_state=24,

    stratify=y_train

)

pipeline.fit(XX_train)

XX_train.shape, X_cv.shape
%%time

forest_param = {

    'n_estimators': [200],

    'max_features': list(range(1, 14, 2)),

    'max_samples': [0.001, 0.1, 0.2, 0.3, 0.5]

}

x_tr = pipeline.transform(XX_train)

x_cv = pipeline.transform(X_cv)

best = (None, 0)

for param in ParameterGrid(forest_param):

    clf = RandomForestClassifier(

        **param,

        random_state=4,

        class_weight='balanced'

    ).fit(x_tr, yy_train)

    score = accuracy_score(y_cv, clf.predict(x_cv))

    if score > best[1]:

        best = (clf, score)

    else:

        pass

best[1]
best[0]
best[0].score(pipeline.transform(X_train), y_train)
accuracy_score(

    DummyClassifier(strategy='prior')

        .fit(X_test, y_test)

        .predict(X_test),

    y_test)
log_reg.score(pipeline.transform(X_eval), y_eval)
svc_l.score(pipeline.transform(X_eval), y_eval)
best[0].score(pipeline.transform(X_eval), y_eval)
best_clf = best[0]

best_clf.fit(pipeline.transform(X_train.append(X_eval)), y_train.append(y_eval))

best_clf.score(pipeline.transform(X_test), y_test)