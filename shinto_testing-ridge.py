import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

print(train_set.shape, test_set.shape)
train = train_set.drop(['id','loss'], axis=1)

test = test_set.drop(['id'], axis=1)

all_data = pd.concat((train, test)).reset_index(drop=True)

#all_data = train_set.drop(['id','loss'], axis=1)

cat_features = [x for x in all_data.select_dtypes(include=['object'])]

num_features = [x for x in all_data.select_dtypes(exclude=['object'])]

for c in range(len(cat_features)):

    all_data[cat_features[c]] = all_data[cat_features[c]].astype('category').cat.codes

all_data.head()
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import  SVR

from sklearn.model_selection import cross_val_score
def do_cross_validation(X_train, y_train):

    models = []

    models.append(('LR', LinearRegression()))

    #models.append(('LASSO', Lasso()))

    #models.append(('EN', ElasticNet()))

    models.append(('Ridge', Ridge()))

    #models.append(('CART', DecisionTreeRegressor()))

    #models.append(('KNN', KNeighborsRegressor()))

    #models.append(('SVR', SVR()))

    results = []

    names = []

    scoring = 'neg_mean_absolute_error'

    for name, model in models:

        cv_results = -cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
X_train = all_data[:train.shape[0]]

y_train = np.log(train_set['loss'])

do_cross_validation(X_train, y_train)
X_test = all_data[train.shape[0]:]

ridge = Ridge()

ridge.fit(X_train, y_train)

submission = pd.DataFrame({'Id': test_set.id, 'loss':np.exp(ridge.predict(X_test))})

submission.head()
submission.to_csv('submissionRidge.csv', index=False)