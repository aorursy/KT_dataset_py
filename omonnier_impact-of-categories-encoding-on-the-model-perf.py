%matplotlib inline



import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb



from sklearn.ensemble import GradientBoostingRegressor



from sklearn.linear_model import SGDRegressor



from sklearn.metrics import mean_absolute_error



from sklearn.model_selection import KFold



from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('../input/train.csv')
catFeatureslist = []

for colName,x in train.iloc[1,:].iteritems():

    if(str(x).isalpha()):

        catFeatureslist.append(colName)
for cf in catFeatureslist:

    le = LabelEncoder()

    le.fit(train[cf].unique())

    train[cf] = le.transform(train[cf])
ax = sns.violinplot(train.cat100, train.loss)

ax.axis([-1,15,0,6000])
clf_gbr = GradientBoostingRegressor(

    loss='ls',

    learning_rate=0.1,

    n_estimators=50,

    max_depth=5,

    max_features=0.12,

    random_state=69,

    subsample=0.5,

    verbose=0)
clf_sgdr = SGDRegressor(

    fit_intercept=False,

    loss='squared_loss',

    penalty='elasticnet',

    alpha=0.03,

    l1_ratio=0.7,

    learning_rate='invscaling',

    random_state=42,

    shuffle=True)
def evaluateModelPerf(clf, train, Y, Y_scaler=None):

    clf.fit(train, Y)

    print("Coefficient of determination on training set:", clf.score(train, Y))

    

    print("Score and Mean Absolute Error on the cross validation sets:")

    cv = KFold(n_splits=5, shuffle=True, random_state=33)

    maes = []

    scores = []

    for _, test_index in cv.split(train):

        Y_predict = clf.predict(train.iloc[test_index])

        if Y_scaler is not None:

            Y_scaled = Y[test_index] * Y_scaler.scale_ + Y_scaler.mean_

            Y_predict_scaled = Y_predict * Y_scaler.scale_ + Y_scaler.mean_

            mae = mean_absolute_error(Y_scaled, Y_predict_scaled)

        else:

            mae = mean_absolute_error(Y[test_index], Y_predict)

        score = clf.score(train.iloc[test_index], Y[test_index])

        print("score: {}, MAE: {}".format(score, mae))

        maes.append(mae)

        scores.append(score)



    print("Average score: {}".format(np.average(scores)))

    print("Average MAE: {}".format(np.average(maes)))
Y = train.loss

train = train.drop(["id", "loss"], axis=1)
evaluateModelPerf(clf_gbr, train, Y)
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train)

train_scaled = pd.DataFrame(train_scaled)

train_scaled.columns = train.columns

train = train_scaled



Y = scaler.fit_transform(Y[:, None])[:, 0]
evaluateModelPerf(clf_sgdr, train, Y, Y_scaler=scaler)
train = pd.read_csv('../input/train.csv')
categoriesOrder = {}

for cat in catFeatureslist:

    medians = train.groupby([cat])['loss'].median()

    ordered_medians = sorted(medians.keys(), key=lambda x: medians[x])

    categoriesOrder[cat] = ordered_medians
def tranformCategories(X):

    for cat, order in categoriesOrder.items():

        class_mapping = {v: order.index(v) for v in order}

        X[cat] = X[cat].map(class_mapping)

    return X



ft = FunctionTransformer(tranformCategories, validate=False)

train = ft.fit_transform(train)
ax = sns.violinplot(train.cat100, train.loss)

ax.axis([-1,15,0,6000])
Y = train.loss

train = train.drop(["id", "loss"], axis=1)
evaluateModelPerf(clf_gbr, train, Y)
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train)

train_scaled = pd.DataFrame(train_scaled)

train_scaled.columns = train.columns

train = train_scaled



Y = scaler.fit_transform(Y[:, None])[:, 0]
evaluateModelPerf(clf_sgdr, train, Y, Y_scaler=scaler)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("Categories found in test data, but not present in train data:")

for cat in catFeatureslist:

    testCat = set(test[cat].unique())

    trainCat = set(train[cat].unique())

    missing = testCat - trainCat

    if missing:

        nb_samples = 0

        for m in missing:

            nb_samples += test[test[cat] == m].shape[0]

        print("Feature: {}. Missing categories: {}. Number of samples: {}".format(cat, list(missing), nb_samples))
train = train.drop(["id", "loss"], axis=1)

test = test.drop(["id"], axis=1)

train = ft.fit_transform(train)

test = ft.fit_transform(test)
test[np.isnan(test.cat92)]
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

imp.fit(train)

test = imp.transform(test)

test = pd.DataFrame(test)

test.columns = train.columns
test[np.isnan(test.cat92)]
all_labels = set()

for cat in catFeatureslist:

    all_labels.update(train[cat].unique())

le = LabelEncoder()

le.fit(list(all_labels))

for cat in catFeatureslist:

    train[cat] = le.transform(train[cat])
corr = train[catFeatureslist].corr()
sns.heatmap(corr)
for cat in catFeatureslist:

    corr_order = corr[cat].order()

    print("{} gets maximum correlation factor with {} (corr={:.2})".format(

        cat,

        corr_order.index[-2],

        corr_order[-2]

    ))