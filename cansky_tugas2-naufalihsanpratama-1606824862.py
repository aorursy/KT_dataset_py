import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

hr = pd.read_csv('../input/hrcsv/hr_data.csv')

hr.head()
for item in hr:

    if(hr[item].dtype == np.float64 or hr[item].dtype == np.int64):

        if hr[item].isnull().any():

            hr[item] = hr[item].fillna(hr[item].median())

    else:

        hr[item] = hr[item].fillna("KOSONG")
numerical_features = ['satisfaction_level','last_evaluation',

                      'number_project','average_montly_hours',

                      'time_spend_company','Work_accident',

                      'promotion_last_5years']



numerical = pd.get_dummies(hr[numerical_features])

numerical.head()
categorical_features = ['sales','salary']

categorical = pd.get_dummies(hr[categorical_features])

categorical.head()
x = np.hstack((numerical,categorical))

y = hr['left']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



X_train
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train,y_train)

logreg.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

rfc.score(X_test,y_test)

from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier()

abc.fit(X_train,y_train)

abc.score(X_test,y_test)
all_features = [item for item in numerical] + [item for item in categorical]

len(all_features)
# from sklearn.tree import export_graphviz



# estimator = rfc.estimators_[5]

# export_graphviz(estimator, 

#                 out_file='tree.dot', 

#                 feature_names = all_features,

#                 rounded = True, proportion = False, 

#                 precision = 2, filled = True)



# from subprocess import call

# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# from IPython.display import Image

# Image(filename = 'tree.png')
importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, all_features[indices[f]], importances[indices[f]]))





plt.figure(figsize=(7,7))

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), indices)

plt.xlim([-1, X_train.shape[1]])

plt.show()
sample_X_train = X_train[0:1000]

sample_y_train = y_train[0:1000]



sample_x_test = X_test[0:500]

sample_y_test = y_test[0:500]
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import classification_report

# from sklearn.svm import SVC



# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

#                      'C': [1, 10, 100, 1000]},

#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



# scores = ['precision', 'recall']



# for score in scores:

#     print("# Tuning hyper-parameters for %s" % score)

#     print()



#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,

#                        scoring='%s_macro' % score)

#     clf.fit(sample_X_train, sample_y_train)



#     print("Best parameters set found on development set:")

#     print()

#     print(clf.best_params_)

#     print()

#     print("Grid scores on development set:")

#     print()

#     means = clf.cv_results_['mean_test_score']

#     stds = clf.cv_results_['std_test_score']

#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):

#         print("%0.3f (+/-%0.03f) for %r"

#               % (mean, std * 2, params))

#     print()



#     print("Detailed classification report:")

#     print()

#     print("The model is trained on the full development set.")

#     print("The scores are computed on the full evaluation set.")

#     print()

#     y_true, y_pred = y_test, clf.predict(X_test)

#     print(classification_report(y_true, y_pred))

#     print()

# # Note the problem is too easy: the hyperparameter plateau is too flat and the

# # output model is the same for precision and recall with ties in quality.
train = pd.read_csv('../input/russianhouse/train.csv')

train.info()
train = pd.read_csv('../input/russianhouse/train.csv',parse_dates=['timestamp'])

train.info()
train = pd.read_csv('../input/russianhouse/train.csv')

test = pd.read_csv('../input/russianhouse/test.csv')

train_test = pd.concat([train,test])

train_test.info()
macro = pd.read_csv('../input/russianhouse/macro.csv')

macro_train_test = pd.merge_ordered(train_test,macro,on=['timestamp'])

macro_train_test.info()
for item in macro_train_test.columns:

    if(macro_train_test[item].dtype == np.float64 or macro_train_test[item].dtype == np.int64):

        if macro_train_test[item].isnull().any():

            macro_train_test[item] = macro_train_test[item].fillna(macro_train_test[item].median())

    else:

        macro_train_test[item] = macro_train_test[item].fillna("KOSONG")
numerical_features = list()

categorical_features = list() 



for item in macro_train_test:

    if(macro_train_test[item].dtype == np.float64 or macro_train_test[item].dtype == np.int64):

        numerical_features.append(item)

    else:

        categorical_features.append(item)



print('Numerical ', len(numerical_features))

print('Categorical ', len(categorical_features))
training = macro_train_test

x = training.drop(['price_doc'], axis=1)

y = macro_train_test['price_doc']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
numerical = pd.get_dummies(macro_train_test[numerical_features])

numerical = numerical.drop(['price_doc'], axis=1)

numerical.head(3)
categorical = pd.get_dummies(macro_train_test[categorical_features])

categorical.head(3)
x = np.hstack((numerical,categorical))

y = macro_train_test['price_doc']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error



def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



regr = LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



print('Linear Regression score: %f' % regr.score(X_test, y_test))



# Make predictions using the testing set

y_pred = regr.predict(X_test)



print("RMSE: %f" % mean_squared_error(y_test, y_pred))

print('MAE score: %f' % mean_absolute_error(y_test, y_pred))

print('SMAPE score: %f' % smape(y_test, y_pred))
from sklearn.linear_model import Lasso



lasso = Lasso()



# Train the model using the training sets

lasso.fit(X_train, y_train)



print('Lasso Score: %f' % lasso.score(X_train, y_train))



# Make predictions using the testing set

y_pred = lasso.predict(X_test)



print("RMSE: %f" % mean_squared_error(y_test, y_pred))

print('MAE score: %f' % mean_absolute_error(y_test, y_pred))

print('SMAPE score: %f' % smape(y_test, y_pred))
from sklearn.linear_model import Ridge



ridge = Ridge()



# Train the model using the training sets

ridge.fit(X_train, y_train)



print('Ridge Score: %f' % ridge.score(X_train, y_train))



# Make predictions using the testing set

y_pred = ridge.predict(X_test)



print("RMSE: %f" % mean_squared_error(y_test, y_pred))

print('MAE score: %f' % mean_absolute_error(y_test, y_pred))

print('SMAPE score: %f' % smape(y_test, y_pred))