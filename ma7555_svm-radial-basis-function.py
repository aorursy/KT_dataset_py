# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

from scipy.stats import norm

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import mutual_info_classif

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

warnings.simplefilter(action='ignore', category=FutureWarning)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.set_index('id', drop=True, inplace=True)

target = train.target

train.drop('target', axis=1, inplace=True)

test.set_index('id', drop=True, inplace=True)
print(train.dtypes)
for column in train.columns:

    if train[column].dtype != np.float64:

        print(column, train[column].dtype)
train['wheezy-copper-turtle-magic'].describe()
float_features = list(train.columns)

float_features.remove('wheezy-copper-turtle-magic')



plt.figure(figsize=(10,4))

for feature in float_features:

    g = sns.kdeplot(train[feature], shade=True)

    g.legend().set_visible(False)

plt.show()
mu = np.mean(train[float_features[0]])

sigma = np.std(train[float_features[0]])



x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)

plt.plot(x, norm.pdf(x, mu, sigma))

sns.kdeplot(train[float_features[0]])

plt.show()
# Feature selection -- To be implemented later on

features = train.columns



# Splitting into train sets and test sets

x_train, x_test, y_train, y_test = train_test_split(train[features], target, 

                                                    train_size=0.8, test_size=0.2,

                                                    random_state=555)
classifier = SVC(kernel = 'rbf')

classifier.fit(x_train[:2500].drop('wheezy-copper-turtle-magic', axis=1), y_train[:2500])

print(classifier.score(x_test.drop('wheezy-copper-turtle-magic', axis=1), y_test))
classifier = DecisionTreeClassifier(criterion='entropy', random_state=555)

classifier.fit(x_train[:2500].drop('wheezy-copper-turtle-magic', axis=1), y_train[:2500])

print(classifier.score(x_test.drop('wheezy-copper-turtle-magic', axis=1), y_test))
train.groupby('wheezy-copper-turtle-magic').size()
len(train)/512
# Classify using turtle-magic as index for datasets without any parameter tuning...

score_trees = []

score_svm = []



for i in range(512):

    sub_train = x_train[x_train['wheezy-copper-turtle-magic'] == i]

    sub_target_train = target.loc[sub_train.index]

    sub_test = x_test[x_test['wheezy-copper-turtle-magic'] == i]

    sub_target_test = target.loc[sub_test.index]

    

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=555)

    classifier.fit(sub_train, sub_target_train)

    score_trees.append(classifier.score(sub_test, sub_target_test))

    

    classifier = SVC(kernel = 'rbf')

    classifier.fit(sub_train, sub_target_train)

    score_svm.append(classifier.score(sub_test, sub_target_test))

    if i % 40 == 0: print('Completed: {:.1f}%'.format(i*100/512))

print('Trees:', np.mean(score_trees))

print('SVM:', np.mean(score_svm))
# Feature Selection, Grid Search with cross-validation for every turtle-magic

# This will take a  long time.. 

score_svm = []

best_c = []

best_gamma = []

best_features_mask = []



for i in range(512):

    sub_train = train[train['wheezy-copper-turtle-magic'] == i]

    sub_target_train = target.loc[sub_train.index]

    

    sel = VarianceThreshold(threshold=3) # returns around 40 features +- ..

    

    sub_train = sel.fit_transform(sub_train)

    best_features_mask.append(sel.get_support())



    C_range = [0.75, 1.5, 3]

    gamma_range = [0.0012, 0.0016, 0.002, 0.0024, 0.0028, 0.0032, 0.0036]

    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.075)

    grid = GridSearchCV(SVC(kernel = 'rbf'), param_grid=param_grid, cv=cv)

    grid.fit(sub_train, sub_target_train)



    best_c.append(grid.best_params_['C'])

    best_gamma.append(grid.best_params_['gamma'])



    if i % 15 == 0: print("Best parameters for turtle-magic {}: (C: {} & gamma: {}) - Best score of {:.3f} - Picked n_features = {}".format(i, grid.best_params_['C'], grid.best_params_['gamma'], grid.best_score_, sub_train.shape[1]))

    if i % 15 == 0: print('Completed: {:.1f}%'.format(i*100/512))

submission = pd.read_csv('../input/sample_submission.csv')

submission.set_index('id', drop=False, inplace=True)

submission.head()
for i in range(512):



    sub_train = train[train['wheezy-copper-turtle-magic'] == i]

    sub_target_train = target.loc[sub_train.index]

    

    sub_test = test[test['wheezy-copper-turtle-magic'] == i]

    

    sub_features = sub_train.columns[best_features_mask[i]]

    

    classifier = SVC(kernel = 'rbf', C=best_c[i], gamma=best_gamma[i])

    classifier.fit(sub_train[sub_features], sub_target_train)

    

    prediction = classifier.predict(sub_test[sub_features])

    df = pd.DataFrame({'id': sub_test.index, 'target': prediction})

    df.set_index('id', inplace=True)

    submission.update(df)

    if i % 40 == 0: print('Completed: {:.1f}%'.format(i*100/512))

submission.head(20)
submission.to_csv('submission.csv', index=False)