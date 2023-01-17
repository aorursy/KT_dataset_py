# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate



import lightgbm as lgb



from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/eval-lab-2-f464/train.csv')
test = pd.read_csv('../input/eval-lab-2-f464/test.csv')
dataset.head()
dataset.describe()
len(dataset)
dataset.isnull().sum()
cor = dataset.corr()
cor
dataset.drop(['id'], axis = 1, inplace = True)
X = dataset[dataset.drop(['class'], axis = 1).columns].values

y = dataset['class'].values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 20/120)
clf = DecisionTreeClassifier(random_state = 42)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
clf = RandomForestClassifier(random_state = 42)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
clf = GradientBoostingClassifier(random_state = 42)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
clf = AdaBoostClassifier(random_state = 42)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
clf = ExtraTreesClassifier(random_state = 42, max_depth = 10, n_estimators = 1000, min_samples_split = 2)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))

scoring = ['accuracy']

scores = cross_validate(clf, X, y, cv=6, scoring = scoring)



print(np.mean(scores['test_accuracy']))
clf = BernoulliNB()

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
clf = LogisticRegression(random_state = 42, penalty='l1')

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
"""gsc = GridSearchCV(

         XGBClassifier(),

        param_grid={

            'max_depth':[1,2,3],

            'eta': (0.14,0.15,0.16,0.17,0.18,0.19)

        },

        cv=6,  verbose=1, n_jobs=-1)

gsc.fit(train_x, train_y)

print(gsc.best_params_)

print(accuracy_score(train_y, gsc.predict(train_x)))

print(accuracy_score(test_y, gsc.predict(test_x)))"""
clf = SGDClassifier()

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))

print(accuracy_score(train_y, clf.predict(train_x)))
clf = KNeighborsClassifier(n_neighbors = 5, leaf_size = 10, weights = 'distance', algorithm = 'auto')

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))

print(accuracy_score(train_y, clf.predict(train_x)))
clf = XGBClassifier()

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
"""gsc = GridSearchCV(

         ExtraTreesClassifier(random_state = 47),

        param_grid={

            'bootstrap' : [True,False],

            'n_estimators': [600,700,800],

            'max_features' : ['auto','sqrt'],

            'max_depth' : [30,50,70],

            'min_samples_split' : [2,5,8],

            'min_samples_leaf': [1,2,4]

        },

        cv=3, verbose=1, n_jobs=-1)

gsc.fit(X, y)

print(gsc.best_params_)

print(accuracy_score(train_y, gsc.predict(train_x)))

print(accuracy_score(test_y, gsc.predict(test_x)))"""
clf = ExtraTreesClassifier(random_state = 42)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))

print(accuracy_score(train_y, clf.predict(train_x)))
clf = ExtraTreesClassifier(random_state = 0)

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))

print(accuracy_score(train_y, clf.predict(train_x)))
clf = GaussianNB()

clf.fit(train_x, train_y)

print(accuracy_score(test_y, clf.predict(test_x)))
test_X = test[test.drop('id', axis = 1).columns].values
gsc = GridSearchCV(

         KNeighborsClassifier(),

        param_grid={

            'n_neighbors': [1,2,3,4,5,6,7,8,9],

            'algorithm' : ['auto','ball_tree','kd_tree','brute'],

            'metric' : ['euclidean','manhattan','chebyshev'],

            'p' : [1,2,3,4],

            'leaf_size' : [1,2,3]

        },

        cv=6, verbose=1, n_jobs=-1)

gsc.fit(X, y)

print(gsc.best_params_)

print(accuracy_score(train_y, gsc.predict(train_x)))

print(accuracy_score(test_y, gsc.predict(test_x)))

test_Y = gsc.predict(test_X)
data = {'id':test['id'],'class':test_Y}

df = pd.DataFrame(data)
df.to_csv(path_or_buf  ='output.csv', index = False)