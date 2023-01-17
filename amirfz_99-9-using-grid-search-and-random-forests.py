import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv('../input/HR_comma_sep.csv')

data = data.rename(columns={'sales': 'dept'})
data.head()
data.shape
from sklearn.preprocessing import LabelEncoder



class preproc():

    

    def __init__(self, data, cols):

        self.data = data

        

    def transform(self, dummies=False):

        if dummies:

            print('getting dummies for cat. variables...')

            self.data = pd.get_dummies(self.data, columns=cols)

            return self.data

        else:

            for col in cols:

                print('label encoding...')

                le = LabelEncoder()

                le.fit(self.data[col])

                self.data[col] = le.transform(self.data[col]) 

                print(le.classes_)

            return self.data
cols = ['dept', 'salary']

pp = preproc(data, cols)

data = pp.transform(dummies=False)
data.head()
data.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('left', axis=1), data['left'], test_size=0.3)
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
class RandomForestClassifierWithCoef(RandomForestClassifier):

    def fit(self, *args, **kwargs):

        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)

        self.coef_ = self.feature_importances_
from sklearn.feature_selection import RFECV
rf = RandomForestClassifierWithCoef()

rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='accuracy')

rfecv.fit(X_train, y_train)



print("Optimal number of features : %d" % rfecv.n_features_)
X_train.columns.values
rfecv.ranking_
plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
#X_train = X_train.loc[:,rfecv.support_]
X_train.columns.values
pipe = Pipeline([

    ('clf', RandomForestClassifier(random_state=0))

])
%%time



param_grid = {

        'clf__n_estimators': [3, 10, 30, 100],

        'clf__criterion': ['gini', 'entropy'],

        'clf__class_weight': [None, 'balanced', 'balanced_subsample']

        }



grid = GridSearchCV(pipe, cv=3, param_grid=param_grid, scoring='accuracy')

grid.fit(X_train, y_train)
grid.best_params_
grid.best_score_
grid.cv_results_['mean_test_score']