# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
label = train_data["Survived"]

train_data = train_data.drop(["Survived"], axis=1)

train_data.head()
train_data.info()
train_data["Embarked"].value_counts()
train_data["Sex"].value_counts()
train_data["Age"].value_counts()
train_data["Fare"].describe()
from sklearn.base import BaseEstimator

from sklearn.base import TransformerMixin



class ColumnSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, columns = ["Pclass", "Sex", "Age", "Parch", "Fare", "Cabin", "Embarked"]):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        X = X.set_index("PassengerId")

        X = X.fillna("NaN")

        

        # we also keep just the first letter of the cabin if we know

        X["Cabin"] = [x[0] if x != "NaN" else "UKN" for x in X["Cabin"]]

        

        assert isinstance(X, pd.DataFrame)



        try:

            return X[self.columns]

        except KeyError:

            cols_error = list(set(self.columns) - set(X.columns))

            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

            

cs = ColumnSelector()

transformed = cs.fit_transform(train_data)

transformed.head()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

import warnings



class DataImputer(BaseEstimator, TransformerMixin):

    

    

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        X["Unknowed_Age"] = (X["Age"] == "NaN")

        X["Unknowed_Age"] = [1 if x else 0 for x in X["Unknowed_Age"]]

        X["Age"] = [x if x != "NaN" else np.nan for x in X["Age"]]

        X["Age"] = [X["Age"].mean() if np.isnan(x) else x for x in X["Age"]]

        X["Age"] = X["Age"].astype(int)

        X["Fare"] = [x if x != "NaN" else np.nan for x in X["Fare"]]

        X["Fare"] = [X["Fare"].mean() if np.isnan(x) else x for x in X["Fare"]]

        X["Unknowed_Cabin"] = [1 if x == "UKN" else 0 for x in X["Cabin"]]

        return X



data_imp = DataImputer()

transformed = data_imp.fit_transform(transformed)

transformed.head()
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



class DataEncoder(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        sex_cat = X[['Sex']]

        ordinal_encoder = OneHotEncoder()

        sex_cat_1hot = ordinal_encoder.fit_transform(sex_cat)

        sex = pd.DataFrame(sex_cat_1hot.toarray(), columns=['Male', 'Female'])

        sex.index = X.index

        X["Male"] = sex["Male"]

        X["Female"] = sex["Female"]

        X = X.drop(["Sex"], axis=1)



        label_encoder = LabelEncoder()

        X["Cabin"] = label_encoder.fit_transform(X[["Cabin"]])

        X["Embarked"] = label_encoder.fit_transform(X[["Embarked"]])

        

        return X





custom_encoder = DataEncoder()

transformed = custom_encoder.fit_transform(transformed)

transformed.head()
transformed["Fare"].describe()
from sklearn.preprocessing import StandardScaler



class DataScaler(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        scaler = StandardScaler()

        X["Fare"] = scaler.fit_transform(X[["Fare"]])

        return X





scaler = DataScaler()

transformed = scaler.fit_transform(transformed)

transformed.head()
from sklearn.pipeline import Pipeline



full_pipeline = Pipeline([

    ('selector', ColumnSelector()),

    ('imputer', DataImputer()),

    ('encoder', DataEncoder()),

    ('scaler', DataScaler()),

])

train = full_pipeline.fit_transform(train_data)

train.head()
train.info()
from sklearn.linear_model import LogisticRegression, SGDClassifier

from scipy.stats import uniform

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict, RandomizedSearchCV

from sklearn.metrics import roc_auc_score, accuracy_score



X = train.copy()

Y = label.copy()



classifiers = dict()

scores = dict()



sgd_cl = SGDClassifier()

distributions = dict(tol=[1e-3, 1e-2, 0.1], alpha=[0.0001, 0.001], penalty=['l2', 'l1', 'none', 'elasticnet'])

rscv = RandomizedSearchCV(sgd_cl, distributions, cv=5, n_iter=200)

best_model = rscv.fit(X, Y)

scores["sgd_cl"] = accuracy_score(Y, best_model.predict(X))

classifiers["sgd_cl"] = best_model



log_reg = LogisticRegression()

distributions = dict(C=uniform(loc=0, scale=4), solver=["lbfgs", "saga"], penalty=['l2', 'none'])

rscv = RandomizedSearchCV(log_reg, distributions, cv=5, n_iter=200)

best_model = rscv.fit(X, Y)

scores["log_reg"] = accuracy_score(Y, best_model.predict(X))

classifiers["log_reg"] = best_model



lda = LinearDiscriminantAnalysis()

distributions = dict(tol=[1e-5, 1e-4, 1e-3])

rscv = RandomizedSearchCV(lda, distributions, cv=5, n_iter=100)

best_model = rscv.fit(X, Y)

scores["lda"] = accuracy_score(Y, best_model.predict(X))

classifiers["lda"] = best_model



knc = KNeighborsClassifier()

distributions = dict(weights=['uniform', 'distance'], leaf_size=[20, 30, 40])

rscv = RandomizedSearchCV(knc, distributions, cv=5, n_iter=100)

best_model = rscv.fit(X, Y)

scores["knc"] = accuracy_score(Y, best_model.predict(X))

classifiers["knc"] = best_model



gnb = GaussianNB()

distributions = dict(var_smoothing=[1e-10, 1e-9, 1e-8, 1e-7])

rscv = RandomizedSearchCV(gnb, distributions, cv=5, n_iter=100)

best_model = rscv.fit(X, Y)

scores["gnb"] = accuracy_score(Y, best_model.predict(X))

classifiers["gnb"] = best_model



dtc = DecisionTreeClassifier()

distributions = dict(splitter=['random', 'best'], criterion=['gini', 'entropy'], class_weight=['balanced', None])

rscv = RandomizedSearchCV(dtc, distributions, cv=5, n_iter=200)

best_model = rscv.fit(X, Y)

scores["dtc"] = accuracy_score(Y, best_model.predict(X))

classifiers["dtc"] = best_model



svc = SVC()

distributions = dict(C=uniform(loc=0, scale=4), gamma=['scale', 'auto'], tol=[1e-4, 1e-3, 1e-2], probability=[True, False])

rscv = RandomizedSearchCV(svc, distributions, cv=5, n_iter=200)

best_model = rscv.fit(X, Y)

scores["svc"] = accuracy_score(Y, best_model.predict(X))

classifiers["svc"] = best_model



forest_clf = RandomForestClassifier()

distributions = dict(n_estimators=[10,100,1000], criterion=["gini", "entropy"])

rscv = RandomizedSearchCV(forest_clf, distributions, cv=5, n_iter=200)

best_model = rscv.fit(X, Y)

scores["forest_clf"] = accuracy_score(Y, best_model.predict(X))

classifiers["forest_clf"] = best_model



print(scores)
picked_model = classifiers["dtc"]

picked_model.best_estimator_.feature_importances_
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test = full_pipeline.fit_transform(test_data)

test.head()
test.info()
predictions = picked_model.predict(test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Submission saved!")