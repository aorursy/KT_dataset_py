# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

FILE_PATH = '../input/titanic/train.csv'

titanic = pd.read_csv(FILE_PATH)

titanic.head()
titanic.info()
titanic.describe()
titanic['Pclass'].value_counts()
import matplotlib.pyplot as plt

titanic.hist(bins=50, figsize=(20,15))
corr_matrix = titanic.corr()

corr_matrix['Survived'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ['Survived', 'Fare', 'Parch', 'Age', 'Pclass']

scatter_matrix(titanic[attributes], figsize=(20,12))
shuffled_indices = np.random.permutation(len(titanic))

titanic_shuffled = titanic.iloc[shuffled_indices]

titanic_shuffled.shape
train_features = titanic_shuffled.drop("Survived", axis=1)

train_labels = titanic_shuffled["Survived"].copy()
train_features
train_features.info()
from sklearn.base import BaseEstimator, TransformerMixin



class AttributeDropper(BaseEstimator, TransformerMixin):

    def __init__(self, drop_attributes=True):

        self.drop_attributes = drop_attributes

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        if self.drop_attributes:

            return X.drop(['PassengerId', "Name", "Cabin"], axis=1)

        else:

            return X

        

attrib_dropper = AttributeDropper()

new_features = attrib_dropper.fit_transform(train_features)

train_features.shape, new_features.shape
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



num_attributes = ["Age", "SibSp", "Parch", "Fare"]

cat_attributes = ["Sex", "Embarked", "Pclass"]



num_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="median")),

    ("std_scaler", StandardScaler())

])



cat_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("cat", OneHotEncoder())

])



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attributes),

    ("cat", cat_pipeline, cat_attributes)

])



full_pipeline_with_dropped_attribs = Pipeline([

    #("drop", AttributeDropper()),

    ("num_cat", full_pipeline),

])



X_train = full_pipeline_with_dropped_attribs.fit_transform(train_features)

y_train = train_labels

train_features.shape, X_train.shape, y_train.shape

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)



cross_val_score(sgd_clf, X_train, y_train, cv=5, scoring='accuracy', verbose=3)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

cross_val_score(knn_clf, X_train, y_train, cv=5, scoring='accuracy', verbose=3)
from sklearn.model_selection import GridSearchCV

params = [{

    'weights': ['uniform', 'distance'],

    'n_neighbors': [3, 4, 5]

}]

grid_search = GridSearchCV(knn_clf, params, cv=5, verbose=3)

grid_search.fit(X_train, y_train)
grid_search.best_score_
best_model = grid_search.best_estimator_

best_model
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier()

cross_val_score(forest_clf, X_train, y_train, cv=5, scoring='accuracy', verbose=3)
params = [

    {'n_estimators': [3, 10, 30, 100], 'max_features': ['auto', 'sqrt', 'log2']},

    {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': ['auto', 'sqrt', 'log2']}

]

grid_search = GridSearchCV(forest_clf, params, cv=5, verbose=3)

grid_search.fit(X_train, y_train)
grid_search.best_score_
best_model = grid_search.best_estimator_
best_model
training_pipeline_with_predictor = Pipeline([

    ("prep", full_pipeline),

    ("train", best_model)

])

TEST_FILE_PATH = '../input/titanic/test.csv'

X_test = pd.read_csv(TEST_FILE_PATH)

X_test.shape
X_test
X_test['PassengerId']
y_predict = training_pipeline_with_predictor.predict(X_test)
y_predict.shape
y_predict
result = np.c_[X_test['PassengerId'], y_predict]
result
df = pd.DataFrame(result)
df.to_csv('result.csv')