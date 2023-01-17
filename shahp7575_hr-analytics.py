import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('../input/HR_comma_sep.csv')
dataset.info()
dataset.head()
dataset.isnull().any()
dataset.salary.value_counts()
dataset.describe()
dataset.describe(include=['O'])
dataset.hist(bins = 50, figsize = (12, 10))
train_set, test_set = train_test_split(dataset, test_size = 0.2, random_state = 42)
train_set.left.value_counts()/ len(train_set)
test_set.left.value_counts() / len(test_set)
hr_analytics = train_set.copy()
g = sns.FacetGrid(hr_analytics, col = 'left')
g.map(plt.hist, 'satisfaction_level', bins = 30)
g = sns.FacetGrid(hr_analytics, col = 'left', height = 3.0)
g.map(plt.hist, 'last_evaluation', bins = 30)
sns.catplot(x = 'number_project', hue = 'left', kind = 'count', data = hr_analytics)
g = sns.FacetGrid(hr_analytics, col = 'left', height = 3.0)
g.map(plt.hist, 'average_montly_hours', bins = 25)
sns.catplot(x = 'time_spend_company', hue = 'left', kind = 'count', data = hr_analytics)
sns.catplot(x = 'promotion_last_5years', hue = 'left', kind = 'count', data = hr_analytics)
sns.catplot(x = 'Department', hue = 'left', kind = 'count', data = hr_analytics, height = 8)
sns.catplot(x = 'salary', hue = 'Department', col = 'left', kind = 'count', data = hr_analytics, height = 6, palette = 'RdBu')
hr_analytics.head()
corr_matrix = hr_analytics.corr()
corr_matrix.left.sort_values(ascending = False)
hr_analytics[['Department', 'left']].groupby('Department').mean().sort_values(ascending = False, by = 'left')
hr_analytics[['salary', 'left']].groupby('salary').mean().sort_values(ascending = False, by = 'left')
hr_analytics = train_set.drop('left', axis = 1)
hr_analytics_labels = train_set.iloc[:, 6].copy()
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
hr_analytics.head()
num_attribs = ['satisfaction_level', 'Work_accident', 'time_spend_company']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('scaler', StandardScaler())
])
cat_attribs = ['salary', 'Department']

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('onehotencoder', OneHotEncoder(sparse = False))
])
preprocess_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])
hr_analytics_prepared = preprocess_pipeline.fit_transform(hr_analytics)
hr_analytics_prepared
log_clf = LogisticRegression()
log_scores = cross_val_score(log_clf, hr_analytics_prepared, hr_analytics_labels, cv = 10)
log_scores.mean()
knn_clf = KNeighborsClassifier()
knn_scores = cross_val_score(knn_clf, hr_analytics_prepared, hr_analytics_labels, cv = 10)
knn_scores.mean()
tree_clf = DecisionTreeClassifier()
tree_scores = cross_val_score(tree_clf, hr_analytics_prepared, hr_analytics_labels, cv = 10)
tree_scores.mean()
forest_clf = RandomForestClassifier()
forest_scores = cross_val_score(forest_clf, hr_analytics_prepared, hr_analytics_labels, cv = 10)
forest_scores.mean()
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN Classifier', 'Decision Tree Classifier', 'Random Forest Classifier'],
    'Score': [log_scores.mean(), knn_scores.mean(), tree_scores.mean(), forest_scores.mean()]
})
models.sort_values(by = 'Score', ascending=False)
param_grid = [
    {'n_estimators': [100, 200, 500]},
    {'criterion': ['gini', 'entropy']},
    {'max_features': ['auto', 'sqrt', 'log2']}
]
grid_search = GridSearchCV(forest_clf, param_grid, cv = 10, verbose = 3)
grid_search.fit(hr_analytics_prepared, hr_analytics_labels)
grid_search.best_params_
grid_search.best_score_
forest_clf = RandomForestClassifier(n_estimators = 100)
forest_scores = cross_val_score(forest_clf, hr_analytics_prepared, hr_analytics_labels, cv = 10)
forest_scores.mean()
hr_test_prepared = preprocess_pipeline.transform(test_set)
hr_test_prepared
hr_test_labels = test_set.iloc[:, 6].copy()
forest_clf.fit(hr_analytics_prepared, hr_analytics_labels)

y_pred = forest_clf.predict(hr_test_prepared)
y_pred
accuracy_score(hr_test_labels, y_pred)
