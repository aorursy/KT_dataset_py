import os

from pathlib import Path

import pandas as pd

import numpy as np

root = Path("../input/titanic")
os.listdir(root)
train_file, test_file, submission_file = "train.csv", "test.csv", "gender_submission.csv", 
def load_data(root, file_name):

    return pd.read_csv(root/file_name)

train = load_data(root, train_file)

test = load_data(root, test_file)
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
train.shape
test.shape
train.dtypes
train.describe()
import seaborn as sns



from IPython.display import clear_output

from matplotlib import pyplot as plt

%matplotlib inline

with sns.axes_style(style = "darkgrid"):

    grid = sns.FacetGrid(data = train, hue = "Survived", height = 7)

    grid.map(sns.distplot, "Age", kde = True, hist_kws={"alpha":0.8})

    grid.add_legend(fontsize = 12)

columns = ["Pclass", "Embarked", "Survived"]

with sns.axes_style(style = "whitegrid"):

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))

    for c in range(len(axes)):

        sns.countplot(x = columns[c], hue = "Sex", data = train, palette="cubehelix", ax = axes[c])

with sns.axes_style(style = "whitegrid"):

    grid = sns.FacetGrid(data = train, col = "Embarked", row = "Sex", hue = "Survived", height = 4)

    grid.map(sns.countplot, "Pclass", order = [1, 2, 3]).add_legend(fontsize = 12)
with sns.axes_style(style = "whitegrid"):

    fig, axes = plt.subplots(1, 2, figsize = (16, 5))

    columns = ["Pclass", "Embarked"]

    for c in range(len(axes)):

        count = sns.pointplot(x = columns[c], y = "Survived", hue = "Sex", data = train, ax = axes[c])

        count.set_title(f"the probabilities of survived female, male in {columns[c]}".title(), fontsize = 14)

def clean_data(old_data):

    data = old_data.copy()

    data.drop(labels = ["PassengerId", "Cabin", "Name", "Ticket"], axis = 1, inplace = True)

    data['Embarked'].fillna(value = data['Embarked'].mode()[0], inplace = True)



    return data
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y = None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder
def full_preparation(categories, numerical):

    num_pipeline = Pipeline([    

        ('selector', DataFrameSelector(numerical)),

        ('std_scaler', MinMaxScaler(feature_range = (0, 1))),

    ])

    cat_pipeline = Pipeline([    

        ('selector', DataFrameSelector(categories)),

        ('cat_encoder', OneHotEncoder(sparse = False, drop = "first")),

    ])

    full_pipeline = FeatureUnion(transformer_list = [

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])

    

    return full_pipeline
cats = ["Sex", "Embarked"]

nums = ["Pclass", "Age","SibSp", "Parch", "Fare"]
data = clean_data(train)
features, target = data.drop("Survived", axis = 1), data.Survived.to_numpy()

prep = full_preparation(cats, nums)

prepare = prep.fit(features)

prep_features = prep.transform(features)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(prep_features, target, test_size = 0.2, random_state = 42)
len(x_train), len(x_test)
# Evaluate the models

from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV as grid_search

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings("ignore")
clf = CatBoostClassifier(iterations=500, thread_count=4)



params = {

    "depth":range(4, 10, 2),

    "learning_rate":[0.01, 0.1, 0.3],

    "l2_leaf_reg":[1, 3, 5, 10],

    "border_count":[5, 10, 20],

}
%%time



search = grid_search(clf, params, cv=3).fit(x_train, y_train, eval_set=(x_test, y_test))



clear_output(wait=True)
search.best_params_
model = search.best_estimator_

model.best_score_
model.fit(x_train, y_train, plot=True, eval_set=(x_test, y_test))
model.best_score_
perdictions = model.predict(x_test)

accuracy_score(perdictions, y_test)
test_data = clean_data(test)

prep_test = prep.transform(test_data)
predictions = list(model.predict(prep_test))
output = pd.DataFrame({

    

    'PassengerId':test.PassengerId, 

    'Survived':predictions

})



output.to_csv("submission.csv", index = False)