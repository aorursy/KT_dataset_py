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
import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
train.info()
train.describe()
train[:5]
import plotly.graph_objects as go
survived = train[train["Survived"] == 1]
not_survived = train[train["Survived"] == 0]
sex_fig = go.Figure()
sex_fig.add_trace(go.Histogram(
    x = survived['Sex'],
    name = 'survived',
))
sex_fig.add_trace(go.Histogram(
    x = not_survived['Sex'],
    name = 'died',
))
sex_fig.update_layout(
    title = 'Sex',
    height = 500,
    width = 750
)
sex_fig.show()
pclass_fig = go.Figure()
pclass_fig.add_trace(go.Histogram(
    x = survived['Pclass'],
    name = 'survived'
))
pclass_fig.add_trace(go.Histogram(
    x = not_survived['Pclass'],
    name = 'dead'
))
pclass_fig.update_layout(
    title = 'Pclass',
    height = 500,
    width = 750
)
embarked_fig = go.Figure()
embarked_fig.add_trace(go.Histogram(
    x = survived['Embarked'],
    name = 'survived'
))
embarked_fig.add_trace(go.Histogram(
    x = not_survived['Embarked'],
    name = 'dead'
))
embarked_fig.update_layout(
    title = 'Embarked',
    height = 500,
    width = 750
)
# IMPORTANT FEATURES
#    num: PassengerID, Age, SibSp, Parch, Fare
#    cat: Sex, Pclass, Embarked
# TARGET FEATURE 
#    Survived
# FEATURE ENG
#    Drop Features: Name, Ticket, PassengerId
#    Combine Parch and SibSp to a single FamilyCount feature (sum of 2 columns), drop Parch and SibSp
#    Create buckets for Age (0-9, 10-19, 20-29...70-79)
#    Change Cabin values to only first letter, then fillna as individual attribute
train_data = train.drop(columns=['Name','Ticket','PassengerId'])
test_data = test.drop(columns=['Name','Ticket','PassengerId'])
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
# Combine Sibsp and Parch attributes (sum)
class NumAttributeEng(BaseEstimator, TransformerMixin):
    def __init__(self, combine_sibsp_parch = True):
        self.combine_sibsp_parch = combine_sibsp_parch
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        if self.combine_sibsp_parch == True:
            X['Famcount'] = X['SibSp'] + X['Parch']
            X = X.drop(columns=['SibSp', 'Parch'])
        return X
        
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("num_selector", DataFrameSelector(["Fare","Parch","SibSp"])),
    ("num_attribute_eng", NumAttributeEng()),
    ("num_imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

num_pipeline.fit_transform(train_data)
# Create bins for Age attribute and fill null values with no_age category
# Reduce Cabin attribute and fill null values with Z value represen
class CatAttributeEng(BaseEstimator, TransformerMixin):
    def __init__(self, combine_sibsp_parch = True):
        self.combine_sibsp_parch = combine_sibsp_parch
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X['Age_Bins'] = pd.cut(x=X['Age'], bins=[0,10,20,30,40,50,60,70,80])
        X['Age_Bins'] = X['Age_Bins'].cat.add_categories('No_Age')
        X[['Age_Bins']] = X[['Age_Bins']].fillna('No_Age')
        X['Age_Bins'] = X['Age_Bins'].astype(str)
        
        X = X.drop(columns=['Age'])
        return X
from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ("cat_selector", DataFrameSelector(["Sex","Pclass","Embarked","Age"])),
    ("cat_attributes_eng", CatAttributeEng()),
    ("cat_imputer", SimpleImputer(strategy='most_frequent')),
    ("cat_encoder", OneHotEncoder()),
])

cat_pipeline.fit_transform(train_data)
from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
forest_clf.fit(X_train, y_train)
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)
titanic_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
titanic_submission.to_csv('titanic_submission.csv', index=False)