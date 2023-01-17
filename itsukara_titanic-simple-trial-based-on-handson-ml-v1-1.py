# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
TITANIC_PATH = "../input"
def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
def prep(data_in):
    data_out = data_in.copy()
    data_out["AgeBucket"] = data_out["Age"] // 15 * 15
    data_out["RelativesOnboard"] = data_out["SibSp"] + data_out["Parch"]
    
    data_out['Salutation'] = data_out.Name.str.extract(' ([A-Za-z]+).', expand=False)
    data_out['Salutation'] = data_out['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_out['Salutation'] = data_out['Salutation'].replace('Mlle', 'Miss')
    data_out['Salutation'] = data_out['Salutation'].replace('Ms', 'Miss')
    data_out['Salutation'] = data_out['Salutation'].replace('Mme', 'Mrs')
    data_out['Salutation'] = data_out['Salutation'].fillna("None")
    data_out['Salutation'] = np.where((data_out['Salutation']).isin(['Mr', 'Miss', 'Mrs', 'Master', 'Rare']), data_out['Salutation'], 'None')

    data_out['Cabin_Lett'] = data_out['Cabin'].apply(lambda x: str(x)[0])
    data_out['Cabin_Lett'] = np.where((data_out['Cabin_Lett']).isin(['n', 'C', 'B', 'D', 'E', 'F', 'A']), data_out['Cabin_Lett'], 'None')
    
    return data_out

from sklearn.ensemble import AdaBoostClassifier
adab_clf = AdaBoostClassifier(n_estimators=100)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score

y_train = train_data["Survived"]

num_pipeline_mod2 = Pipeline([
        ("select_numeric_mod2", DataFrameSelector(["AgeBucket", "RelativesOnboard", "Fare"])),
        ("imputer", Imputer(strategy="median"))
    ])

cat_pipeline_mod2 = Pipeline([
        ("select_cat_mod2", DataFrameSelector(["Pclass", "Sex", "Embarked", "Salutation", "Cabin_Lett"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

preprocess_pipeline_mod2 = FeatureUnion(transformer_list=[
        ("num_pipeline_mod2", num_pipeline_mod2),
        ("cat_pipeline_mod2", cat_pipeline_mod2),
    ])

X_train_mod2 = preprocess_pipeline_mod2.fit_transform(prep(train_data))

forest_mod_scores = cross_val_score(adab_clf, X_train_mod2, y_train, cv=10)
forest_mod_scores.mean()
train_data_prep = prep(train_data)
test_data_prep = prep(test_data)
test_data_prep['Cabin_Lett'].value_counts()
train_data_prep['Cabin_Lett'].value_counts()
adab_clf.fit(X_train_mod2, y_train)

X_test_mod2 = preprocess_pipeline_mod2.fit_transform(prep(test_data))
y_test = adab_clf.predict(X_test_mod2)
y_test
test_data['Survived'] = y_test
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index = False)
