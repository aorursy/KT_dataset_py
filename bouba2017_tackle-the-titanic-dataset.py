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
df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')

df_gender=pd.read_csv('../input/titanic/gender_submission.csv')
df_train.head()
df_test.head()
df_train.info()
#Check for Null Values

df_train.isnull().sum()
# Let's take a look at the numerical attributes

df_train.describe()
#Let's count the target 

df_train['Survived'].value_counts()
#Now let's take a quick look at all the categorical attributes

df_train['Pclass'].value_counts()
df_train['Sex'].value_counts()
df_train['Embarked'].value_counts()
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self 

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



num_pipeline=Pipeline([

    ('select_numeric', DataFrameSelector(['Age','SibSp','Parch','Fare'])),

    ('imputer', SimpleImputer(strategy='median'))

])
num_pipeline.fit_transform(df_train)
class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder
cat_pipeline=Pipeline([

    ('select_cat', DataFrameSelector(['Pclass', 'Sex', 'Embarked'])),

    ('imputer', MostFrequentImputer()),

    ('cat_encoder', OneHotEncoder(sparse=False)),

])
cat_pipeline.fit_transform(df_train)
from sklearn.pipeline import FeatureUnion

preprocess_pipeline=FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline),

])
X_train=preprocess_pipeline.fit_transform(df_train)

X_train
y_train=df_train['Survived']
from sklearn.svm import SVC



svm_clf=SVC(gamma='auto')

svm_clf.fit(X_train, y_train)
X_test=preprocess_pipeline.transform(df_test)
y_pred=svm_clf.predict(X_test)
from sklearn.model_selection import cross_val_score



svm_scores=cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()
from sklearn.ensemble import RandomForestClassifier



forest_clf=RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores=cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

plt.plot([1]*10, svm_scores, ".")

plt.plot([2]*10, forest_scores, ".")

plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))

plt.ylabel("Accuracy", fontsize=14)

plt.show()