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
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

#gs_df = pd.read_csv('../input/gender_submission.csv')
test_df

# Age must be median value or other(mean). Not drop rows.

# Name, Ticket, and Cabin are not usable. Maybe

# PassengerId + survived  -> csv file -> submission

# Pclass 1 + 1 = Pclass 2? Maybe not. So it must be categorized, also `Embarked` column.
test_df.describe() 
train_df
class Preprocess(object):

    

    def __init__(self):

        self.age_median = None

        self.fare_median = None

    

    def transform(self, df: pd.DataFrame):

        self.age_median = df['Age'].median()

        self.fare_median = df['Fare'].median()

        

    def fit(self, df: pd.DataFrame):

        if self.age_median is None:

            raise('Must execute transform!')

        if self.fare_median is None:

            raise('Must execute transform!')

            

        df['Age'] = df['Age'].fillna(self.age_median)

        df['Fare'] = df['Fare'].fillna(self.fare_median)

        return df
train_df.describe()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder





class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    """

    Convert a data type of 'Sex' string to boolean 

    """

    def __init__(self):

        pass

        

    def fit(self, X, y=None):

        return self 

    

    def transform(self, X, y=None):

        X['Sex_bin'] = X['Sex'] == 'male'

        

        return X



    

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values

    



num_attribs = ["Age", "SibSp", "Parch", "Fare", "Sex_bin"]

cat_attribs = ["Pclass"]



num_pipeline = Pipeline([

    ('attribs_adder', CombinedAttributesAdder()),

    ('selector', DataFrameSelector(num_attribs)),

    ('std_scaler', StandardScaler()),

])



cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('cat_encoder', OneHotEncoder()),

])
from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

])
from  sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(train_df, test_size=0.2, random_state=42)



pre = Preprocess()

pre.transform(train_set)

# fill and drop rows

train_set = pre.fit(train_set)



# divide data and label

train_x = train_set.drop('Survived', axis=1)

train_label = train_set['Survived'].copy()
# standardizing and selecting columns and onthotencoding

train_prepared = full_pipeline.fit_transform(train_x)
# linear svc

from sklearn.svm import LinearSVC



clf = LinearSVC()



clf.fit(train_prepared, train_label)
# ensemble RFC

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()



clf.fit(train_prepared, train_label)
test_set = pre.fit(test_set)

test_x = test_set.drop('Survived', axis=1)

test_label = test_set['Survived'].copy()



test_prepared = full_pipeline.fit_transform(test_x)
y_pred = clf.predict(test_prepared)
(y_pred == test_label).sum() / len(y_pred)
test_id = test_df['PassengerId'].copy()
test_set = pre.fit(test_df)

test_prepared = full_pipeline.fit_transform(test_set)
test_prepared.toarray()
a = clf.predict(test_prepared)
test_survived = pd.DataFrame(a, columns=['Survived'])
test_survived['PassengerId'] = test_id.values
test_survived
test_id
test_set
test_survived.to_csv('my_first_submission.csv', index=False)