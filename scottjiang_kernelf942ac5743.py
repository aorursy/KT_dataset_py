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
import os 
import sklearn
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
# load data
name_list = ["train_raw", "test_raw", "submission"]
file_list = ["train.csv", "test.csv", "gender_submission.csv"]

for name, file in zip(name_list, file_list):
    exec(name + '= pd.read_csv("../input/"+file)')    
del name, file, name_list, file_list
train_raw.info()
train_raw.head(5)
# Strtifiedshuffledsplit by sex 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, validation_index in split.split(train_raw, train_raw["Sex"]):
    train_set = train_raw.loc[train_index]
    validation_set = train_raw.loc[validation_index]    
print(train_set["Sex"].value_counts()/len(train_set))
print(validation_set["Sex"].value_counts()/len(validation_set))
# fetch labels
train_attri = train_set.drop("Survived", axis=1)
train_labels = train_set["Survived"].copy()
# drop useless attributes
drop_id = [0,2,7,9] 
class Drop_useless_attri(BaseEstimator, TransformerMixin):
    def __init__(self, drop_attributes=True):
        self.drop_attributes = drop_attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.drop_attributes:
            set_dropped_without_column = np.delete(X, drop_id, axis=1)
            column_name = np.array(train_attri.columns.values.tolist())
            columns = np.delete(column_name, drop_id).tolist()
            set_dropped = pd.DataFrame(set_dropped_without_column, columns=columns)
            return set_dropped
        else: 
            return X      
drop_attri = Drop_useless_attri(drop_attributes=True)
train_attri_dropped = drop_attri.transform(train_attri.values)
# 对于dropped后的数据 train_attri   test_attri 划分
cat_attri_index = ["Sex", "Embarked"]
train_num = train_attri_dropped.drop(cat_attri_index, axis=1)
train_cat = train_attri_dropped[cat_attri_index]
num_attri = list(train_num)
cat_attri = list(train_cat) 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names]
    
class MultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        one_hot = pd.get_dummies(X,columns=X.columns)
        return one_hot
    
class Fillna_for_cat(BaseEstimator, TransformerMixin):
    def __init__(self, attri_names):
        self.attri_names = attri_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        most = X.describe().loc["top", "Embarked"]
        X[self.attri_names] = pd.DataFrame(X[self.attri_names].fillna(most))
        return X
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attri)),
        ('imputer', sklearn.preprocessing.Imputer(strategy="median")),
        ('standarddcale', StandardScaler())      
        ])    
    
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attri)),
        ('fillna', Fillna_for_cat(["Embarked"])),
        ('1hot_ecoder', MultiLabelBinarizer())
        ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
        ])
    
train_prepared = full_pipeline.fit_transform(train_attri_dropped)

from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score

X = train_prepared
y = train_labels.values

clf = SVC()   
       
clf.fit(X,y)

validation_attri = validation_set.drop("Survived", axis=1)

drop_attri = Drop_useless_attri(drop_attributes=True)
validation_attri_dropped = drop_attri.transform(validation_attri.values)


validation_prepared = full_pipeline.fit_transform(validation_attri_dropped)



validation_prediction = clf.predict(validation_prepared)

validation = validation_set["Survived"]


C = 1 - np.sum(np.abs(validation-validation_prediction))/len(validation)

scores = cross_val_score(clf, X, y, cv=10)

def display_scores(scores):
    print("Scores:\t", scores)
    print("Mean Scores:\t", scores.mean())
    print("Standard Deviation:\t", scores.std())
    
display_scores(scores)    