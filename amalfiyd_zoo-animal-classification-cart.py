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
# Importing additional modules

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
# Creating dataframe selector class

from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names=attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
# Importing data into memory

df_class = pd.read_csv('/kaggle/input/zoo-animal-classification/class.csv')

df_zoo = pd.read_csv('/kaggle/input/zoo-animal-classification/zoo.csv')
df_class.head(5)
df_class.info()
df_class.describe()
# Create animal names to class mapping

df_class['animal_names_arr'] = df_class['Animal_Names'].apply(lambda x: x.split(","))

df_class['animal_names_arr']



class_dict = {}



for ix, row in df_class['animal_names_arr'].iteritems():

    for an in row:

        class_dict[an] = ix
df_zoo.head(5)
df_zoo.info()
df_zoo.describe()
# Checking number of unique values per columns

for col in df_zoo.columns:

    print(col, " NUnique : ", df_zoo[col].nunique())
# Columns definition

unused_cols = ['animal_name']

target_col = ['class_type']

binary_cols = ['hair','feathers','eggs','milk','airborne',

              'aquatic','predator','toothed','backbone','breathes',

              'venomous','fins','tail','domestic','catsize']

ordinal_cols = ['legs']
# Creating Pipeline

# Dropping unused_cols

# df_zoo.drop(unused_cols, axis=1, inplace=True)



# Pipelone 01. Ordinal

pipeline_ord = Pipeline([

    ('selector', DataFrameSelector(ordinal_cols)),

    ('ord', OrdinalEncoder()),

])



p_binary = np.array(df_zoo[binary_cols])

p_ordinal = pipeline_ord.fit_transform(df_zoo)

p_target = df_zoo[target_col]



X = np.c_[p_binary, p_ordinal]

y = np.c_[p_target]
# Create train and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)

print(X_test.shape)
# Modelling

from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()

parameters = {

    'criterion': ['gini','entropy'],

    'splitter' : ['best','random'],

    'min_samples_split' : [2,5,10],

    'min_samples_leaf' : [2,5,10],

    'max_leaf_nodes' : [None,2,5,10],

    'max_depth' : [10,8,6,3],

    'random_state': [42],

}



grid = GridSearchCV(clf, parameters, 

                   n_jobs=-1, cv=3,

                   verbose=1, scoring="accuracy")
# Fit the model

grid.fit(X_train, y_train)
# Printing best parameters

grid.best_params_
# Check grid search score

grid.best_score_
# Get the best clf

best_clf = grid.best_estimator_
# Predict test values

y_pred = best_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)



print("Accuracy : ", acc)