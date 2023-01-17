# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

print("Done 1")



data_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
pd.set_option('display.max_columns', 500) #This is a very handy tool for large column datasets

data.head()
data_type = data.dtypes.reset_index() #Reset index gives an index to the dataframe

data_type.columns=["columns", "data_type"]

data_type.head()



data_type.groupby('data_type').aggregate('count').reset_index()
def isNullCount(data):

    """

    Function to always compute the nullcount of a dataset

    

    Input: Dataframe of Dataset

    Output: Dataframe of columns and number of nulls, A List of Null_columns

    

    """

    data_null = data.isnull().sum().reset_index()

    data_null.columns = ["column", "null_count"]

    null_column = data_null[data_null['null_count'] > 0]['column'].tolist()

    return data_null, null_column



data_null, null_column = isNullCount(data)
data_null_6000 = data_null[data_null["null_count"] > 6000]

#Drop the columns of data_null_6000 from main data

null_6000_list = data_null_6000.column.values



null_list = data_null.column.tolist()



#make a copy of the data befor going forward

data_copy = data.copy()



data_copy_notnull = data_copy.drop([null for null in null_6000_list], axis=1)



new_data = data_copy_notnull.copy()
color = sns.diverging_palette

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(range(new_data.shape[0]), new_data.price_doc, alpha=0.2)

plt.xlabel('index', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(new_data, test_size=0.2, random_state=42)

len(train_set)
prep_train = train_set.drop(['price_doc', 'timestamp'], axis=1)

y_label = train_set['price_doc'].copy()



data_type = prep_train.dtypes.reset_index() #Reset index gives an index to the dataframe

data_type.columns=["columns", "data_type"]

data_type.head()



data_type.groupby('data_type').aggregate('count').reset_index()
#Let's get the columns with numeric values

prep_train_num = prep_train.select_dtypes(include=["int64", "float64"])



#Column with object values

prep_train_obj = prep_train.select_dtypes(include=["object"])
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import LabelBinarizer

from sklearn.pipeline import TransformerMixin

from sklearn.base import BaseEstimator



num_attributes = list(prep_train_num)

cat_attributes = list(prep_train_obj)



from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import Imputer



class NewLabelBinarizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        arrayed = np.array([])

        for i in range(X.shape[1]):

            col = X[:, i].reshape(-1, 1)

            binarizer = LabelBinarizer().fit_transform(col)

            arrayed = np.hstack([arrayed, binarizer]) if arrayed.size else binarizer

        return arrayed

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values

    def fit_transform(self, X, y=None):

        return self.fit(X,y).transform(X)



num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attributes)),

        ('imputer', Imputer(strategy="median"))

    ])



cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(cat_attributes)),

        ('binarizer', NewLabelBinarizer())

        #('cat_binarizer', MultiColumnLabelEncoder())

    ])

    

full_pipeline = FeatureUnion(transformer_list=[

        ('num_pipeline', num_pipeline),

        ('cat_pipeline', cat_pipeline)

    ])
final_prep = full_pipeline.fit_transform(prep_train)
final_prep.shape
#Make RLSME Scorer

def rmsle(predicted, actual):

    return np.sqrt(np.square(np.log(predicted + 1) - np.log(actual + 1)).mean())



from sklearn.metrics import make_scorer

scorer = make_scorer(rmsle, greater_is_better=False)



#Train some model on the data

def display_scores(scores):

    print("Scores:", scores)

    print("Score_mean", scores.mean())

    print("Score_std", scores.std())
from sklearn.model_selection import cross_val_score

import xgboost as xgb



model1 = xgb.XGBRegressor()

scores = cross_val_score(model1, final_prep, y_label, scoring=scorer, cv=10)

display_scores(-scores)
data_test = data_test.drop([null for null in null_6000_list], axis=1)



data_test.isnull().values.any()
median = data_test.median()



test_prep = data_test.fillna(data_test.median())

test_prep = test_prep.fillna(method='pad')
prep_test = test_prep.drop('timestamp', axis=1)

final_test = full_pipeline.fit_transform(prep_test)
final_prep.shape, final_test.shape
train_model1 = model1.fit(final_prep, y_label)
pred1 = train_model1.predict(final_test)
sub = pd.DataFrame(data= {'id': prep_test['id'].ravel()})

sub['price_doc'] = pred1

sub.to_csv("submission.csv", index = False, header = True)