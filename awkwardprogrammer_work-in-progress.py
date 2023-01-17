# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn, scipy, matplotlib

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

DATA_DIR = '../input/'

TRAIN_FILE_NAME = 'train.csv'

TEST_FILE_NAME = 'test.csv'



def load_titanic_data(file_name = TRAIN_FILE_NAME, data_dir=DATA_DIR):

    return pd.read_csv(os.path.join(data_dir, file_name))



titanic_data = load_titanic_data()
titanic_data.info()
titanic_data.head()
titanic_data.describe()
%matplotlib inline

import matplotlib.pyplot as plt

titanic_data.hist(bins=50, figsize=(20,15))

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



train_set, test_set = train_test_split(titanic_data, test_size=.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)



for train_index, test_index in split.split(titanic_data, titanic_data['Pclass']):

    strat_train_set = titanic_data.loc[train_index]

    strat_test_set = titanic_data.loc[test_index]



print(*(obj['Pclass'].value_counts()/len(obj) for obj in (titanic_data, strat_train_set, strat_test_set, train_set, test_set)))
for set in (strat_train_set, strat_test_set):

    set.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis = 1, inplace=True)
titanic = strat_train_set.copy()
corr_matrix=titanic.corr()

corr_matrix['Survived'].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix



attributes = ['Survived', 'Fare', 'Pclass']

scatter_matrix(titanic[attributes], figsize=(12,8))
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier





class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values

    

class LabelBinarizerWrapper(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.encoders = []

    

    def fit(self, X, y=None):

        for i in np.arange(X.shape[1]):

            self.encoders.append(LabelBinarizer())

            self.encoders[-1].fit(X[:,i])

            

        return self

    

    def transform(self, X):

        return np.concatenate([self.encoders[i].transform(X[:,i]) for i in np.arange(X.shape[1])], axis=1)

    

class CategoricalImputer(BaseEstimator, TransformerMixin):

    STRATEGIES = ('most_frequent')

    def __init__(self, strategy = 'most_frequent'):

        assert(strategy in CategoricalImputer.STRATEGIES)

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)

        return self

    def transform(self, X):

        return X.fillna(self.fill)



titanic = strat_train_set.drop('Survived', axis=1)

titanic_labels = strat_train_set['Survived'].copy()

cat_attribs = ['Sex', 'Embarked']

titanic_num = titanic.drop(cat_attribs, axis=1)



num_attribs = list(titanic_num)



num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attribs)),

        ('imputer', Imputer(strategy='median')),

        ('std_scaler', StandardScaler()),

    ])



cat_pipeline = Pipeline([

        ('imputer', CategoricalImputer()),

        ('selector', DataFrameSelector(cat_attribs)),

        ('label_binarizer', LabelBinarizerWrapper())

   ])



full_pipeline = FeatureUnion(transformer_list=[

        ('num_pipeline', num_pipeline),

        ('cat_pipeline', cat_pipeline),

    ])



titanic_prepared = full_pipeline.fit_transform(titanic)

classifier = RandomForestClassifier(**{

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

}

)

classifier.fit(titanic_prepared, titanic_labels)
from sklearn.metrics import mean_squared_error



X_test = full_pipeline.transform(strat_test_set.drop('Survived', axis=1))

y_test = strat_test_set['Survived'].copy()



final_predictions = classifier.predict(X_test)

print(classifier.score(X_test,y_test))
from pandas import Series

titanic_test_set = load_titanic_data(TEST_FILE_NAME)

test_predictions = classifier.predict(full_pipeline.transform(titanic_test_set))

dataframe = titanic_test_set['PassengerId'].to_frame('PassengerId')

dataframe['Survived'] = Series(test_predictions, index = dataframe.index)

dataframe.to_csv('results.csv',index=False)