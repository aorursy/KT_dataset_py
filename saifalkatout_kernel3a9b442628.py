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



from sklearn.model_selection import StratifiedShuffleSplit

# Any results you write to the current directory are saved as output.
housing = pd.read_csv('../input/california-housing-prices/housing.csv')

#create housing object from data

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits = 1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

#stratification is done and we created test set

housing = strat_train_set.drop("median_house_value", axis = 1)

h_labels = strat_train_set["median_house_value"].copy()

#split the values from the labels

median = housing["total_bedrooms"].median()

housing["total_bedrooms"].fillna(median,inplace=True)

#set the null values of total bedrooms to median

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")

housing_num = housing.drop("ocean_proximity",axis = 1)

imputer.fit(housing_num)

imputer.statistics_

#or handle null values with imputer then preform fit

housin_text = housing[["ocean_proximity"]]

from sklearn.preprocessing import OrdinalEncoder

ordi = OrdinalEncoder()

housing_text_enc = ordi.fit_transform(housin_text)

#we used the ordinal encoder because the one-hot encoder creates unnecessary space

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator,TransformerMixin

rooms_ix, household_ix = 3,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room 

    def fit(self, X,y=None):

        return self # nothing else to do 

    def transform(self, X,y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix] 

        return np.c_[X,rooms_per_household]

#we define combinedattributesadder class 

num_pipeline = Pipeline([

    ('imputer',SimpleImputer(strategy="median")),

    ('attribs_adder',CombinedAttributesAdder()),

    ('std_scaler',StandardScaler()),

])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#we pipeline the number category

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([

    ("num",num_pipeline,num_attribs),

    ("cat",OrdinalEncoder(),cat_attribs)

])

#full pipeline 

housing_finished = full_pipeline.fit_transform(housing)