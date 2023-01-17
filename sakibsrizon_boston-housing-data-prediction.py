# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



housing = pd.read_csv("../input/california-housing-prices/housing.csv")

housing.head(5)
housing = pd.read_csv("../input/california-housing-prices/housing.csv")

housing.describe()
housing.hist(bins = 50, figsize = (20,15))

plt.show()
def split_train_test(data, test_ratio):

    np.random.seed(42)

    shuffeled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffeled_indices[:test_set_size]

    train_indices = shuffeled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set), "train +", len(test_set), "test")



import hashlib

def test_set_check(identifier, test_ratio, hash):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio



def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))

    return data.loc[~in_test_set], data.loc[in_test_set]



housing_with_id = housing.reset_index()

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

train_set.head(20)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

housing.head()



from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(housing, housing["income_cat"]):

 strat_train_set = housing.loc[train_index]

 strat_test_set = housing.loc[test_index]

    

housing["income_cat"].value_counts() / len(housing)   
strat_train_set["income_cat"].value_counts() / len(housing)
strat_test_set["income_cat"].value_counts() / len(housing)
# Droped the 'income_cat' colum for restoring the data set

for set in (strat_test_set, strat_train_set):

    set.drop(["income_cat"], axis = 1, inplace = True)
# Backup the train data set 

housing_new = strat_train_set.copy()

# plot the train data set 

housing_new.plot(kind = "scatter", x = "longitude", y = "latitude", alpha =  0.1,

    s = housing_new["population"] /100, label = "population",

    c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True ,)

plt.legend()
# finding Out the corelation coefficient

corr_matrix = housing_new.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)
# Cleaning Data

# Separationg a targeted column to apply cleaning functions

housing_new = strat_train_set.drop("median_house_value", axis = 1)

housing_labels = strat_train_set["median_house_value"].copy()



# Eleminating invalid values

median =  housing_new["total_bedrooms"].median()

housing_new["total_bedrooms"].fillna(median)

# using sklearn imputer for filling up the missing value

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy = "median")

housing_num = housing_new.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

#imputer stores missing values in it's stattistics_ instance.

imputer.statistics_

#implementing imputer to all numarical valuses

#finding out the median value for the dataset to train the imputer model

housing_num.median().values
#applying the imputer model to the new dataset

x = imputer.transform(housing_num)

#this is holding a numpy array. So we need to convert to the pandas dataframe formate

housing_tr = pd.DataFrame(x, columns=housing_num.columns)
#We leftout "oceans_proximity" columns because of text attribute. So now converting this text labels to numbers to calculate the median value

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

housing_cat = housing["ocean_proximity"]

housing_cat_encoded = encoder.fit_transform(housing_cat)

housing_cat_encoded

#showing the encoded text values

print(encoder.classes_)
# Converting this to OneHotEncoder to find out the more similer values in this array

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse = True)

housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

housing_cat_1hot.toarray()
# using LabelBinarizer to convert this sparse matrix to dense matrix

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat) 

housing_cat_1hot
# Custom Transformar 

from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedroom_ix, population_ix, household_ix = 3,4,5,6

    

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

        

    def fit(self, X, y = None):

        return self

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[: , household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedroom_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing_new.values)
# Transformation Pipeline

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer', Imputer(strategy = "median")),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler()),

])



housing_num_tr = num_pipeline.fit_transform(housing_num)
# Using Labelbinizer to transform Cat (text) pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn_features.transformers import DataFrameSelector

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names=attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values

    

# Custom LabelBinarizer 

class MYLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, sparse_output=False):

        self.sparse_output = sparse_output

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        enc = LabelBinarizer(sparse_output=self.sparse_output)

        return enc.fit_transform(X)

    

num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_attribs)),

    ('imputer', Imputer(strategy = "median")),

    ('attr_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler()),

])



cat_pipeline = Pipeline([

    ('selector', DataFrameSelector (cat_attribs)),

    ('label_binarizer', MYLabelBinarizer()),

])



full_pipeline = FeatureUnion( transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

])



# Applying Pipeline to training dataset



housing_perpared = full_pipeline.fit_transform(housing_new)

housing_perpared
housing_perpared.shape
# Linear Regression model appliying to perpared dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(housing_perpared, housing_labels)
housing_labels.iloc[:5]
housing_new.columns
housing_new.iloc[:5]
# prediction with Linear Regression model 



# some_data = housing.iloc[:5]

# some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(housing_new)

print("Predictions: \t", lin_reg.predict(some_data_prepared))