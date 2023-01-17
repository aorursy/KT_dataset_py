import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

housing = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
housing.head()
housing.hist(bins = 50, figsize=(20,15))
plt.show()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
#create strata to separate incomes - an important attribute to predict housing prices
housing['income_cat'] = pd.cut(housing['median_income'],
                              bins = [0. ,1.5, 3.0, 4.5, 6.0, np.inf],
                              labels = [1,2,3,4,5])
housing['income_cat'].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set['income_cat'].value_counts()/len(strat_test_set)
#remove income_cat
for set_ in (strat_train_set,strat_test_set):
    set_.drop('income_cat', axis = 1, inplace = True)
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1)
#result actually resembles california
#plot housing prices
# s - radius of circle, represents districts population
# c - represents price
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.4,
             s = housing['population']/100, label = 'population', figsize = (10,7),
             c = 'median_house_value', cmap = plt.get_cmap('jet'), colorbar = True)
plt.legend()
corr_matrix = housing.corr()
#how much does each attribute correlate with the median housing value
corr_matrix['median_house_value'].sort_values(ascending = False)
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()
#replace missing values using SimpleImupter, replace with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
#median only computed on numerical attributes, copy data without text attribute
housing_num = housing.drop('ocean_proximity', axis = 1)
imputer.fit(housing_num)
#transform training set by replacing missing values with learned medians
X = imputer.transform(housing_num)

#put back into pandas dataframe
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
housing_cat = housing[['ocean_proximity']]
#convert categories from text to numbers
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#view list of categories
ordinal_encoder.categories_
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/ X[:,rooms_ix]
            return np.c_[X, rooms_per_household,population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scalar', StandardScaler())])
housing_num_tr = num_pipeline.fit_transform(housing_num)
