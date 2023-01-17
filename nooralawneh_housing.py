# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
plt.figure(figsize=(10,7))

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

        s=housing["population"]/100, label="population", figsize=(15,8),

        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,

    )

plt.legend

#to know if there is any values that are close to each other 

corr_matrix=housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)



print(corr_matrix)
#seperate the feautures from the responses

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
#print the splited data :

#training dataset , testing dataset

print(len(train_set),"train +", len(test_set),"test")
from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)





for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

 

for set in (strat_train_set, strat_test_set):

    set.drop(["income_cat"], axis=1, inplace=True)
corr_matrix.plot.hist()
housing.hist(bins=50, figsize=(20,15))

plt.show()
#### prepareing data for machine learnin
#seperate features from responses

housing=strat_train_set.drop("median_house_value",axis=1)

housing_label=strat_train_set["median_house_value"].copy()
#handling missing features

housing.dropna(subset=["total_bedrooms"])
#handling missing features



mySimpleImputer = SimpleImputer(strategy="mean")

housing_num = housing.drop('ocean_proximity',axis=1)

mySimpleImputer.fit(housing_num)

mySimpleImputer.statistics_

housing_num.median().values

x=mySimpleImputer.transform(housing_num)
#handling text and categorical attribute 

housing_cat= housing[["ocean_proximity"]]

housing_cat.head(10)
#converting to numbers 

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder=OrdinalEncoder()

housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)

housing_cat_encoded[:10]

#to ensure encoding neutrality 

from sklearn.preprocessing import OneHotEncoder

cat_encoder=OneHotEncoder()

housing_cat_1hot=cat_encoder.fit_transform(housing_cat)

housing_cat_1hot

housing_cat_1hot.toarray()
#custom Transsformers

from sklearn.base import BaseEstimator,TransformerMixin

rooms_ix,household_ix=3,6

class CombinedAttributeAdder(BaseEstimator,TransformerMixin):

    def fit(self,X,y=None):

        return self

    def transform (self,X,y=None):

        rooms_per_houshold=X[:,rooms_ix] / X[:,household_ix]

        return np.c_[X,rooms_per_houshold]



attr_adder= CombinedAttributeAdder()

housing_extra_attribs=attr_adder.transform(housing.values)
#transformation piplines

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline=Pipeline([

    ('imputer',SimpleImputer(strategy="median")),

    ('attribs_adder',CombinedAttributeAdder()),

    ('std_scaler',StandardScaler()),

])

housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer

num_attribs=list(housing_num)

cat_attribs=["ocean_proximity"]

full_pipeline=ColumnTransformer([

    ("num",num_pipeline,num_attribs),

    ("cat",OneHotEncoder(),cat_attribs),

])

housing_prepard=full_pipeline.fit_transform(housing)