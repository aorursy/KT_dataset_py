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

housing.head(4)
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



#stratisfied sampling based on income category



for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

    

#income category is no longer needed after the sampling    

for set in (strat_train_set, strat_test_set):

    set.drop(["income_cat"], axis=1, inplace=True)
plt.figure(figsize=(10,7))

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

        s=housing["population"]/100, label="population", figsize=(15,8),

        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,

    )

plt.legend

matrix = housing.corr()

#matrix['median_income'].sort_values()

print(matrix)
matrix.plot.hist()
housing.hist(bins=50, figsize=(20,15))

plt.show()

#CHECK WHAT THESE DRAWINGS MEAN BECAUSE I CANT UNDERSTAND ANYTHING
housing.iloc[[1,2,3,4]] #returns values at given indicies
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
housing = strat_train_set.drop("median_house_value", axis=1) #removing the expected result

housing_labels = strat_train_set["median_house_value"].copy() #and then keeping a copy of these result (so we can compare at the end)
#handling missing values using SimpleImputer class

mySimpleImputer = SimpleImputer(strategy="mean")



#imputers work on numerical values only so we have to remove the last 'object' in our data

housing_numeric = housing.drop('ocean_proximity',axis=1)



#now we can work with the data

mySimpleImputer.fit(housing_numeric)



#this way, it computed the mean for all attributes



res = mySimpleImputer.transform(housing_numeric)



#the result is a numPy array, now we just convert it back into a DataFrame



housing_no_null = pd.DataFrame(res, columns = housing_numeric.columns)

#convering text to numbers

myEncoder = LabelEncoder()

housing_ocean = housing['ocean_proximity']

housing_ocean_encoded = myEncoder.fit_transform(housing_ocean)

#the problem with this is that ML algorithm will assume that two nearby values are related more than two distant values, so we use OneHotEncoder to fix this



myOneHotEncoder = OneHotEncoder()

housing_ocean_encoded = myOneHotEncoder.fit_transform(housing_ocean_encoded.reshape(-1,1)) #reshape because fit_transform expects a 2D array, but we provided an array with one column
#feature scalling 

scaler = MinMaxScaler()

scaler.fit(housing_numeric)

housing_numeric_scalled = scaler.transform(housing_numeric)

housing_numeric_scalled = pd.DataFrame(housing_numeric_scalled, columns=housing_numeric.columns)

print(housing_numeric_scalled)

#why? EDIT: because large number affect the ML algorithm, so, this way, computation becomes faster
#we can do all steps at once using pipelines



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler()),])



housing_numeric_tr = num_pipeline.fit_transform(housing_numeric)



print(housing)
num_attribs = list(housing_numeric)

full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", OneHotEncoder(), ['ocean_proximity']),])



housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)