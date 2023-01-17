# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading data

data_path = "../input/housing.csv"

housing = pd.read_csv(data_path)



# see the basic info

housing.info()

housing.head(10)
housing.describe()
# boxplot 

housing.boxplot(['median_house_value'], figsize=(10, 10))
# histogram

housing.hist(bins=50, figsize=(15, 15))
housing['ocean_proximity'].value_counts()
op_count = housing['ocean_proximity'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(op_count.index, op_count.values, alpha=0.7)

plt.title('Ocean Proximity Summary')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Ocean Proximity', fontsize=12)

plt.show()

# housing['ocean_proximity'].value_counts().hist()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set), "train + ", len(test_set), "test")
# check whether the test set is the same for every run

test_set.head(10)
housing['median_income'].hist()
housing['income_cat'] = np.ceil(housing['median_income']/1.5)

# DataFrame.where(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False, raise_on_error=None)

# Where cond is True, keep the original value. Where False, replace with corresponding value from other

housing['income_cat'].where(housing['income_cat']<5, 5.0, inplace=True)

housing['income_cat'].hist()
# stratified sampling based on income categories

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]



strat_test_set.head(10)
housing['income_cat'].value_counts() / len(housing)
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
# we need to do the random sampling again to include income_cat column

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



test_set['income_cat'].value_counts() / len(test_set)
# drop the income_cat attributes

for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)
# check the dropping result

strat_test_set.info()
housing = strat_train_set.copy()

housing.info()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# option s: radius of each circle represent the population/100

# option c: color represents the median price

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 

    s=housing['population']/100, label='population', figsize=(10,7), 

    c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# Anscombe's quartet: https://seaborn.pydata.org/examples/anscombes_quartet.html

sns.set(style="ticks")

anscombe = pd.read_csv("../input/anscombe.csv")



# Show the results of a linear regression within each dataset

sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=anscombe,

           col_wrap=2, ci=None, palette="muted", size=4,

           scatter_kws={"s": 50, "alpha": 1})
# Pearson's r, aka, standard correlation coefficient for every pair

corr_matrix = housing.corr()

# Check the how much each attribute correlates with the median house value

corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,12))
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2, figsize=(10,10))
# calculated attributes

housing['rooms_per_household'] = housing['total_rooms']/housing['households']

housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']

housing['population_per_household'] = housing['population']/housing['households']



# checkout the correlations again

corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
housing.info()
housing = strat_train_set.drop("median_house_value", axis=1) # drop target labels for training set

housing_labels = strat_train_set["median_house_value"].copy() # this is the target label vector

housing.info()
# using Scikit-Learn Imputer

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')



# remove non-numerical attributes for Imputer by making a copy of the dataframe

housing_num = housing.drop('ocean_proximity', axis=1)



imputer.fit(housing_num)  # this computes median for each attributes and store the result in statistics_ variable

imputer.statistics_  # same result as housing_num.median().values
# see attributes with missing values

housing_num.info()
x = imputer.transform(housing_num)  # this is a Numpy array

housing_tr = pd.DataFrame(x, columns=housing_num.columns)  # change a Numpy array to a DataFrame

housing_tr.info()  # no missing values
# Approach 1

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

housing_cat = housing['ocean_proximity']

housing_cat.head()
housing_cat_encoded = encoder.fit_transform(housing_cat)

housing_cat_encoded
print(encoder.classes_)  # '<1H OCEAN' is 0, 'INLAND' is 1, etc.
# Approach 2

# reshape

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()  # don't forget the ()!!!

housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))  # this returns a sparse SciPy matrix

housing_cat_1hot.toarray()  # convert the sparse matrix to numpy array
# Combine Approch 1 and 2 in one shot

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat)

housing_cat_1hot
# A custom transformer

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6  # hardcoded just for this dataset



class CombineAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix]/ X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



            

housing.head()  # note that rooms_per_household, and population_per_household already calculated before
attr_adder = CombineAttributesAdder(add_bedrooms_per_room=False)  # add_bedrooms_per_room is called a hyperparameter

housing_extra_attribs = attr_adder.transform(housing.values)  # housing.values is the numpy N-array representation of the DataFrame

housing_extra_attribs
# check the stats of the training set for feature scaling

housing_tr.describe()
# Transformation Pipeline

# name/estimator pairs for pipeline steps

# each estimator except the last one must be transformers with fit_transform() method

# pipeline fit() calls each fit_transform() of each estimator and fit() for the last estimator

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer',Imputer(strategy='median')),

    ('attribs_adder', CombineAttributesAdder()),

    ('std_scaler', StandardScaler()),

])



housing_num_tr = num_pipeline.fit_transform(housing_num)
# this is the fix to the problem at https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize

# CategoricalEncoder should be used instead of LabelEncoder in the latest version of Scikit-Learn

# Definition of the CategoricalEncoder class, copied from PR #9151.

# Just run this cell, or copy it to your code, do not try to understand it (yet).



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from sklearn.preprocessing import LabelEncoder

from scipy import sparse



class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,

                 handle_unknown='error'):

        self.encoding = encoding

        self.categories = categories

        self.dtype = dtype

        self.handle_unknown = handle_unknown



    def fit(self, X, y=None):

        """Fit the CategoricalEncoder to X.

        Parameters

        ----------

        X : array-like, shape [n_samples, n_feature]

            The data to determine the categories of each feature.

        Returns

        -------

        self

        """



        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:

            template = ("encoding should be either 'onehot', 'onehot-dense' "

                        "or 'ordinal', got %s")

            raise ValueError(template % self.handle_unknown)



        if self.handle_unknown not in ['error', 'ignore']:

            template = ("handle_unknown should be either 'error' or "

                        "'ignore', got %s")

            raise ValueError(template % self.handle_unknown)



        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':

            raise ValueError("handle_unknown='ignore' is not supported for"

                             " encoding='ordinal'")



        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)

        n_samples, n_features = X.shape



        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]



        for i in range(n_features):

            le = self._label_encoders_[i]

            Xi = X[:, i]

            if self.categories == 'auto':

                le.fit(Xi)

            else:

                valid_mask = np.in1d(Xi, self.categories[i])

                if not np.all(valid_mask):

                    if self.handle_unknown == 'error':

                        diff = np.unique(Xi[~valid_mask])

                        msg = ("Found unknown categories {0} in column {1}"

                               " during fit".format(diff, i))

                        raise ValueError(msg)

                le.classes_ = np.array(np.sort(self.categories[i]))



        self.categories_ = [le.classes_ for le in self._label_encoders_]



        return self



    def transform(self, X):

        """Transform X using one-hot encoding.

        Parameters

        ----------

        X : array-like, shape [n_samples, n_features]

            The data to encode.

        Returns

        -------

        X_out : sparse matrix or a 2-d array

            Transformed input.

        """

        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)

        n_samples, n_features = X.shape

        X_int = np.zeros_like(X, dtype=np.int)

        X_mask = np.ones_like(X, dtype=np.bool)



        for i in range(n_features):

            valid_mask = np.in1d(X[:, i], self.categories_[i])



            if not np.all(valid_mask):

                if self.handle_unknown == 'error':

                    diff = np.unique(X[~valid_mask, i])

                    msg = ("Found unknown categories {0} in column {1}"

                           " during transform".format(diff, i))

                    raise ValueError(msg)

                else:

                    # Set the problematic rows to an acceptable value and

                    # continue `The rows are marked `X_mask` and will be

                    # removed later.

                    X_mask[:, i] = valid_mask

                    X[:, i][~valid_mask] = self.categories_[i][0]

            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])



        if self.encoding == 'ordinal':

            return X_int.astype(self.dtype, copy=False)



        mask = X_mask.ravel()

        n_values = [cats.shape[0] for cats in self.categories_]

        n_values = np.array([0] + n_values)

        indices = np.cumsum(n_values)



        column_indices = (X_int + indices[:-1]).ravel()[mask]

        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),

                                n_features)[mask]

        data = np.ones(n_samples * n_features)[mask]



        out = sparse.csc_matrix((data, (row_indices, column_indices)),

                                shape=(n_samples, indices[-1]),

                                dtype=self.dtype).tocsr()

        if self.encoding == 'onehot-dense':

            return out.toarray()

        else:

            return out
# given a list of attributes names, this transformer converts the dataframe to a numpy array

from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
# create two pipelines and use feture union to join them

num_attribs = list(housing_num)

cat_attribs = ['ocean_proximity']



num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_attribs)),

    ('imputer', Imputer(strategy='median')),

    ('attribs_adder', CombineAttributesAdder()),

    ('std_scaler', StandardScaler()),

])



cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('label_binarizer', CategoricalEncoder()),

    # ('label_binarizer', LabelBinarizer()),  # LabelBinarizer does not work this way with last Scikit-Learn

])

housing_num_tr = num_pipeline.fit_transform(housing)

housing_num_tr.shape

num_attribs
housing_cat_tr = cat_pipeline.fit_transform(housing)

housing_cat_tr
from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline),

])



# run the whole pipeline

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape
# Linear Regression

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)  # housing_prepared are independent variables and housing_labels are dependent variables
# test out the linear regression model

some_data = housing.iloc[:5]  # choose the first five observations

some_labels = housing_labels.iloc[:5]

some_data
some_data_prepared = full_pipeline.transform(some_data)

some_data_prepared

print('Actual Prices:', list(some_labels))  # actual prices
# print predicted prices

print('Predicted Prices:', lin_reg.predict(some_data_prepared))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
housing_labels.describe()
# Try Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
# 10-fold cross validation

from sklearn.model_selection import cross_val_score



# for decision tree

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)



# for linear regression

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
print('Scores:', tree_rmse_scores)
print('Mean:', tree_rmse_scores.mean())
print('Standard Deviation:', tree_rmse_scores.std())
print('Scores:', lin_rmse_scores)
print('Mean:', lin_rmse_scores.mean())
print('Standard Deviation:', lin_rmse_scores.std())
# Try Random Forest, which is an Ensemble Learning model

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)



forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

print('Mean:', forest_rmse_scores.mean())
# use GridSearch to find best hyperparameter combinations

from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators':[3, 10, 30], 'max_features': [2, 4, 6, 8]},  # try 3x4=12 combinations

    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features': [2, 3, 4]},  # try 2x3=6 combinations

]



forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')  # each model is trained 5 times, so (12+6)*5 = 80 rounds of training in total

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_  # best parameters
grid_search.best_estimator_  # best estimators
# The importance of the features

feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']

cat_one_hot_attribs = list(encoder.classes_)

attributes = num_attribs = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_  # best model



# see the best rmse on the validation set

best_valiation_score = grid_search.best_score_

best_validation_rmse = np.sqrt(-best_valiation_score)

best_validation_rmse
# see the final rmse on the test set

X_test = strat_test_set.drop('median_house_value', axis=1)

y_test = strat_test_set['median_house_value'].copy()



X_test_prepared = full_pipeline.transform(X_test)  # note DO NOT USE fit_transform!! not need to fit anymore

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse