import pandas as pd

import numpy as np
path = '../input/housing/housing.csv'

housing = pd.read_csv(path)
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt 

housing.hist(bins=50, figsize=(20,15))

plt.show()
# Create Income Categories

housing["income_cat"] = pd.cut(housing["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

                               labels=[1, 2, 3, 4, 5])



housing['income_cat'].hist()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# Drop the Income Category



for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude')
# Check the high density population



housing.plot(kind='scatter', x='longitude', y='latitude', alpha = 0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

)

plt.legend()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# Another way of finding correlation



from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()
# housing.dropna(subset=["total_bedrooms"])    # option 1

# housing.drop("total_bedrooms", axis=1)       # option 2



# median = housing["total_bedrooms"].median()  # option 3

# housing["total_bedrooms"].fillna(median, inplace=True)
from sklearn.impute import SimpleImputer
# Impute Numerical Values



imputer = SimpleImputer(strategy="median")



housing_num = housing.drop("ocean_proximity", axis=1)



imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)     # Returns Numpy Array
# Converting back to Pandas



housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_cat = housing[['ocean_proximity']]

housing_cat.head(10)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()



housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

housing_cat_encoded[:10]
ordinal_encoder.categories_
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()



housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_
from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]



        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
# Pipeline to Handle Numerical Data



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)
# Pipeline to Handle Numerical and Text Categorical Data



from sklearn.compose import ColumnTransformer



num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()



lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print(list(some_labels))
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)



lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)         # Scoring function is opposite of MSE, that's why negative sign



lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



display_scores(tree_rmse_scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)



housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
# Cross Validation - Random Forest



forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)



display_scores(forest_rmse_scores)