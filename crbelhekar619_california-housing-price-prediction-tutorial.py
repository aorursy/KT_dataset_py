#Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

%matplotlib inline
# Loading the dataset
housing  = pd.read_csv('../input/california-housing-prices/housing.csv')
housing.head()
#To get a quick desciption of the data, in particular the total number of rows, and each attribute’s type and number of non-null values 
housing.info()
housing["ocean_proximity"].value_counts()
# To undestand the summary of the numerical attributes.
housing.describe()
# To plot a histogram to understand the data
housing.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots.png")
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()
# The median income is a very important attribute to predict median housing prices.
# We need to ensure that the test set is representative of the various categories of incomes in the whole dataset.
# Therefore, we are creating an income category column to divide median_income is different categories (5 here)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts()
housing["income_cat"].hist()
# Now we need to do stratified sampling based on the income category. 
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# To check whether we have divided the instances in all the income categories proportionally.
housing["income_cat"].value_counts() / len(housing)
# Drop the income_cat column from the datasets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", figsize = (8,6), alpha=0.1)
plt.savefig("visualization_plot.png")
# Now let's take housing prices into consideration
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.title('California housing prices')
plt.legend()
plt.savefig("housing_prices_scatterplot.png")
# If you are aware of California map, you can see that the housing prices are high near the coastal area.

# Optional
import matplotlib.image as mpimg
california_img=mpimg.imread('../input/california-housing-feature-engineering/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.savefig("california_housing_prices_plot.png")
plt.show()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# Using the corr_matrix, we can see the attributes that are likely to be correlate. 

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig("scatter_matrix_plot.png")
# The median house value seems to be highly correlated to the median_income among others.

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.savefig("income_vs_house_value_scatterplot.png")
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
# Finding the correlation
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)]
sample_incomplete_rows.head()
#housing.dropna(subset=["total_bedrooms"]) # option 1
#housing.drop("total_bedrooms", axis=1) # option 2
#housing["total_bedrooms"].fillna(housing["total_bedrooms"].median()) # option 3

# We are using Scikit-Learn's Imputer function here.

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

#Remove the text attribute because median can only be calculated on numerical attributes
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)

# The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable.
imputer.statistics_
#The trained imputer can transform the training set by replacing missing values by the learned medians
X = imputer.transform(housing_num)
# X is a Numpy Array, change it to dataframe using pandas DataFrame function. 
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing.index)
housing_tr.head()
from sklearn.preprocessing import OneHotEncoder

housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
from sklearn.preprocessing import FunctionTransformer

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape