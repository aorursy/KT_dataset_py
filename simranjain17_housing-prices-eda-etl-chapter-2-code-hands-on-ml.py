import pandas as pd
housing = pd.read_csv("../input/housing-handsonml-chapter2/housing.csv")
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
%matplotlib inline 
#only in a Jupyter notebook
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
#From scratch methid to split data based on length
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
#better way to split data using hash values - this needs a distinct identifier column, for which index or 
#combination of lat and longitude can be used

import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
#best package for test set splitting
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#creatinf 5 categories of meidam income for stratified splitting
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
#stratified splitting
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts() / len(housing)
#removing the additionam median value category column that was added
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
#copying the train set to work, so that OG train set is not disturbed
housing = strat_train_set.copy()
#gepgraphical scatter plot
housing.plot(kind="scatter", x="longitude", y="latitude")
#setting alpha for better visualisation
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#The radius of each circle represents the district’s population and the color represents the price . Using predefined color map to map housing prices
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,) 

plt.legend()
#Correlation Matrix
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
#using pandas plot to certain more correlated attributes to check visually
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
#zooming on the one with mediam income and house value
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
#creating custom variables to check if they are better than the existing ones
#One should always reiterating this step when some pattern is noticed
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
#Seeing correlations again to check if customs performed better
#bedrooms per room seems better clearly than just bedrooms or rooms
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
#making x and y - training and label datasets
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#Missing value handling - three options
housing.dropna(subset=["total_bedrooms"]) # option 1 - get rid of NA rows
housing.drop("total_bedrooms", axis=1) # option 2 - get rid of the column itself
median = housing["total_bedrooms"].median() # option 3 - impute median values
housing["total_bedrooms"].fillna(median, inplace=True)
#imputer function to imputr median values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
#remiving the categorical value because imputer only works on Numerical values
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
#imputed values
imputer.statistics_
housing_num.median().values
#transforming train set with imputed values
X = imputer.transform(housing_num)
#converting the above numpy array to a dataframe
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
#label encoding the categorial column - ocean proximity
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
print(encoder.classes_)
#one hot encoding better than label, fit_transform() expects a 2D array, but housing_cat_encoded is a 1D array, so we need to reshape it:17
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
#output is a scipy matrix
#coverting to numpy array
housing_cat_1hot.toarray()
# We can apply both transformations (from text categories to integer categories, then from integer categories to one-hot vectors) in one shot using the LabelBinarizer class:
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot

#this returns a dense NumPy array by default. You can get a sparse matrix instead by passing sparse_output=True to the LabelBinarizer constructor.
#Feature Scaling - very important. all columns should be on similar scale. Y does not need to be scaled
'''
Two mwthods - minmax scaler and Standardization
Min-max scaling (many people call this normalization) is quite simple: values are shifted and rescaled
so that they end up ranging from 0 to 1. We do this by subtracting the min value and dividing by the max
minus the min. Scikit-Learn provides a transformer called MinMaxScaler for this. It has a
feature_range hyperparameter that lets you change the range if you don’t want 0–1 for some reason.
Standardization is quite different: first it subtracts the mean value (so standardized values always have a
zero mean), and then it divides by the variance so that the resulting distribution has unit variance. Unlike
min-max scaling, standardization does not bound values to a specific range, which may be a problem for
some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1). However,
standardization is much less affected by outliers. For example, suppose a district had a median income
equal to 100 (by mistake). Min-max scaling would then crush all the other values from 0–15 down to 0–
0.15, whereas standardization would not be much affected. Scikit-Learn provides a transformer called
StandardScaler for standardization.
'''
#Self defined pipeline function
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]
            
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
#if we could feed a Pandas DataFrame directly into our pipeline, instead of having ,
#to first manually extract the numerical columns into a NumPy array. There is nothing in Scikit-Learn to
#handle Pandas DataFrames,19 but we can write a custom transformer for this task

#self defined pipeline function
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#Self defined label binarier class
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
#pipeline code - basically all the preprocessing steps together - imputing, scaling, encoding for both numerical and categorical columns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy = "median")),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler()),
                        ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
                         ('selector', DataFrameSelector(num_attribs)),
                         ('imputer', SimpleImputer(strategy = "median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                        ])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)), 
                         ('label_binarizer', MyLabelBinarizer()),
                        ])

full_pipeline = FeatureUnion(transformer_list = [("num_pipeline", num_pipeline), 
                                                 ("cat_pipeline", cat_pipeline),
                                                ])

# And we can now run the whole pipeline simply:
#to get the finally prepared dataset
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape
#trying linear regression on the transformed data
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#seeing some predictions on test values
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
#seeing MSE of the linear regression model - not good - definitely underfitting- need to try a more complex model 
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#trying decision tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
#seeinf rmse
#overfits excessively giving 0 rmse
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
'''
Cross Validation :Cross-validation is a technique for evaluating ML models by training several ML models on subsets 
of the available input data and evaluating them on the complementary subset of the data. 
Use cross-validation to detect overfitting, ie, failing to generalize a pattern.

You use the k-fold cross-validation method to perform cross-validation. 
In k-fold cross-validation, you split the input data into k subsets of data (also known as folds). 
You train an ML model on all but one (k-1) of the subsets, and then evaluate the model on the subset that was not used for training. 
This process is repeated k times, with a different subset reserved for evaluation (and excluded from training) each time.
'''
#doing cross validation to avoid overfitting and beter analysyse train-test RMSE
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
#decision tree results after cross validation - still performs poorly
display_scores(tree_rmse_scores)
#linear regression with cross validation - still performs poorly
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
#Random forest regression = ensemble/combination of many decision trees to form a better model 
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

#result metrics of the random forest model  - much better than linear regression and decision tree moseld -RMSE is so much less
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
#Random forest with Cross validation
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
#pickle to save models
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")
#Grid search on random forest - to hyper parameter tune the model 
# and find the best parameter combination that gives the most accurate predictions
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
#listing the best parameters
grid_search.best_params_
#finding the nest estimator model
grid_search.best_estimator_
#listing all grid search models with their RMSEs
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
'''
Alternatives to GridSearchCV is 
- Randomised Search CV where it picks random combinations to find bet values
- Ensemble Methods
'''
#listing feature importances of all independent variables
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
#combining feature importances with parameter names and listing in desecending order
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
#applying the final best model that came from grid search
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#printing the final best RMSE - lowest found so far
final_rmse
'''
The idea is to find the model with the right complexity and then keep hyperparameter tuning it to find the best model
'''
'''
present your solution (highlighting what you have
learned, what worked and what did not, what assumptions were made, and what your system’s limitations
are), document everything, and create nice presentations with clear visualizations and easy-to-remember
statements
'''