# imports

import os

import pandas as pd
HOUSING_PATH=os.path.join("../input","california-housing-prices")
def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = os.path.join(housing_path,"housing.csv")

    return pd.read_csv(csv_path)
housing = load_housing_data()
housing.head()
housing.columns
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50,figsize=(20,15))

plt.show()
import numpy as np

# Fix the random seed

np.random.seed(42)
# Implementing basic train-test split

def split_train_test_data(total_data, test_frac=0.2):

    # Shuffle the indices of the data randomly

    shuffled_indices = np.random.permutation(len(total_data))

    # Length of test set

    test_set_len = int(test_frac*len(total_data))

    # Test Indices

    test_indices = shuffled_indices[:test_set_len]

    # Train Indices

    train_indices = shuffled_indices[test_set_len:]

    

    return total_data.iloc[train_indices], total_data.iloc[test_indices]
# My split data

train_set, test_set = split_train_test_data(housing)
# Split using Scikit-learn

from sklearn.model_selection import train_test_split



sk_train_set, sk_test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Comparing data splits

train_set.head()
sk_train_set.head()
# A more sophisticated split method: Keep the test set consistent even when new data is added to the total set

# When there's new data added to the set, the test set will definitely have all of the original test data despite shuffling

from zlib import crc32



def test_set_check(data_id, test_frac):

    data_hash_value = crc32(np.int64(data_id)) & 0xffffffff

    return data_hash_value < test_frac * 2**32
def split_by_id(total_data, test_frac, id_column):

    ids = total_data[id_column]

    ids_in_test_set = ids.apply(lambda id_: test_set_check(id_, test_frac))

    

    return total_data[~ids_in_test_set], total_data[ids_in_test_set]
# Create an id field for the dataset



# Option 1) Add the index field

housing_with_id = housing.reset_index()

train_set_stable, test_set_stable = split_by_id(housing_with_id, 0.2, "index")
train_set_stable.head()
# Option 2) Create a more stable id using latitude and longitude

housing_with_id = housing

housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]

train_set_stable, test_set_stable = split_by_id(housing_with_id, 0.2, "id")
train_set_stable.head()
# Create a income category to see how the income values are distributed

housing["income_cat"] = pd.cut(housing["median_income"], 

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

                               labels = [1, 2, 3, 4, 5])



housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
# Original data's distribution across categories

housing["income_cat"].value_counts()/len(housing)
# Stratified test set's distribution across categories

strat_test_set["income_cat"].value_counts()/len(strat_test_set)
# Drop the "income_cat" feature from the stratified datasets



for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)

    set_.drop("id", axis=1, inplace=True)
# Create a copy of the stratified data for analysis

housing = strat_train_set.copy()
housing.plot(kind="scatter", # Type of plot

             x="longitude",

             y="latitude",

             alpha=0.4, # Transparency to visualize data easier

             s=housing["population"]/100, # Scale the points by the population; Gives us a sense of density

             label="Population",

             figsize=(20,14),

             c="median_house_value", # Color code by the house value; Gives us a sense of where homes cost more

             cmap=plt.get_cmap("jet"), # Blue indicates low values and Red indicates high values

             colorbar=True)



plt.legend()
# Compute the correlation of the target variable (median housing price) with all other features

corr_matrix = housing.corr()



corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]



# The scatter matrix shows this relationship visually

scatter_matrix(housing[attributes], figsize=(24,16))
# Of these attributes, the correlation with median income is most interesting

housing.plot(kind="scatter", x="median_income", y="median_house_value",

             alpha=0.3,figsize=(12,8))
# These are some features recommended by the author and as can be seen below, the bedrooms_per_room feature could be useful

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] 

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"] = housing["population"]/housing["households"]
# Recompute the Correlation

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
# Revert to the clean training set for the next steps

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels since this is our training set

housing_labels = strat_train_set["median_house_value"].copy()
# Extracting the rows with missing info

incomplete_rows = housing[housing.isnull().any(axis=1)]

incomplete_rows.head()
incomplete_rows.dropna(subset=["total_bedrooms"]) # Option 1: Drop the rows which have NaNs for this feature
incomplete_rows.drop("total_bedrooms",axis=1).head() # Option 2: Drop the feature completely
median_value = housing["total_bedrooms"].median()

incomplete_rows["total_bedrooms"].fillna(median_value,inplace=True) # Option 3: Fill with something like the median value of the feature

incomplete_rows.head()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_numeric = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_numeric)
# Let's look at the statistics computed by the imputer

imputer.statistics_
# Does it match the statistics of the original data?

housing_numeric.median().values
# Transform the values in the training set

housing_numeric_transformed = imputer.transform(housing_numeric) 
housing_tr = pd.DataFrame(housing_numeric_transformed, columns=housing_numeric.columns,

                          index=housing_numeric.index)
housing_tr.head()
housing_cat = housing[["ocean_proximity"]]

housing_cat.head(10)
from sklearn.preprocessing import OrdinalEncoder



ordinal_encoder = OrdinalEncoder()

housing_cat_enc = ordinal_encoder.fit_transform(housing_cat)

housing_cat_enc[:10]
ordinal_encoder.categories_ # labels run from 0 to 4
from sklearn.preprocessing import OneHotEncoder



cat_encoder = OneHotEncoder()

housing_cat_one_hot = cat_encoder.fit_transform(housing_cat)

housing_cat_one_hot
from sklearn.base import BaseEstimator, TransformerMixin



# index of columns we are interested in

rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6



# Needs 3 functions to be defined without *args or **kwargs

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    

    # Function 1

    def __init__(self, add_bedrooms_per_room=True): 

        self.add_bedrooms_per_room = add_bedrooms_per_room

    

    # Function 2

    def fit(self, X, y=None):

        return self # nothing to do here

    

    # Function 3 : Adding the features we tried out earlier

    def transform(self, X):

        rooms_per_household = X[:,rooms_idx]/X[:,households_idx]

        population_per_household = X[:, population_idx]/X[:, households_idx]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:,bedrooms_idx]/X[:,rooms_idx]

            return np.c_[X, rooms_per_household,population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X,rooms_per_household,population_per_household]

        

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)

        
housing_extra_attribs = pd.DataFrame(housing_extra_attribs,

                                     columns=list(housing.columns) + ["rooms_per_household", "population_per_household"],

                                     index=housing.index)



housing_extra_attribs.head()
# Putting together all the steps of data processing into a pipeline

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler # For feature scaling so that all numeric attributes are in the same range



numeric_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")), # First Do this

    ('attribs_adder', CombinedAttributesAdder()), # Next Do this

    ('std_scaler', StandardScaler()), # Finally do this

])



housing_numeric_tr = numeric_pipeline.fit_transform(housing_numeric)
housing_numeric_tr
from sklearn.compose import ColumnTransformer



numeric_attr = list(housing_numeric)

cat_attr = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

    ("num", numeric_pipeline, numeric_attr), # One pipeline for numeric data

    ("cat", OneHotEncoder(), cat_attr), # One pipeline for categorical data

])



housing_prepared = full_pipeline.fit_transform(housing) # One pipeline to rule them all ;) 
housing_prepared
housing_prepared.shape, housing.shape # Note that the extra columns are the features we added plus 5 for one-hot encoding
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
# Evaluate on a few training examples

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data) # Pre-process the sample in the same way 

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))
from sklearn.metrics import mean_squared_error



# How good is the model on the training set? Look's like it's underfitting quite a bit

lin_train_preds = lin_reg.predict(housing_prepared)



lin_reg_mse = mean_squared_error(lin_train_preds, housing_labels)

lin_reg_rmse = np.sqrt(lin_reg_mse)

lin_reg_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)
tree_train_preds = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(tree_train_preds, housing_labels)

tree_rmse = np.sqrt(tree_mse)

tree_rmse # The Decision Tree is clearly overfitting!
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(housing_prepared, housing_labels)
forest_train_preds = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(forest_train_preds, housing_labels)

forest_rmse = np.sqrt(forest_mse)

forest_rmse # Random Forest seems a lot better!
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, housing_labels,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores: ", scores)

    print("Mean: ", scores.mean())

    print("Standard Deviation: ", scores.std())

    

display_scores(tree_rmse_scores) 
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

                              scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores) # Decision Tree performs worse than Linear Regression!
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                                scoring="neg_mean_squared_error", cv=10, n_jobs=-1)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV



# Specify the parameters you are interesting in trying out

param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]}, # Try a 3x4 combination first

    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}, # Then try a 2x3 combination with bootstrapping set to False

]



forest_reg = RandomForestRegressor(random_state=42)



# Train Across 5 folds so totally (12 + 6) * 5 = 90 rounds of training

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                          scoring="neg_mean_squared_error",

                          return_train_score=True,

                          n_jobs=-1)



grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

    'n_estimators': randint(low=1, high=200),

    'max_features': randint(low=1, high=8 ),

}



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                               n_iter=10, cv=5, scoring="neg_mean_squared_error", # Note that we've asked it to run 10 times

                               random_state=42, n_jobs=-1)

rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = numeric_attr + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True) # Let's look at the features in their order of importance
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

Y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(final_predictions, Y_test)

final_rmse = np.sqrt(final_mse)
final_rmse
# Let's compute a 95% confidence interval on our result

from scipy import stats



confidence=0.95

squared_errors = (final_predictions - Y_test)**2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, 

                         loc=squared_errors.mean(),

                         scale=stats.sem(squared_errors)))



from sklearn.svm import SVR



param_grid_svm = [

    {'kernel': ['linear'], 'C': [10., 30., 100.]},

    {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30.], 

     'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},    

]



svm_reg = SVR()

grid_search_svm = GridSearchCV(svm_reg, param_grid_svm, cv=5, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)

grid_search_svm.fit(housing_prepared, housing_labels)



svm_mse = grid_search_svm.best_score_

svm_rmse = np.sqrt(-svm_mse)

svm_rmse # Looks much worse than the Random Forest Regressor



# Best Parameters for SVM Regressor

grid_search_svm.best_params_



from scipy.stats import expon, reciprocal



param_distribs_svm = {

    'kernel': ['linear', 'rbf'],

    'C': reciprocal(20, 2000),

    'gamma': expon(scale=1.0),

}



svm_reg = SVR()

rnd_search_svm = RandomizedSearchCV(svm_reg, param_distributions=param_distribs_svm,

                                n_iter=26, cv=5, scoring='neg_mean_squared_error', 

                                verbose=2, random_state=42, n_jobs=-1)

rnd_search_svm.fit(housing_prepared, housing_labels)



svm_mse = rnd_search_svm.best_score_

svm_rmse = np.sqrt(-svm_mse)

svm_rmse # Seems a lot better right?



rnd_search_svm.best_params_

# Let's see if we still have this data from the Random Forest we ran a while back

feature_importances
# Method to extract indices of the top-K most important features

def extract_top_k_indices(importances, k):

    # Get the indices we care about

    return np.sort(np.argpartition(importances,-k)[-k:]) 
k=5

top_k_indices = extract_top_k_indices(feature_importances, k)

top_k_indices
feature_names = np.array(attributes)[top_k_indices]

feature_names
sorted(zip(feature_importances, attributes), reverse=True)[:k]
class TopKFeatureSelector(BaseEstimator, TransformerMixin):

    # function 1

    def __init__(self, feature_importances, k):

        self.k = k

        self.feat_imp = feature_importances

    # function 2

    def fit(self, X, y=None):

        self.top_k_indices = extract_top_k_indices(self.feat_imp, self.k)

        return self

    # function 3

    def transform(self, X):

        return X[:, self.top_k_indices]
feat_select_pipeline = Pipeline([

    ('preparation', full_pipeline), # same one as before

    ('feature_selection', TopKFeatureSelector(feature_importances, k)) # New block 

])
housing_top_k_prepared = feat_select_pipeline.fit_transform(housing)

housing_top_k_prepared[:3] 
housing_prepared[0:3, top_k_indices] # Check to see if we got it right
end_to_end_pipeline = Pipeline([

    ('preparation', full_pipeline), # Prepare the Data

    ('feature_selection', TopKFeatureSelector(feature_importances, k)), # Choose the best features

    ('svm_reg', SVR(**rnd_search_svm.best_params_)) # Make Predictions

])
end_to_end_pipeline.fit(housing, housing_labels)
sample_data = housing.iloc[:5]

sample_labels = housing_labels.iloc[:5]



print("Predictions: ", end_to_end_pipeline.predict(sample_data))

print("True Labels: ", list(sample_labels))
param_grid_auto = [{

    'preparation__num__imputer__strategy': ['median', 'most_frequent']

}]



auto_pipeline = Pipeline([

    ('preparation', full_pipeline),

    ('svm_reg', SVR(**rnd_search_svm.best_params_))

])



grid_search_prep = GridSearchCV(auto_pipeline, param_grid_auto, cv=5,

                                scoring='neg_mean_squared_error', verbose=2)

grid_search_prep.fit(housing, housing_labels)
grid_search_prep.best_params_