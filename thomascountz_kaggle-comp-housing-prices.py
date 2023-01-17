! pip install matplotlib numpy pandas scipy scikit-learn
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Load the CSV into a pandas DataFrame

housing = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col="Id")
# View the `head()` (first few rows and columns) of the DataFrame

housing.head()
# Get meta information on the data: data types, column names, etc

housing.info()
housing.columns[housing.isna().any()].tolist()
housing.corr()["SalePrice"].sort_values(ascending=False)
housing.describe()['OverallQual']
housing['OverallQual'].value_counts().sort_index()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["OverallQual"]):

    strat_train_set = housing.iloc[train_index]

    strat_test_set = housing.iloc[test_index]
from sklearn.model_selection import train_test_split



# Compare stratification with random sampling



rand_train_set, rand_test_set = train_test_split(housing, test_size=0.2, random_state=42)



def overall_quality_proprotion(data):

    return data["OverallQual"].value_counts() / len(data)



overall = overall_quality_proprotion(housing)

stratified = overall_quality_proprotion(strat_test_set)

random = overall_quality_proprotion(rand_test_set)



pd.DataFrame({

    'Overall': overall,

    'Stratified': stratified,

    'Random': random,

    'Strat %error': 100 * stratified / overall - 100,

    'Random %error': 100 * random / overall - 100

}).sort_index()
# Save these sets for next time



strat_train_set.to_csv('train_set.csv', index="Id")

strat_test_set.to_csv('test_set.csv', index="Id")
# Load models



# train_set = pd.read_csv('train_set.csv', index_col=0)

# test_set = pd.read_csv('test_set.csv', index_col=0)



train_set = strat_train_set.copy()

test_set = strat_test_set.copy()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# function for comparing different approaches

def score_dataset(features, labels):

    """

    Function for compring different data preparation approaches



    Parameters

    ----------

    features : array_like (same len as labels)

        the features for a given dataset

    labels   : array_like (same len as features)

        the labels for a given dataset



    Returns

    -------

    loss: float or ndarray of floats

        a value in a string

        

    MAE output is non-negative floating point. The best value is 0.0.

    """

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    

    return mean_absolute_error(y_test, preds)
# For this first round of data clean (removing `null` values), we'll need to remove

# any categorical data so that we can evalute each null-cleaning approach

prep_train = train_set.copy().select_dtypes(exclude=['object'])



# Keep track of scores so we can compare them at the end

scores = pd.DataFrame(columns = [

    "approach",

    "score"

])
# Drop rows

dropped_null_rows = prep_train.dropna()



dropped_null_features = dropped_null_rows.drop("SalePrice", axis=1)

dropped_null_labels = dropped_null_rows["SalePrice"].copy()



score = score_dataset(dropped_null_features, dropped_null_labels)

scores = scores.append({"approach": "drop_rows", "score": score}, ignore_index=True)





print("Rows Dropped: ", len(prep_train[prep_train.isna().any(axis=1)]))

print("Total Rows: ", len(prep_train))

print("Score: ", score)
# Drop columns

dropped_null_cols = prep_train.dropna(axis=1)



dropped_null_features = dropped_null_cols.drop("SalePrice", axis=1)

dropped_null_labels = dropped_null_cols["SalePrice"].copy()



score = score_dataset(dropped_null_features, dropped_null_labels)

scores = scores.append({"approach": "drop_cols", "score": score}, ignore_index=True)





print("Columns Dropped: ", len(prep_train.columns[prep_train.isna().any()]))

print("Total Columns: ", len(prep_train.columns))

print("Score: ", score)
# SimpleImputer

from sklearn.impute import SimpleImputer



# We instantiate it with a strategy, in this case, median

imputer = SimpleImputer(strategy="median")



imputed_null_values = pd.DataFrame(imputer.fit_transform(prep_train))



# SimpleImputer removed column names...

imputed_null_values.columns = prep_train.columns



imputed_null_features = imputed_null_values.drop("SalePrice", axis=1)

imputed_null_labels = imputed_null_values["SalePrice"].copy()



score = score_dataset(imputed_null_features, imputed_null_labels)

scores = scores.append({"approach": "median_impute", "score": score}, ignore_index=True)





print("Values Imputed: ", prep_train.isna().sum().sum())

print("Score: ", score)
# SimpleImputer + `_was_null` column



imputer = SimpleImputer(strategy="median")



null_plus = prep_train.copy()



# Add boolean columns for missing values

for col in prep_train.columns[prep_train.isna().any()]:

     null_plus[col + '_was_missing'] = null_plus[col].isnull()



# Impute in place

imputed_null_plus = pd.DataFrame(imputer.fit_transform(null_plus))



# SimpleImputer removed column names...

imputed_null_plus.columns = null_plus.columns



imputed_plus_features = imputed_null_plus.drop("SalePrice", axis=1)

imputed_plus_labels = imputed_null_plus["SalePrice"].copy()



score = score_dataset(imputed_plus_features, imputed_plus_labels)

scores = scores.append({"approach": "median_impute_plus", "score": score}, ignore_index=True)



print("Values Imputed: ", prep_train.isna().sum().sum())

print("Score: ", score)
scores.sort_values(by="score")
# Get list of categorical variables



object_features = train_set.select_dtypes('object').columns; object_features