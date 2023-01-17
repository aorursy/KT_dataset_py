# Basic stuff
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from tqdm import tqdm

# Preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# Models, metrics etc
from sklearn.linear_model import LinearRegression as LR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score as cvs
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, utils, callbacks
housing_dir = '../input/california-housing-prices/housing.csv'

data = pd.read_csv(housing_dir)

data
# Load the data
data = pd.read_csv(housing_dir)

# Separate the numerical part of the data
data_num = data.drop('ocean_proximity', axis=1)

# Names of the numerical attributes in the data
attribs_num = [*data_num.columns.tolist(), 'ocean_distance'] # #####

# Name of the only categorical attribute in the data
attribs_cat = ['ocean_proximity']

# I initialize an encoder here only to extract the list of labels in the same order in which they will be given later in the pipeline
encoder = OneHotEncoder()
encoder.fit(data[attribs_cat])
oh_labels = encoder.categories_[0].tolist()

# Names of the attributes added by FeatExpander
new_features = [
    'rooms_per_household',
    'bedrooms_per_household',
    'rooms_per_person',
    'bedrooms_per_person',
    'bedrooms_fraction',
    'people_per_household'
]

# Names of columns needed for reconversion of the numpy array returned by column_transformer back into a DataFrame
columns_tr = [*attribs_num, *new_features, 'center_distance', *oh_labels]

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy() # To make sure that we don't change the original DataFrame
        X_cleaned = X.drop(index = data.query(' `median_house_value` >= 500000 | `housing_median_age` >= 50 ').index.values)
        return X_cleaned

class MyImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['total_bedrooms'].fillna(value=X['total_bedrooms'].median(), inplace=True)
        return X


class FeatExpander(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['bedrooms_per_household'] = X['total_bedrooms'] / X['households']
        X['rooms_per_person'] = X['total_rooms'] / X['population']
        X['bedrooms_per_person'] = X['total_bedrooms'] / X['population']
        X['bedrooms_fraction'] = X['total_bedrooms'] / X['total_rooms']
        X['people_per_households'] = X['population'] / X['households']
        return X

#data_num = data.drop('ocean_proximity', axis=1)
#attribs_num = [*data_num.columns.tolist(), 'ocean_distance']
#attribs_cat = ['ocean_proximity']
#encoder = OneHotEncoder()
#encoder.fit(data[attribs_cat])
#oh_labels = encoder.categories_[0].tolist()
#columns_tr = [*attribs_num, *new_features, 'center_distance', *oh_labels]
#print(columns_tr)
        
class DFConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X_df = pd.DataFrame(X)
        X_df.columns = columns_tr
        return X_df
    
center_NW = [-122.94, 37.04]
center_SE = [-118.915, 33.165]
def calculate_center_distance(long_val, lat_val):
    NW_distance = np.sqrt((center_NW[0]-long_val)**2 + (center_NW[1]-lat_val)**2)
    SE_distance = np.sqrt((center_SE[0]-long_val)**2 + (center_SE[1]-lat_val)**2)
    return np.min([NW_distance, SE_distance])
class CenterDistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['center_distance'] = X.apply(lambda x: calculate_center_distance(x['longitude'], x['latitude']), axis=1)
        return X
        
tqdm.pandas()
land_labels = ['<1H OCEAN', 'INLAND']
data_ocean = data.copy().query(' ocean_proximity == "NEAR OCEAN" | ocean_proximity == "NEAR BAY"').sample(frac=1/20, random_state=42) # skipping island districts, because they will obviously be always very far from the inland ones
def calculate_ocean_distance(long_val, lat_val):
    data_ocean['district_distance'] = data_ocean.apply(lambda x: np.sqrt((long_val-x['longitude'])**2 + (lat_val-x['latitude'])**2), axis=1)
    return data_ocean['district_distance'].min()
class OceanDistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['ocean_distance'] = X.progress_apply(lambda x: calculate_ocean_distance(x['longitude'], x['latitude']) if x['ocean_proximity'] in land_labels else 0, axis=1)
        return X
        
        
        
pipeline_num = Pipeline([
    ('imputer', MyImputer()),
    ('feat_expander', FeatExpander()),
    ('center_distance_calculator', CenterDistanceCalculator()),
    ('scaler', StandardScaler())
])      

column_transformer = ColumnTransformer([
    ('num', pipeline_num, attribs_num),
    ('cat', OneHotEncoder(), attribs_cat)
])

pipeline_full = Pipeline([
    ('outlier_remover', OutlierRemover()),
    ('ocean_distance_calculator', OceanDistanceCalculator()),
    ('column_transformer', column_transformer),
    ('df_converter', DFConverter()),
])

data_tr = pipeline_full.fit_transform(data)
data_tr.corr()['median_house_value']
