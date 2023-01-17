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

data.shape, data.columns
print(data.shape)

data.head()
data['ocean_proximity'].value_counts()
data.isnull().sum()
data.describe()
data.hist(bins=50, figsize=(20,15))

plt.show()
corr_mat = data.corr()

corr_mat
corr_mhv = corr_mat['median_house_value'].sort_values()

corr_mhv
fig, ax = plt.subplots(figsize=(15,10))



sns.violinplot(

    x='ocean_proximity', y='median_house_value',

    inner='box',

    data=data, ax=ax

)

plt.show()
ocean_proximity_df = {

    label: data.query(' `ocean_proximity` == @label ')['median_house_value'].describe()

    for label in set(data['ocean_proximity'].values)

}



ocean_proximity_df = pd.DataFrame(ocean_proximity_df).round(1)



ocean_proximity_df
data.isnull().sum()
print("Before: %i NaNs" % data['total_bedrooms'].isnull().sum())



data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)



print("After: %i NaNs" % data['total_bedrooms'].isnull().sum())

data.hist(bins=50, figsize=(20,15))

plt.show()
print("Before: %i datapoints." % data.shape[0])



data_cleaned_1 = data.drop(

    index = data.query(' `median_house_value` >= 500000 | `housing_median_age` >= 52 | `median_income` >= 15 ').index.values,

    inplace = False

)



print("After: %i datapoints." % data_cleaned_1.shape[0])
class OutlierRemover(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy() # To make sure that we don't change the original DataFrame

        X_cleaned = X.drop(index = X.query(' `median_house_value` >= 500000 | `housing_median_age` >= 52 | `median_income` >= 15 ').index.values)

        return X_cleaned

    

outlier_remover = OutlierRemover()

data_cleaned_2 = outlier_remover.transform(data)

data_cleaned_2.shape
np.all(data_cleaned_1 == data_cleaned_2)
data = outlier_remover.transform(data)
sns.scatterplot(

    x='median_income', y='median_house_value',

    alpha=.4,

    data=data

)
mhv_counts = data['median_house_value'].value_counts().sort_index()

mhv_counts.loc[448000:452000]
x = mhv_counts.index

y = mhv_counts.values



plt.plot(x, y)

plt.show()
print(data['ocean_proximity'].value_counts())



encoder = OneHotEncoder()

#data_op = data[['ocean_proximity']]

op_ohe = encoder.fit_transform(data[['ocean_proximity']]).toarray()



op_ohe, op_ohe.shape
for category_i, category in enumerate(encoder.categories_[0]):

    print(category_i, category, data['ocean_proximity'].value_counts()[category], op_ohe[:,category_i].sum())
for category_i, category in enumerate(encoder.categories_[0]):

    data[category] = op_ohe[:, category_i]
data.loc[:, 'ocean_proximity':]
data_expanded_1 = data.copy()



data_expanded_1['rooms_per_household'] = data_expanded_1['total_rooms'] / data_expanded_1['households']

data_expanded_1['bedrooms_per_household'] = data_expanded_1['total_bedrooms'] / data_expanded_1['households']



data_expanded_1['rooms_per_person'] = data_expanded_1['total_rooms'] / data_expanded_1['population']

data_expanded_1['bedrooms_per_person'] = data_expanded_1['total_bedrooms'] / data_expanded_1['population']



data_expanded_1['bedrooms_fraction'] = data_expanded_1['total_bedrooms'] / data_expanded_1['total_rooms']



data_expanded_1['people_per_household'] = data_expanded_1['population'] / data_expanded_1['households']



data_expanded_1.head()
# Names of the new features/columns

new_features = [

    'rooms_per_household',

    'bedrooms_per_household',

    'rooms_per_person',

    'bedrooms_per_person',

    'bedrooms_fraction',

    'people_per_household'

]

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

    

feat_expander = FeatExpander()

data_expanded_2 = feat_expander.transform(data)



data_expanded_2.head()
np.all(data_expanded_1.index == data_expanded_2.index)
data = feat_expander.fit_transform(data)

data.corr()['median_house_value']
# Re-load the original data

data_original = pd.read_csv(housing_dir)



# Separate the numerical part of the data

data_num = data_original.drop('ocean_proximity', axis=1)



# Names of the numerical attributes in the original data

attribs_num = data_num.columns.tolist()



# Name of the only categorical attribute in the original data

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

columns_tr = [*attribs_num, *new_features, *oh_labels]





class OutlierRemover(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy() # To make sure that we don't change the original DataFrame

        X_cleaned = X.drop(index = X.query(' `median_house_value` >= 500000 | `housing_median_age` >= 52 | `median_income` >= 15 ').index.values)

        return X_cleaned



# I made my own imputer, because Scikit-learn's SimpleImputer returns a numpy array, whereas I prefer working on DataFrames

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

    

# Another custom transformer - just to reconvert NumPy arrays returned by column_transformer back into a DataFrame

class DFConverter(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        X_df = pd.DataFrame(X)

        X_df.columns = columns_tr #

        return X_df



# A pipeline for numerical attributes

pipeline_num = Pipeline([

    ('imputer', MyImputer()),

    ('feat_expander', FeatExpander()),

])





column_transformer = ColumnTransformer([

    ('num', pipeline_num, attribs_num), # For the numerical attributes

    ('cat', OneHotEncoder(), attribs_cat), # For the categorical attribute

])



pipeline_full = Pipeline([

    ('outlier_remover', OutlierRemover()), # Remove the outliers

    ('column_transformer', column_transformer), # Process the numerical attributes and the categorical attribute separately and concatenate them after

    ('df_converter', DFConverter()) # Reconvert the concatenated NumPy array back into a DataFrame

])



data_tr = pipeline_full.fit_transform(data_original)

data_tr.columns
sns.set_style('white')



data.plot(

    x='longitude', y='latitude',

    kind='scatter', figsize=(10,7),

    alpha=.4,

    c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True

)
corr_mhv = data.corr()['median_house_value'].sort_values(ascending=False)

corr_mhv
sns.set_style('darkgrid')



fig, ax = plt.subplots(1,2, figsize=(30,12))



sns.scatterplot(

    x = 'longitude', y = 'median_house_value',

    alpha=.33,

    data=data, ax=ax[0]

)

sns.lineplot(

    x='longitude', y='median_house_value',

    ci=None, color='red', linewidth=1, alpha=.8,

    data=data, ax=ax[0]

)

ax[0].set_title('Linear correlation: %.5f' % (data.corr().loc['median_house_value', 'longitude']))



sns.scatterplot(

    x = 'latitude', y = 'median_house_value',

    alpha=.33,

    data=data, ax=ax[1]

)

sns.lineplot(

    x='latitude', y='median_house_value',

    ci=None, color='red', linewidth=1, alpha=.8,

    data=data, ax=ax[1]

)

ax[1].set_title('Linear correlation: %.5f' % (data.corr().loc['median_house_value', 'latitude']))



plt.show()
sns.set_style('white')



data.plot(

    x='longitude', y='latitude',

    kind='scatter', figsize=(10,7),

    alpha=.4,

    c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True

)
hotspot_NW_long = [0, 0] # correlation, longitude

hotspot_SE_long = [0, 0] # ^

inter_hotspot_long = -120 # longitude



while True:

    data_NW = data.copy().query('longitude < @inter_hotspot_long')

    data_SE = data.copy().query('longitude > @inter_hotspot_long')

    

    # hotspot_NW

    for long_val in np.arange(-124, inter_hotspot_long, .01):

        data_NW['hotspot_NW_long'] = data_NW['longitude'].apply(lambda x: abs(long_val-x))

        correlation = data_NW.corr().loc['hotspot_NW_long', 'median_house_value']

        if abs(correlation)>abs(hotspot_NW_long[0]):

            hotspot_NW_long = [correlation, long_val]

    

    # hotspot_SE

    for long_val in np.arange(inter_hotspot_long, -116, .01):

        data_SE['hotspot_SE_long'] = data_SE['longitude'].apply(lambda x: abs(long_val-x))

        correlation = data_SE.corr().loc['hotspot_SE_long', 'median_house_value']

        if abs(correlation)>abs(hotspot_SE_long[0]):

            hotspot_SE_long = [correlation, long_val]

            

    # inter_hotspot

    new_inter_hotspot_long = (hotspot_NW_long[1]+hotspot_SE_long[1])/2

    if new_inter_hotspot_long!=inter_hotspot_long:

        inter_hotspot_long = new_inter_hotspot_long

    else:

        break



print("NW:\t", hotspot_NW_long)

print("SE:\t", hotspot_SE_long)

print("inter-hotspot:\t", inter_hotspot_long)
hotspot_NW_lat = [0, 0] # correlation, latitude

hotspot_SE_lat = [0, 0] # ^

inter_hotspot_lat = 36 # latitude



while True:

    data_NW = data.copy().query('latitude > @inter_hotspot_lat')

    data_SE = data.copy().query('latitude < @inter_hotspot_lat')

    

    # hotspot_NW

    for lat_val in np.arange(inter_hotspot_lat, inter_hotspot_lat+2, .01):

        data_NW['hotspot_NW_lat'] = data_NW['latitude'].apply(lambda x: abs(lat_val-x))

        correlation = data_NW.corr().loc['hotspot_NW_lat', 'median_house_value']

        if abs(correlation)>abs(hotspot_NW_lat[0]):

            hotspot_NW_lat = [correlation, lat_val]

    

    # hotspot_SE

    for lat_val in np.arange(inter_hotspot_lat-2, inter_hotspot_lat, .01):

        data_SE['hotspot_SE_lat'] = data_SE['latitude'].apply(lambda x: abs(lat_val-x))

        correlation = data_SE.corr().loc['hotspot_SE_lat', 'median_house_value']

        if abs(correlation)>abs(hotspot_SE_lat[0]):

            hotspot_SE_lat = [correlation, lat_val]

            

    # inter_hotspot

    new_inter_hotspot_lat = (hotspot_NW_lat[1]+hotspot_SE_lat[1])/2

    if new_inter_hotspot_lat!=inter_hotspot_lat:

        inter_hotspot_lat = new_inter_hotspot_lat

    else:

        break



print("NW:\t", hotspot_NW_lat)

print("SE:\t", hotspot_SE_lat)

print("inter-hotspot:\t", inter_hotspot_lat)
hotspot_NW = [hotspot_NW_long[1], hotspot_NW_lat[1]] # longitude, latitude

hotspot_SE = [hotspot_SE_long[1], hotspot_SE_lat[1]] # ^



inter_hotspot = [inter_hotspot_long, inter_hotspot_lat] # ^



new_hotspot_NW = [0, 0, 0]   # correlation, longitude, latitue

new_hotspot_SE = [0, 0, 0]   # ^



# hotspot_NW

data_NW = data.copy().query('longitude < @inter_hotspot[0] & latitude > @inter_hotspot[1]')

for long_val in tqdm(np.arange(hotspot_NW[0]-.5, hotspot_NW[0]+.5, .05)):

    for lat_val in np.arange(hotspot_NW[1]-.5, hotspot_NW[1]+.5, .05):

        data_NW['hotspot_NW_distance'] = data_NW.apply(lambda x: np.sqrt((x['longitude']-long_val)**2 + (x['latitude']-lat_val)**2), axis=1)

        correlation = data_NW.corr().loc['hotspot_NW_distance', 'median_house_value']

        if abs(correlation)>abs(new_hotspot_NW[0]):

            new_hotspot_NW = [correlation, long_val, lat_val]

            

# hotspot_SE

data_SE = data.copy().query('longitude > @inter_hotspot[0] & latitude < @inter_hotspot[1]')

for long_val in tqdm(np.arange(hotspot_SE[0]-.5, hotspot_SE[0]+.5, .05)):

    for lat_val in np.arange(hotspot_SE[1]-.5, hotspot_SE[1]+.5, .05):

        data_SE['hotspot_SE_distance'] = data_SE.apply(lambda x: np.sqrt((x['longitude']-long_val)**2 + (x['latitude']-lat_val)**2), axis=1)

        correlation = data_SE.corr().loc['hotspot_SE_distance', 'median_house_value']

        if abs(correlation)>abs(new_hotspot_SE[0]):

            new_hotspot_SE = [correlation, long_val, lat_val]



# inter_hotspot

inter_hotspot = [(new_hotspot_NW[1]+new_hotspot_SE[1])/2, (new_hotspot_NW[2]+new_hotspot_SE[2])/2]



print("inter_hotspot:\t", inter_hotspot)

print("NW:\t", new_hotspot_NW)

print("SE:\t", new_hotspot_SE)
hotspot_NW = new_hotspot_NW[1:] # longitude, latitude

hotspot_SE = new_hotspot_SE[1:] # ^





# hotspot_NW

data_NW = data.copy().query('longitude < @inter_hotspot[0] & latitude > @inter_hotspot[1]')

for long_val in tqdm(np.arange(hotspot_NW[0]-.05, hotspot_NW[0]+.05, .01)):

    for lat_val in np.arange(hotspot_NW[1]-.05, hotspot_NW[1]+.05, .01):

        data_NW['hotspot_NW_distance'] = data_NW.apply(lambda x: np.sqrt((x['longitude']-long_val)**2 + (x['latitude']-lat_val)**2), axis=1)

        correlation = data_NW.corr().loc['hotspot_NW_distance', 'median_house_value']

        if abs(correlation)>abs(new_hotspot_NW[0]):

            new_hotspot_NW = [correlation, long_val, lat_val]

            

# hotspot_SE

data_SE = data.copy().query('longitude > @inter_hotspot[0] & latitude < @inter_hotspot[1]')

for long_val in tqdm(np.arange(hotspot_SE[0]-.05, hotspot_SE[0]+.05, .01)):

    for lat_val in np.arange(hotspot_SE[1]-.05, hotspot_SE[1]+.05, .01):

        data_SE['hotspot_SE_distance'] = data_SE.apply(lambda x: np.sqrt((x['longitude']-long_val)**2 + (x['latitude']-lat_val)**2), axis=1)

        correlation = data_SE.corr().loc['hotspot_SE_distance', 'median_house_value']

        if abs(correlation)>abs(new_hotspot_SE[0]):

            new_hotspot_SE = [correlation, long_val, lat_val]



# inter_hotspot

inter_hotspot = [(new_hotspot_NW[1]+new_hotspot_SE[1])/2, (new_hotspot_NW[2]+new_hotspot_SE[2])/2]

print("inter_hotspot:\t", inter_hotspot)



print("NW:\t", new_hotspot_NW)

print("SE:\t", new_hotspot_SE)
hotspot_NW = new_hotspot_NW[1:] # longitude, latitude

hotspot_SE = new_hotspot_SE[1:] # ^
def calculate_hotspot_distance(long_val, lat_val):

    NW_distance = np.sqrt((hotspot_NW[0]-long_val)**2 + (hotspot_NW[1]-lat_val)**2)

    SE_distance = np.sqrt((hotspot_SE[0]-long_val)**2 + (hotspot_SE[1]-lat_val)**2)

    return np.min([NW_distance, SE_distance])

    



data['hotspot_distance'] = data.apply(lambda x: calculate_hotspot_distance(x['longitude'], x['latitude']), axis=1)
data.corr().loc['median_house_value', 'hotspot_distance']
fig, ax = plt.subplots(1,1, figsize=(15,10))



data.plot(

    x='longitude', y='latitude',

    kind='scatter', figsize=(10,7),

    alpha=.4,

    c='hotspot_distance', cmap=plt.get_cmap('Spectral'), colorbar=True, ax=ax

)

ax.scatter(x=hotspot_NW[0], y=hotspot_NW[1], marker='X', color='k')

ax.scatter(x=hotspot_SE[0], y=hotspot_SE[1], marker='X', color='k')



plt.show()
tqdm.pandas()



land_labels = ['<1H OCEAN', 'INLAND']

data_ocean = data.copy().query(' ocean_proximity == "NEAR OCEAN" | ocean_proximity == "NEAR BAY"').sample(frac=1/20, random_state=42) # skipping island districts, because they will obviously be always very far from the inland ones

def calculate_ocean_distance(long_val, lat_val):

    data_ocean['district_distance'] = data_ocean.apply(lambda x: np.sqrt((long_val-x['longitude'])**2 + (lat_val-x['latitude'])**2), axis=1)

    return data_ocean['district_distance'].min()



data['ocean_distance'] = data.progress_apply(lambda x: calculate_ocean_distance(x['longitude'], x['latitude']) if x['ocean_proximity'] in land_labels else 0, axis=1)
data.corr().loc['median_house_value', 'ocean_distance']
data.corr().loc['median_house_value', ['ocean_distance']+encoder.categories_[0].tolist()]
fig, ax = plt.subplots(1,1, figsize=(15,10))





data.plot(

    x='longitude', y='latitude',

    kind='scatter', figsize=(10,7),

    alpha=.4,

    c='ocean_distance', cmap=plt.get_cmap('Spectral'), colorbar=True, ax=ax

)



plt.show()
scaler = StandardScaler()

num_cols = [col for col in data.columns if col!='median_house_value' and col!='ocean_proximity' and col not in oh_labels]

data[num_cols] = scaler.fit_transform(data[num_cols])

data.describe().round(1)
# Re-load the original data

data_original = pd.read_csv(housing_dir)



# Separate the numerical part of the data

data_num = data_original.drop('ocean_proximity', axis=1)



# Names of the numerical attributes in the original data

attribs_num = [*data_num.columns.tolist(), 'ocean_distance'] # #####



# Name of the only categorical attribute in the original data

attribs_cat = ['ocean_proximity']



# I initialize an encoder here only to extract the list of labels in the same order in which they will be given later in the pipeline

encoder = OneHotEncoder()

encoder.fit(data_original[attribs_cat])

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

columns_tr = [*attribs_num, *new_features, 'hotspot_distance', *oh_labels]





class OutlierRemover(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy() # To make sure that we don't change the original DataFrame

        X_cleaned = X.drop(index = X.query(' `median_house_value` >= 500000 | `housing_median_age` >= 52 | `median_income` >= 15 ').index.values)

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

    

hotspot_NW = [-122.94, 37.04]

hotspot_SE = [-118.915, 33.165]

def calculate_hotspot_distance(long_val, lat_val):

    NW_distance = np.sqrt((hotspot_NW[0]-long_val)**2 + (hotspot_NW[1]-lat_val)**2)

    SE_distance = np.sqrt((hotspot_SE[0]-long_val)**2 + (hotspot_SE[1]-lat_val)**2)

    return np.min([NW_distance, SE_distance])

class HotspotDistanceCalculator(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        X['hotspot_distance'] = X.apply(lambda x: calculate_hotspot_distance(x['longitude'], x['latitude']), axis=1)

        return X

        

tqdm.pandas()

land_labels = ['<1H OCEAN', 'INLAND']

data_ocean = data_original.copy().query(' ocean_proximity == "NEAR OCEAN" | ocean_proximity == "NEAR BAY"').sample(frac=1/20, random_state=42) # skipping island districts, because they will obviously be always very far from the inland ones

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

        

class MyScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        columns_to_norm = [col for col in X.columns if col!='median_house_value']

        X[columns_to_norm] = StandardScaler().fit_transform(X[columns_to_norm])

        return X

        

        

pipeline_num = Pipeline([

    ('imputer', MyImputer()),

    ('feat_expander', FeatExpander()),

    ('hotspot_distance_calculator', HotspotDistanceCalculator()),

    ('scaler', MyScaler())

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



data_original = pd.read_csv(housing_dir)



data_tr = pipeline_full.fit_transform(data_original)
data_tr.describe().round(1)
# Information about correlation of each feature with median_house_value

corr_mhv = data_tr.corr()['median_house_value']



# Set 0: only the original num + OH

set_0 = [a for a in data_original.columns if a!='median_house_value' and a!='ocean_proximity']+oh_labels

# Set 1: absolute correlation above 0.1 (excluding OH) - my original idea

set_1 = [a for a in corr_mhv.index[:-5] if abs(corr_mhv[a])>.1 and a!='median_house_value']

# Set 2: replication of handbook's - original numerical + OH + handbook's combined attributes

set_2 = [a for a in data_original.columns if a!='median_house_value' and a!='ocean_proximity']+['rooms_per_household', 'people_per_household', 'bedrooms_fraction']+oh_labels

# Set 3: absolute correlation above 0.1 + OH - my original idea, but without excluding OH

set_3 = [a for a in corr_mhv.index[:-5] if abs(corr_mhv[a])>.1 and a!='median_house_value']+oh_labels

# Set 4: all attributes

set_4 = [a for a in corr_mhv.index.tolist() if a!='median_house_value']



sets_all = [

    set_0,

    set_1,

    set_2,

    set_3,

    set_4

]



data_X_all = [ data_tr[set_] for set_ in sets_all]



data_y = data_tr['median_house_value']
print("\tLinear Regression:")

lr_cvs_rmse = [] # a list the scores will be written into

for i, data_X in enumerate(data_X_all):

    lr = LR()

    rmse = np.sqrt(-cvs(lr, data_X, data_y, scoring='neg_mean_squared_error', cv=10, n_jobs=-1))

    lr_cvs_rmse.append(rmse)

    print(f"Set {i}:\tMean: {rmse.mean().round(2)}\tStd: {rmse.std().round(2)}")
print("\tLinear Regression:")

for i, data_X in enumerate(data_X_all):

    print(f"Set {i}:")

    for run_i in range(5):

        lr = LR()

        rmse = np.sqrt(-cvs(lr, data_X, data_y, scoring='neg_mean_squared_error', cv=10, n_jobs=-1))

        print(f"\tRun {run_i}:\tMean: {rmse.mean().round(2)}\tStd: {rmse.std().round(2)}")
print("\tDecision Tree Regressor:")

dtr_cvs_rmse = []

for i, data_X in enumerate(data_X_all):

    rmses = 0

    print(f"Set {i}:")

    for run_i in range(5):

        dtr = DTR()

        rmse = np.sqrt(-cvs(dtr, data_X, data_y, scoring='neg_mean_squared_error', cv=10, n_jobs=-1))

        rmses += rmse

        print(f"\tRun {run_i}:\tMean: {rmse.mean().round(2)}\tStd: {rmse.std().round(2)}")

    rmses /= 5

    dtr_cvs_rmse.append(rmses)
print("\tDecision Tree Regressor:")

for i, rmse in enumerate(dtr_cvs_rmse):

    print(f"Set {i}:\tMean: {rmse.mean().round(2)}\tStd: {rmse.std().round(2)}")
print("\tRandom Forest Regressor:")

rfr_cvs_rmse = []

for i, data_X in enumerate(data_X_all):

    rfr = RFR()

    rmse = np.sqrt(-cvs(rfr, data_X, data_y, scoring='neg_mean_squared_error', cv=10, n_jobs=-1))

    rfr_cvs_rmse.append(rmse)

    print(f"Set {i}:\tMean: {rmse.mean().round(2)}\tStd: {rmse.std().round(2)}")
def lr_scheduler(epoch, lr):

    if epoch==110 or epoch==130:

        return lr/3

    else:

        return lr



callbacks_list = [

    callbacks.LearningRateScheduler(lr_scheduler),

    callbacks.ReduceLROnPlateau(factor=.1, monitor='val_loss', patience=3),

    #callbacks.ModelCheckpoint(filepath='model_best.h5', monitor='val_loss', save_best_only=True, save_freq='epoch'),

]



def build_model(n_features):

    model = models.Sequential(layers=[

        layers.Dense(32, activation='relu', kernel_regularizer='l2', input_shape=(n_features,)),

        layers.BatchNormalization(),

        layers.Dropout(.1),

        layers.Dense(64, activation='relu', kernel_regularizer='l2'),

        layers.BatchNormalization(),

        layers.Dropout(.1),

        layers.Dense(64, activation='relu', kernel_regularizer='l2'),

        layers.BatchNormalization(),

        layers.Dropout(.1),

        layers.Dense(64, activation='relu', kernel_regularizer='l2'),

        layers.BatchNormalization(),

        layers.Dropout(.1),

        layers.Dense(64, activation='relu', kernel_regularizer='l2'),

        layers.BatchNormalization(),

        layers.Dropout(.1),

        layers.Dense(1)

    ])

    return model

    



train_data, test_data = tts(data_tr, test_size=.1, random_state=42)
histories = []

models_ = [] # with an underscore (_), because 'models' name is taken by a Keras module



for i, set_ in tqdm(enumerate(sets_all)):

    train_X, train_y = train_data[set_], train_data['median_house_value']

    

    model = build_model(n_features=train_X.shape[1])

    

    model.compile(

        optimizer='rmsprop',

        loss='mse',

        metrics=['mae']

    )



    history = model.fit(

        train_X, train_y,

        validation_split=.1,

        callbacks = callbacks_list,

        epochs=150, batch_size=32,

        shuffle=True,

        verbose=0

    )

    

    histories.append(history)

    models_.append(model)

    
history = histories[1]

epochs = np.arange(1, len(history.history['loss'])+1)

print("epochs:", len(epochs))



train_loss = history.history['loss']

val_loss = history.history['val_loss']

plt.plot(epochs, train_loss, 'r-', label='train_loss')

plt.plot(epochs, val_loss, 'g--', label='val_loss')

plt.legend()

print("Training and validation loss:")

plt.show()



train_mae = history.history['mae']

val_mae = history.history['val_mae']

plt.plot(epochs, train_mae, 'r-', label='train_mae')

plt.plot(epochs, val_mae, 'g--', label='val_mae')

plt.legend()

print("Training and validation MAE:")

plt.show()



lr = history.history['lr']

plt.plot(epochs, lr, 'b--', label='lr')

plt.legend()

print("Learning rate:")

plt.show()
nn_rmse = []



print("\tNeural Network:")

for i, set_ in enumerate(sets_all):

    train_X, train_y = train_data[set_], train_data['median_house_value']

    test_X, test_y = test_data[set_], test_data['median_house_value']

    

    model = models_[i]

    

    train_rmse = np.sqrt(model.evaluate(train_X, train_y, verbose=0)[0])

    test_rmse = np.sqrt(model.evaluate(test_X, test_y, verbose=0)[0])

    

    nn_rmse.append(test_rmse)

    

    print(f"Set {i}:\tTrain: {train_rmse.round(2)}\tTest: {test_rmse.round(2)}")

    
all_cvs_rmse = [

    lr_cvs_rmse,

    dtr_cvs_rmse,

    rfr_cvs_rmse

]



scores_df = pd.DataFrame({

    f'Set {i}': [rmse[i].mean().round(2) for rmse in all_cvs_rmse]+[nn_rmse[i].round(2)] for i in range(len(sets_all))

})

scores_df['Handbook RMSE'] = [69052.46, 71407.69, 50182.30, None]

scores_df.index = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Neural Network']



scores_df