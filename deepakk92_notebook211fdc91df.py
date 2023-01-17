import numpy as np
import pandas as pd
import gc
import sys
import os
import random
pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
pd.options.display.float_format

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

"""
    Load training data from csv file
"""
def load_training_data(file_name):
    return pd.read_csv(file_name)


"""
    Load house properties data
"""
def load_properties_data(file_name):

    # Helper function for parsing the flag attributes
    def convert_true_to_float(df, col):
        df.loc[df[col] == 'true', col] = '1'
        df.loc[df[col] == 'Y', col] = '1'
        df[col] = df[col].astype(float)

    prop = pd.read_csv(file_name, dtype={
        'propertycountylandusecode': str,
        'hashottuborspa': str,
        'propertyzoningdesc': str,
        'fireplaceflag': str,
        'taxdelinquencyflag': str
    })

    for col in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']:
        convert_true_to_float(prop, col)

    return prop


"""
    Assign better names to all feature columns of 'properties' table
"""
def rename_columns(prop):
    prop.rename(columns={
        'parcelid': 'parcelid',  # Unique identifier of parcels
        'airconditioningtypeid': 'cooling_id',  # type of cooling system (if any), 1~13
        'architecturalstyletypeid': 'architecture_style_id',  # Architectural style of the home, 1~27
        'basementsqft': 'basement_sqft',  # Size of the basement
        'bathroomcnt': 'bathroom_cnt',  # Number of bathrooms (including fractional bathrooms)
        'bedroomcnt': 'bedroom_cnt',  # Number of bedrooms
        'buildingclasstypeid': 'framing_id',  # The building framing type, 1~5
        'buildingqualitytypeid': 'quality_id',  # building condition from best (lowest) to worst (highest)
        'calculatedbathnbr': 'bathroom_cnt_calc',  # Same meaning as 'bathroom_cnt'?
        'decktypeid': 'deck_id',  # Type of deck (if any)
        'finishedfloor1squarefeet': 'floor1_sqft',  # Size of finished living area on first floor
        'calculatedfinishedsquarefeet': 'finished_area_sqft_calc',  # calculated total finished living area
        'finishedsquarefeet12': 'finished_area_sqft',  # Same meaning as 'finished_area_sqft_calc'?
        'finishedsquarefeet13': 'perimeter_area',  # Perimeter living area
        'finishedsquarefeet15': 'total_area',  # Total area
        'finishedsquarefeet50': 'floor1_sqft_unk',  # Same meaning as 'floor1_sqft'?
        'finishedsquarefeet6': 'base_total_area',  # Base unfinished and finished area
        'fips': 'fips',  # Federal Information Processing Standard code
        'fireplacecnt': 'fireplace_cnt',  # Number of fireplaces in the home (if any)
        'fullbathcnt': 'bathroom_full_cnt',  # Number of full bathrooms
        'garagecarcnt': 'garage_cnt',  # Total number of garages
        'garagetotalsqft': 'garage_sqft',  # Total size of the garages
        'hashottuborspa': 'spa_flag',  # Whether the home has a hot tub or spa
        'heatingorsystemtypeid': 'heating_id',  # type of heating system, 1~25
        'latitude': 'latitude',  # latitude of the middle of the parcel multiplied by 1e6
        'longitude': 'longitude',  # longitude of the middle of the parcel multiplied by 1e6
        'lotsizesquarefeet': 'lot_sqft',  # Area of the lot in sqft
        'poolcnt': 'pool_cnt', # Number of pools in the lot (if any)
        'poolsizesum': 'pool_total_size',  # Total size of the pools
        'pooltypeid10': 'pool_unk_1',
        'pooltypeid2': 'pool_unk_2',
        'pooltypeid7': 'pool_unk_3',
        'propertycountylandusecode': 'county_landuse_code',
        'propertylandusetypeid': 'landuse_type_id' ,  # Type of land use the property is zoned for, 25 categories
        'propertyzoningdesc': 'zoning_description',  # Allowed land uses (zoning) for that property
        'rawcensustractandblock': 'census_1',
        'regionidcity': 'city_id',  # City in which the property is located (if any)
        'regionidcounty': 'county_id',  # County in which the property is located
        'regionidneighborhood': 'neighborhood_id',  # Neighborhood in which the property is located
        'regionidzip': 'region_zip',
        'roomcnt': 'room_cnt',  # Total number of rooms in the principal residence
        'storytypeid': 'story_id',  # Type of floors in a multi-story house, 1~35
        'threequarterbathnbr': 'bathroom_small_cnt',  # Number of 3/4 bathrooms
        'typeconstructiontypeid': 'construction_id',  # Type of construction material, 1~18
        'unitcnt': 'unit_cnt',  # Number of units the structure is built into (2=duplex, 3=triplex, etc)
        'yardbuildingsqft17': 'patio_sqft',  # Patio in yard
        'yardbuildingsqft26': 'storage_sqft',  # Storage shed/building in yard
        'yearbuilt': 'year_built',  # The year the principal residence was built
        'numberofstories': 'story_cnt',  # Number of stories or levels the home has
        'fireplaceflag': 'fireplace_flag',  # Whether the home has a fireplace
        'structuretaxvaluedollarcnt': 'tax_structure',
        'taxvaluedollarcnt': 'tax_parcel',
        'assessmentyear': 'tax_year',  # The year of the property tax assessment (2015 for 2016 data)
        'landtaxvaluedollarcnt': 'tax_land',
        'taxamount': 'tax_property',
        'taxdelinquencyflag': 'tax_overdue_flag',  # Property taxes are past due as of 2015
        'taxdelinquencyyear': 'tax_overdue_year',  # Year for which the unpaid propert taxes were due
        'censustractandblock': 'census_2'
    }, inplace=True)


"""
    Convert some categorical variables to 'category' type
    Convert float64 variables to float32
    Note: In LightGBM, negative integer value for a categorical feature will be treated as missing value
"""
def retype_columns(prop):

    def float_to_categorical(df, col):
        df[col] = df[col] - df[col].min()  # Convert the categories to have smaller labels (start from 0)
        df.loc[df[col].isnull(), col] = -1
        df[col] = df[col].astype(int).astype('category')

    list_float2categorical = ['cooling_id', 'architecture_style_id', 'framing_id',
                             'heating_id', 'county_id', 'construction_id', 'fips', 'landuse_type_id',
                             'county_landuse_code_id','zoning_description_id']

    # Convert categorical variables to 'category' type, and float64 variables to float32
    for col in prop.columns:
        if col in list_float2categorical:
            float_to_categorical(prop, col)
        elif prop[col].dtype.name == 'float64':
            prop[col] = prop[col].astype(np.float32)

    gc.collect()


"""
    Compute and return datetime aggregate feature tables from a training set
    The returned tables can be joined for both training and inference
"""
def compute_datetime_aggregate_features(train):
    # Add temporary year/month/quarter columns
    dt = pd.to_datetime(train.transactiondate).dt
    train['year'] = dt.year
    train['month'] = dt.month
    train['quarter'] = dt.quarter

    # Median logerror within the category
    logerror_year = train.groupby('year').logerror.median().to_frame() \
                                .rename(index=str, columns={"logerror": "logerror_year"})
    logerror_month = train.groupby('month').logerror.median().to_frame() \
                                .rename(index=str, columns={"logerror": "logerror_month"})
    logerror_quarter = train.groupby('quarter').logerror.median().to_frame() \
                                .rename(index=str, columns={"logerror": "logerror_quarter"})

    logerror_year.index = logerror_year.index.map(int)
    logerror_month.index = logerror_month.index.map(int)
    logerror_quarter.index = logerror_quarter.index.map(int)

    # Drop the temporary columns
    train.drop(['year', 'month', 'quarter'], axis=1, errors='ignore', inplace=True)

    return logerror_year, logerror_month, logerror_quarter


"""
    Add aggregrate datetime features to a feature table
    The input table needs to have a 'transactiondate' columns
    The 'transactiondate' column is deleted from the table in the end
"""
def add_datetime_aggregate_features(df, logerror_year, logerror_month, logerror_quarter):
    # Add temporary year/month/quarter columns
    dt = pd.to_datetime(df.transactiondate).dt
    df['year'] = dt.year
    df['month'] = dt.month
    df['quarter'] = dt.quarter

    # Join the aggregate features
    df = df.merge(how='left', right=logerror_year, on='year')
    df = df.merge(how='left', right=logerror_month, on='month')
    df = df.merge(how='left', right=logerror_quarter, on='quarter')

    # Drop the temporary columns
    df = df.drop(['year', 'month', 'quarter', 'transactiondate'], axis=1, errors='ignore')
    return df


"""
    Add simple 'year', 'month', and 'quarter' categorical features to a DataFrame
"""
def add_simple_datetime_features(df):
    dt = pd.to_datetime(df.transactiondate).dt
    df['year'] = (dt.year - 2016).astype(int)
    df['month'] = (dt.month).astype(int)
    df['quarter'] = (dt.quarter).astype(int)
    df.drop(['transactiondate'], axis=1, inplace=True)


"""
    Look at how complete (i.e. no missing value) each feature is
"""
def print_complete_percentage(df):
    complete_percent = []
    total_cnt = len(df)
    for col in df.columns:
        complete_cnt = total_cnt - (df[col].isnull()).sum()
        complete_cnt -= (df[col] == -1).sum()
        complete_percent.append((col, complete_cnt * 1.00 / total_cnt))
    complete_percent.sort(key=lambda x: x[1], reverse=True)
    for col, percent in complete_percent:
        print("{}: {}".format(col, percent))
%%time
# Load in properties data
prop_2016 = load_properties_data("/kaggle/input/zillow-prize-1/properties_2016.csv")
prop_2017 = load_properties_data("/kaggle/input/zillow-prize-1/properties_2017.csv")

assert len(prop_2016) == len(prop_2017)
print("Number of properties: {}".format(len(prop_2016)))
print("Number of property features: {}".format(len(prop_2016.columns)-1))
def get_landuse_code_df(prop_2016, prop_2017):
    temp = prop_2016.groupby('county_landuse_code')['county_landuse_code'].count()
    landuse_codes = list(temp[temp >= 300].index)
    temp = prop_2017.groupby('county_landuse_code')['county_landuse_code'].count()
    landuse_codes += list(temp[temp >= 300].index)
    landuse_codes = list(set(landuse_codes))
    df_landuse_codes = pd.DataFrame({'county_landuse_code': landuse_codes,
                                     'county_landuse_code_id': range(len(landuse_codes))})
    return df_landuse_codes

def get_zoning_desc_code_df(prop_2016, prop_2017):
    temp = prop_2016.groupby('zoning_description')['zoning_description'].count()
    zoning_codes = list(temp[temp >= 5000].index)
    temp = prop_2017.groupby('zoning_description')['zoning_description'].count()
    zoning_codes += list(temp[temp >= 5000].index)
    zoning_codes = list(set(zoning_codes))
    df_zoning_codes = pd.DataFrame({'zoning_description': zoning_codes,
                                     'zoning_description_id': range(len(zoning_codes))})
    return df_zoning_codes

def process_columns(df, df_landuse_codes, df_zoning_codes):
    df = df.merge(how='left', right=df_landuse_codes, on='county_landuse_code')
    df = df.drop(['county_landuse_code'], axis=1)
    
    df = df.merge(how='left', right=df_zoning_codes, on='zoning_description')
    df = df.drop(['zoning_description'], axis=1)
    
    df.loc[df.county_id == 3101, 'county_id'] = 0
    df.loc[df.county_id == 1286, 'county_id'] = 1
    df.loc[df.county_id == 2061, 'county_id'] = 2
    
    df.loc[df.landuse_type_id == 279, 'landuse_type_id'] = 261
    return df

rename_columns(prop_2016)
rename_columns(prop_2017)

df_landuse_codes = get_landuse_code_df(prop_2016, prop_2017)
df_zoning_codes = get_zoning_desc_code_df(prop_2016, prop_2017)
prop_2016 = process_columns(prop_2016, df_landuse_codes, df_zoning_codes)
prop_2017 = process_columns(prop_2017, df_landuse_codes, df_zoning_codes)

retype_columns(prop_2016)
retype_columns(prop_2017)

prop_2017.head()

train_2016 = load_training_data("/kaggle/input/zillow-prize-1/train_2016_v2.csv")
train_2017 = load_training_data("/kaggle/input/zillow-prize-1/train_2017.csv")

print("Number of 2016 transaction records: {}".format(len(train_2016)))
print("Number of 2017 transaction records: {}".format(len(train_2017)))
print("\n", train_2016.head())
print("\n", train_2017.head())
for prop in [prop_2016, prop_2017]:
    prop['avg_garage_size'] = prop['garage_sqft'] / prop['garage_cnt']
    
    prop['property_tax_per_sqft'] = prop['tax_property'] / prop['finished_area_sqft_calc']
    
    # Rotated Coordinates
    prop['location_1'] = prop['latitude'] + prop['longitude']
    prop['location_2'] = prop['latitude'] - prop['longitude']
    prop['location_3'] = prop['latitude'] + 0.5 * prop['longitude']
    prop['location_4'] = prop['latitude'] - 0.5 * prop['longitude']
    
    # 'finished_area_sqft' and 'total_area' cover only a strict subset of 'finished_area_sqft_calc' in terms of 
    # non-missing values. Also, when both fields are not null, the values are always the same.
    # So we can probably drop 'finished_area_sqft' and 'total_area' since they are redundant
    # If there're some patterns in when the values are missing, we can add two isMissing binary features
    prop['missing_finished_area'] = prop['finished_area_sqft'].isnull().astype(np.float32)
    prop['missing_total_area'] = prop['total_area'].isnull().astype(np.float32)
    prop.drop(['finished_area_sqft', 'total_area'], axis=1, inplace=True)
    prop['missing_bathroom_cnt_calc'] = prop['bathroom_cnt_calc'].isnull().astype(np.float32)
    prop.drop(['bathroom_cnt_calc'], axis=1, inplace=True)
    
    # 'room_cnt' has many zero or missing values
    # On the other hand, 'bathroom_cnt' and 'bedroom_cnt' have few zero or missing values
    # Add an derived room_cnt feature by adding bathroom_cnt and bedroom_cnt
    prop['derived_room_cnt'] = prop['bedroom_cnt'] + prop['bathroom_cnt']
    
    # Average area in sqft per room
    mask = (prop.room_cnt >= 1)  # avoid dividing by zero
    prop.loc[mask, 'avg_area_per_room'] = prop.loc[mask, 'finished_area_sqft_calc'] / prop.loc[mask, 'room_cnt']
    
    # Use the derived room_cnt to calculate the avg area again
    mask = (prop.derived_room_cnt >= 1)
    prop.loc[mask,'derived_avg_area_per_room'] = prop.loc[mask,'finished_area_sqft_calc'] / prop.loc[mask,'derived_room_cnt']
    
prop_2017.head()
def add_aggregate_features(df, group_col, agg_cols):
    df[group_col + '-groupcnt'] = df[group_col].map(df[group_col].value_counts())
    print(df[group_col + '-groupcnt'])
    new_columns = []  # New feature columns added to the DataFrame

    for col in agg_cols:
        aggregates = df.groupby(group_col, as_index=False)[col].agg([np.mean])
        aggregates.columns = [group_col + '-' + col + '-' + s for s in ['mean']]
        new_columns += list(aggregates.columns)
        df = df.merge(how='left', right=aggregates, on=group_col)
        
    for col in agg_cols:
        mean = df[group_col + '-' + col + '-mean']
        diff = df[col] - mean
        
        df[group_col + '-' + col + '-' + 'diff'] = diff
        if col != 'year_built':
            df[group_col + '-' + col + '-' + 'percent'] = diff / mean
        
    # Set the values of the new features to NaN if the groupcnt is too small (prevent overfitting)
    threshold = 100
    df.loc[df[group_col + '-groupcnt'] < threshold, new_columns] = np.nan
    
    # Drop the mean features since they turn out to be not useful
    df.drop([group_col+'-'+col+'-mean' for col in agg_cols], axis=1, inplace=True)
    
    gc.collect()
    return df

group_col = 'region_zip'
agg_cols = ['lot_sqft', 'year_built', 'finished_area_sqft_calc',
            'tax_structure', 'tax_land', 'tax_property', 'property_tax_per_sqft']
prop_2016 = add_aggregate_features(prop_2016, group_col, agg_cols)
prop_2017 = add_aggregate_features(prop_2017, group_col, agg_cols)

prop_2017.head(10)
train_2016 = train_2016.merge(how='left', right=prop_2016, on='parcelid')
train_2017 = train_2017.merge(how='left', right=prop_2017, on='parcelid')
train = pd.concat([train_2016, train_2017], axis=0, ignore_index=True)

print("\nCombined training set size: {}".format(len(train)))

# Add datetime features to training data
add_simple_datetime_features(train)

train.head(10)

def catboost_drop_features(features):
    # id and label (not features)
    unused_feature_list = ['parcelid', 'logerror']

    # too many missing (LightGBM is robust against bad/unrelated features, so this step might not be needed)
    missing_list = ['framing_id', 'architecture_style_id', 'story_id', 'perimeter_area', 'basement_sqft', 'storage_sqft']
    unused_feature_list += missing_list

    # not useful
    bad_feature_list = ['county_landuse_code_id','zoning_description_id','fireplace_flag', 'deck_id', 'pool_unk_1', 'construction_id', 'county_id', 'fips']
    unused_feature_list += bad_feature_list

    

    return features.drop(unused_feature_list, axis=1, errors='ignore')
%%time
# Read DataFrames from hdf5
features_2016 = prop_2016  # All features except for datetime for 2016
features_2017 = prop_2017  # All features except for datetime for 2017
train = train # Concatenated 2016 and 2017 training data with labels

catboost_features = catboost_drop_features(train)
print("Number of features for CatBoost: {}".format(len(catboost_features.columns)))
catboost_features.head(5)


# Specify feature names and categorical features for CatBoost
feature_names = [s for s in catboost_features.columns]
categorical_features = ['cooling_id', 'heating_id', 'landuse_type_id', 'year', 'month', 'quarter']

categorical_indices = []
for i, n in enumerate(catboost_features.columns):
    if n in categorical_features:
        categorical_indices.append(i)
print(categorical_indices)
# Prepare training and cross-validation data
catboost_label = train.logerror.astype(np.float32)
print(catboost_label.head())

# Transform to Numpy matrices
catboost_X = catboost_features.values
catboost_X=catboost_X.astype(object)
for i, n in enumerate(catboost_X):
    for j, m in enumerate(n):
        if j in categorical_indices:
            catboost_X[i,j]=int(catboost_X[i,j])
catboost_y = catboost_label.values

# Perform shuffled train/test split
np.random.seed(42)
random.seed(10)
X_train, X_val, y_train, y_val = train_test_split(catboost_X, catboost_y, test_size=0.2)



print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))

params = {}
params['loss_function'] = 'MAE'
params['eval_metric'] = 'MAE'
params['nan_mode'] = 'Min'  # Method to handle NaN (set NaN to either Min or Max)
params['random_seed'] = 0

params['iterations'] = 1000  # default 1000, use early stopping during training
params['learning_rate'] = 0.17  # default 0.03

params['border_count'] = 254  # default 254 (alias max_bin, suggested to keep at default for best quality)

params['max_depth'] = 6  # default 6 (must be <= 16, 6 to 10 is recommended)
params['random_strength'] = 1  # default 1 (used during splitting to deal with overfitting, try different values)
params['l2_leaf_reg'] = 5  # default 3 (used for leaf value calculation, try different values)
params['allow_const_label'] = True
params['bagging_temperature'] = 1
def transform_test_features(features_2016, features_2017):
    test_features_2016 = catboost_drop_features(features_2016)
    test_features_2017 = catboost_drop_features(features_2017)
    
    test_features_2016['year'] = 0
    test_features_2017['year'] = 1
    
    # 11 and 12 lead to bad results, probably due to the fact that there aren't many training examples for those two
    test_features_2016['month'] = 10
    test_features_2017['month'] = 10
    
    test_features_2016['quarter'] = 4
    test_features_2017['quarter'] = 4
    
    return test_features_2016, test_features_2017

"""
    Helper method that makes predictions on the test set and exports results to csv file
    'models' is a list of models for ensemble prediction (len=1 means using just a single model)
"""
def predict_and_export(models, features_2016, features_2017, file_name):
    # Construct DataFrame for prediction results
    submission_2016 = pd.DataFrame()
    submission_2017 = pd.DataFrame()
    submission_2016['ParcelId'] = features_2016.parcelid
    submission_2017['ParcelId'] = features_2017.parcelid
    
    test_features_2016, test_features_2017 = transform_test_features(features_2016, features_2017)
    
    pred_2016, pred_2017 = [], []
    for i, model in enumerate(models):
        print("Start model {} (2016)".format(i))
        pred_2016.append(model.predict(test_features_2016))
        print("Start model {} (2017)".format(i))
        pred_2017.append(model.predict(test_features_2017))
    
    # Take average across all models
    mean_pred_2016 = np.mean(pred_2016, axis=0)
    mean_pred_2017 = np.mean(pred_2017, axis=0)
    
    submission_2016['201610'] = [float(format(x, '.4f')) for x in mean_pred_2016]
    submission_2016['201611'] = submission_2016['201610']
    submission_2016['201612'] = submission_2016['201610']

    submission_2017['201710'] = [float(format(x, '.4f')) for x in mean_pred_2017]
    submission_2017['201711'] = submission_2017['201710']
    submission_2017['201712'] = submission_2017['201710']
    
    submission = submission_2016.merge(how='inner', right=submission_2017, on='ParcelId')
    
    print("Length of submission DataFrame: {}".format(len(submission)))
    print("Submission header:")
    print(submission.head())
    submission.to_csv(file_name, index=False)
    return submission, pred_2016, pred_2017 
# Remove outliers (if any) from training data
outlier_threshold = 0.4
mask = (abs(catboost_y) <= outlier_threshold)
catboost_X = catboost_X[mask, :]
catboost_y = catboost_y[mask]
print("catboost_X: {}".format(catboost_X.shape))
print("catboost_y: {}".format(catboost_y.shape))

# Train multiple models
bags = 8
models = []
params['iterations'] = 1000
for i in range(bags):
    print("Start training model {}".format(i))
    params['random_seed'] = i
    np.random.seed(42)
    random.seed(36)
    model = CatBoostRegressor(**params)
    model.fit(catboost_X, catboost_y, cat_features=categorical_indices, verbose=False)
    models.append(model)
    
# Sanity check (make sure scores on a small portion of the dataset are reasonable)
for i, model in enumerate(models):
    print("model {}: {}".format(i, abs(model.predict(X_val) - y_val).mean() * 100))


file_name = '/kaggle/working/final_catboost_ensemble_x8.csv'
submission, pred_2016, pred_2017 = predict_and_export(models, features_2016, features_2017, file_name)
import pandas as pd

lgb_single = pd.read_csv('/kaggle/input/lgb-catboost/final_lgb_single.csv')
catboost_x8 = pd.read_csv('/kaggle/input/lgb-catboost/final_catboost_ensemble_x8.csv')
print("Finished Loading the prediction results.")

weight = 0.7
stack = pd.DataFrame()
stack['ParcelId'] = lgb_single['ParcelId']
for col in ['201610', '201611', '201612', '201710', '201711', '201712']:
    stack[col] = weight * catboost_x8[col] + (1 - weight) * lgb_single[col]

print(stack.head())
stack.to_csv('/kaggle/working/final_stack.csv', index=False)