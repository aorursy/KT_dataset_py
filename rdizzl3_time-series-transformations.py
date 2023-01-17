# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy import stats

from scipy.special import inv_boxcox

from typing import Tuple

import lightgbm as lgb

from datetime import timedelta



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



FIGURE_SIZE = (20, 10)

plt.rcParams['axes.grid'] = True



%matplotlib inline
data = pd.read_pickle('/kaggle/input/m5-full-training-dataset/m5_competition_data.pkl')



# Get sample ids

np.random.seed(1985)

sample_ids = np.random.choice(data['id'].unique(), 50)

data = data.loc[data['id'].isin(sample_ids)]



# Look at the first few observations of the sample

data.head()
# Define power trnnsformations and their inverses

# Square root transformation

def square_root_transformation(x: pd.Series) -> pd.Series:

    return np.sqrt(x)



def square_root_inverse_transformation(x: pd.Series) -> pd.Series:

    return np.square(x)



# Cube root transformation

def cube_root_transformation(x: pd.Series) -> pd.Series:

    return x ** (1 / 3)



def cube_root_inverse_transformation(x: pd.Series) -> pd.Series:

    return x ** 3



# Log transformation

def log_transformation(x: pd.Series) -> pd.Series:

    # Function np.log1p = log(x + 1)

    return np.log1p(x)



def log_inverse_transformation(x: pd.Series) -> pd.Series:

    # Function np.expm1(x) = exp(x) - 1

    return np.expm1(x)



# Box-cox transformation

def box_cox_transformation(x: pd.Series) -> Tuple[np.array, float]:

    x_transformed, lambda_value = stats.boxcox(x)

    return x_transformed, lambda_value

    

def box_cox_inverse_transformation(x: pd.Series, lambda_value: float) -> pd.Series:

    return inv_boxcox(x, lambda_value)
# Square Root

# Apply transformation to the data

data['square_root_transformation_demand'] = data.groupby('id')['demand'].apply(lambda x: square_root_transformation(x))



# Apply the inverse square root transformation

data['square_root_inv_transformation_demand'] = data.groupby('id')['square_root_transformation_demand'].apply(lambda x: square_root_inverse_transformation(x))



# Cube Root

data['cube_root_transformation_demand'] = data.groupby('id')['demand'].apply(lambda x: cube_root_transformation(x))



# Apply the inverse square root transformation

data['cube_root_inv_transformation_demand'] = data.groupby('id')['cube_root_transformation_demand'].apply(lambda x: cube_root_inverse_transformation(x))



# Log Root

data['log_transformation_demand'] = data.groupby('id')['demand'].apply(lambda x: log_transformation(x))



# Apply the inverse square root transformation

data['log_inv_transformation_demand'] = data.groupby('id')['log_transformation_demand'].apply(lambda x: log_inverse_transformation(x))



# Box-cox transformation

box_cox_data = []

box_cox_inverse_transform_lambda_map = {}

for group, group_df in data.groupby('id'):

    box_cox_transformed_data, lambda_value = box_cox_transformation(group_df['demand'] + 1)

    group_df['box_cox_transformation_demand'] = box_cox_transformed_data

    box_cox_data.append(group_df)

    box_cox_inverse_transform_lambda_map.update({group: lambda_value})

    

box_cox_data = pd.concat(box_cox_data)



# Apply inverse transformation

all_power_transformed_data = []

for group, group_df in box_cox_data.groupby('id'):

    lambda_value = box_cox_inverse_transform_lambda_map.get(group)

    group_df['box_cox_inv_transformation_demand'] = box_cox_inverse_transformation(group_df['box_cox_transformation_demand'], lambda_value) - 1

    all_power_transformed_data.append(group_df)

    

all_power_transformed_data = pd.concat(all_power_transformed_data)
def plot_transformations(df: pd.DataFrame, transformation: str) -> None:

    # Get axes for multiple plots

    fig, axes = plt.subplots(nrows=1, ncols=3)

    

    # Original data

    df.set_index('date')['demand'].plot(figsize=FIGURE_SIZE, ax=axes[0], color='blue')

    

    # Transformed data

    transformed_column_name = f'{transformation}_transformation_demand'

    df.set_index('date')[transformed_column_name].plot(figsize=FIGURE_SIZE, ax=axes[1], color='red')

    

    # Inverse Transformed data

    inverse_transformed_data = f'{transformation}_inv_transformation_demand'

    df.set_index('date')[inverse_transformed_data].plot(figsize=FIGURE_SIZE, ax=axes[2], color='orange')
# Get a single id so we can take a look at some plots

single_id = 'FOODS_1_073_TX_1_validation'

single_id_data = all_power_transformed_data.loc[all_power_transformed_data['id'] == single_id]

single_id_data.head()
plot_transformations(single_id_data, 'square_root')
plot_transformations(single_id_data, 'cube_root')
plot_transformations(single_id_data, 'log')
plot_transformations(single_id_data, 'box_cox')
# Before we apply our transformation let's make sure the data is sorted

data = data.sort_values(['id', 'date'])
data['differenced_trasnformation_demand'] = data.groupby('id')['demand'].diff().values

data.head()
data['differenced_demand_filled'] = np.where(pd.isnull(data['differenced_trasnformation_demand']), data['demand'], data['differenced_trasnformation_demand'])

data.head()
data['differenced_inv_transformation_demand'] = data.groupby('id')['differenced_demand_filled'].cumsum()

np.testing.assert_array_equal(data['demand'].values, data['differenced_inv_transformation_demand'].values)
single_id = 'FOODS_1_073_TX_1_validation'

single_id_data = data.loc[data['id'] == single_id]

single_id_data.head()
# Plot of differenced data

single_id_data.set_index('date')['differenced_trasnformation_demand'].plot(figsize=FIGURE_SIZE)
def build_temporal_features(data: pd.DataFrame) -> pd.DataFrame:

    # Temporal features

    data['date'] = pd.to_datetime(data['date'])

    data['year'] = data['date'].dt.year

    data['month'] = data['date'].dt.month

    data['week'] = data['date'].dt.week

    data['day'] = data['date'].dt.day

    data['dayofweek'] = data['date'].dt.dayofweek

    data['quarter'] = data['date'].dt.quarter

    data['week_of_month'] = data['day'].apply(lambda x: np.ceil(x / 7)).astype(np.int8)

    data['is_weekend'] = (data['dayofweek'] > 5).astype(np.int8)

    

    return data
data = build_temporal_features(data)

data.head()
cutoff_date = data['date'].max() - timedelta(days=28)



feature_columns = [

    'year',

    'month',

    'week',

    'day',

    'dayofweek',

    'quarter',

    'week_of_month',

    'is_weekend'

]



target_column = ['differenced_trasnformation_demand']



identifier_columns = ['id', 'date', 'demand']



X_train = data.loc[data['date'] <= cutoff_date]

X_valid = data.loc[data['date'] > cutoff_date]



# Filter X_train, X_valid

X_train = X_train[feature_columns + identifier_columns + target_column]

X_valid = X_valid[feature_columns + identifier_columns + target_column]



# Drop values where we do not have a target value

X_train = X_train.dropna()



# Define target

y_train, y_valid = X_train['differenced_trasnformation_demand'].values, X_valid['differenced_trasnformation_demand'].values
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'boost_from_average': False,

    'verbose': -1,

} 



lgb_train_data, lgb_valid_data = lgb.Dataset(X_train[feature_columns], y_train), lgb.Dataset(X_valid[feature_columns], y_valid)



model = lgb.train(params, lgb_train_data, 200)

X_valid['y_preds'] = model.predict(X_valid[feature_columns])

X_valid.head()
keep_columns = ['id', 'date', 'y_preds']



X_train_last = X_train.groupby('id').last().reset_index()



# We need the same columns to concatenate

X_train_last = X_train_last.rename(columns={'demand': 'y_preds'})

X_train_last = X_train_last[keep_columns]



X_valid = X_valid[keep_columns]



predictions = pd.concat([X_train_last, X_valid], axis=0)

predictions = predictions.sort_values(['id', 'date'])

predictions.head()
predictions['y_preds'] = predictions.groupby('id')['y_preds'].cumsum()

predictions = predictions.reset_index(drop=True)
def mask_first(x: pd.Series) -> np.array:

    result = np.ones_like(x)

    result[0] = 0

    return result



mask = predictions.groupby(['id'])['id'].transform(mask_first).astype(bool)

predictions = predictions.loc[mask]
predictions.head()