import os  

import time

import numpy as np    #work with numpy arrays

import pandas as pd    #working with dataframes

import seaborn as sns  #rendering beautiful plots

import matplotlib.pyplot as plt #python's basic plotting library

from sklearn.preprocessing import LabelEncoder    #encoding categories to numerical values

#to view plots inline here's a magic function

%matplotlib inline  

import warnings     #to remove any warnings that causes headache

warnings.filterwarnings('ignore')
train=pd.read_csv('../input/X_train.csv')

y=pd.read_csv('../input/y_train.csv')

test=pd.read_csv('../input/X_test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

print("\nX_train shape: {}, X_test shape: {}".format(train.shape, test.shape))

print("y_train shape: {}, submission shape: {}".format(y.shape, sub.shape))
def display_all(df):

    with pd.option_context("display.max_rows",4,"display.max_columns",10):

        display(df)

display_all(train)
list(pd.unique(y['surface']))
plt.figure(figsize=(10,6))

plt.title("Training labels")

ax = sns.countplot(y='surface', data=y)
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def feature_extraction(actual):

    new = pd.DataFrame()

    actual['total_angular_velocity'] = actual['angular_velocity_X'] + actual['angular_velocity_Y'] + actual['angular_velocity_Z']

    actual['total_linear_acceleration'] = actual['linear_acceleration_X'] + actual['linear_acceleration_Y'] + actual['linear_acceleration_Z']

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    

    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    

    def f1(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    def f2(x):

        return np.mean(np.abs(np.diff(x)))

    

    for col in actual.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()

        new[col + '_min'] = actual.groupby(['series_id'])[col].min()

        new[col + '_max'] = actual.groupby(['series_id'])[col].max()

        new[col + '_std'] = actual.groupby(['series_id'])[col].std()

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']

        

        # Change. 1st order.

        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(f2)

        

        # Change of Change. 2nd order.

        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(f1)

        

        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new
def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        

        return mis_val_table_ren_columns

     
train_df = feature_extraction(train)

test_df = feature_extraction(test)

# train_df.to_csv('New_train2.csv')

# test_df.to_csv('New_test2.csv')
missing_values_table(train_df) 
train_df['acc_vs_vel_std']=train_df['acc_vs_vel_std'].fillna(0)

train_df['acc_vs_vel_mean_change_of_abs_change']=train_df['acc_vs_vel_mean_change_of_abs_change'].fillna(0)
missing_values_table(train_df) 
train_df.fillna(0, inplace = True)

test_df.fillna(0, inplace = True)

train_df.replace(-np.inf, 0, inplace = True)

train_df.replace(np.inf, 0, inplace = True)

test_df.replace(-np.inf, 0, inplace = True)

test_df.replace(np.inf, 0, inplace = True)
train_df=train_df.astype('float32')

test_df=test_df.astype('float32')

train_df.info()
le = LabelEncoder()

target = le.fit_transform(y['surface'])    #label Encoding is required to convert names to num
target
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(random_state=42)

from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
# rf = RandomForestClassifier()



# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)



# rf_random.fit(train_df, target)
# best_random = rf_random.best_estimator_

# best_random.get_params()

# pred=best_random.predict(test_df)
# sub['surface'] = le.inverse_transform(pred)

# sub.to_csv('sample2.csv', index=False)
# from sklearn.model_selection import GridSearchCV

# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'bootstrap': [False],

#     'max_depth': [40, 50, 60, 80],

#     'max_features': ['auto','sqrt'],

#     'min_samples_leaf': [1, 2, 3],

#     'min_samples_split': [4, 3, 2],

#     'n_estimators': [500, 900, 1000, 1200]

# }

# # Create a based model

# rf = RandomForestClassifier()

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 3)

# # Fit the grid search to the data

# grid_search.fit(train_df, target)
# best_model=grid_search.best_estimator_

# x=grid_search.best_params_

# best_model.set_params()

# best_model.get_params()
# best_model.fit(train_df,target)
# pred2=best_model.predict(test_df)

# pred2
# sub['surface'] = le.inverse_transform(pred2)

# sub.to_csv('sample-4.csv', index=False)

# sub.head(3)