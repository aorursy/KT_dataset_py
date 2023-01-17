# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# linear algebra

import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

pd.set_option('display.max_columns', 100)



# Matplotlib for visualization

from matplotlib import pyplot as plt

# display plots in the notebook

%matplotlib inline



# Seaborn for easier visualization

import seaborn as sns

sns.set_style('darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load real estate data from CSV

df = pd.read_csv('/kaggle/input/realestatetycoon/real_estate_data.csv')
# Exploratory Analysis

# Start with Basics

# Dataframe dimensions

df.shape
# Column datatypes

df.dtypes
# Display first 5 observations

df.head()
# Display last 5 rows of data

df.tail()
# Plot numerical distributions

# Plot histogram grid

df.hist(figsize=(14,14), xrot=-45)



# Clear the text "residue"

plt.show()
# Histogram for year_built

df.year_built.hist()

plt.show()
# Summarize numerical features

df.describe()
# Summarize categorical features

df.describe(include=['object'])
# Plot bar plot for each categorical feature

for feature in df.dtypes[df.dtypes == 'object'].index:

    sns.countplot(y=feature, data=df)

    plt.show()
# Segment tx_price by property_type and plot distributions

sns.boxplot(y='property_type', x='tx_price', data=df)
# Filter and display only df.dtypes that are 'object'

df.dtypes[df.dtypes == 'object']
# Segment by property_type and display the means within each class

df.groupby('property_type').mean()
# Segment sqft by sqft and property_type distributions

sns.boxplot(y='property_type', x='sqft', data=df)
# Segment by property_type and display the means and standard deviations within each class

df.groupby('property_type').agg(['mean', 'std'])
# Study Correlations. Calculate correlations between numeric features

correlations = df.corr()
# Change color scheme

sns.set_style("white")



# Make the figsize 10 x 8

plt.figure(figsize=(10,8))



# Plot heatmap of correlations

sns.heatmap(correlations, cmap='RdBu_r')

plt.show()



# Generate a mask for the upper triangle

mask = np.zeros_like(correlations)

mask[np.triu_indices_from(mask)] = 1



# Make the figsize 10 x 8

plt.figure(figsize=(10,8))



# Plot heatmap of annotated correlations

sns.heatmap(correlations * 100,

            cmap='RdBu_r',

            annot=True,

            fmt='.0f')



plt.show()
# Make the figsize 10 x 8

plt.figure(figsize=(10,8))



# Plot heatmap of correlations

sns.heatmap(correlations * 100,

            cmap='RdBu_r',

            annot=True,

            fmt='.0f',

            mask=mask)

plt.show()
# Correlations between two features

df[['beds', 'baths']].corr()
# Data Cleaning

# Drop duplicates

df = df.drop_duplicates()

print( df.shape )
df.head()
# Fix Structural Errors

# Display unique values of 'basement'

print( df.basement.unique() )
# Missing basement values should be 0

df.basement.fillna(0, inplace=True)
# Class distributions for 'roof'

sns.countplot(y='roof', data=df)

plt.show()
# 'composition' should be 'Composition'

df.roof.replace('composition', 'Composition', inplace=True)



# 'asphalt' should be 'Asphalt'

df.roof.replace('asphalt', 'Asphalt', inplace=True)



# 'shake-shingle' and 'asphalt,shake-shingle' should be 'Shake Shingle'

df.roof.replace(['shake-shingle', 'asphalt,shake-shingle'], 'Shake Shingle',

                inplace=True)
# Class distribution for 'roof'

sns.countplot(y='roof', data=df)

plt.show()
# Class distributions for 'exterior_walls'

sns.countplot(y='exterior_walls', data=df)

plt.show()
# 'Rock, Stone' should be 'Masonry'

df.exterior_walls.replace('Rock, Stone', 'Masonry', inplace=True)



# 'Concrete' and 'Block' should be 'Concrete Block'

df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)
# Class distributions for 'exterior_walls'

sns.countplot(y='exterior_walls', data=df)

plt.show()
# Class distributions for 'property_type'

sns.countplot(y='property_type', data=df)

plt.show()
# Filter Unwanted Outliers

# Box plot of 'tx_price' using the Seaborn library

sns.boxplot(df.tx_price)

plt.xlim(0, 1000000) # setting x-axis range to be consistent

plt.show()



# Violin plot of 'tx_price' using the Seaborn library

sns.violinplot('tx_price', data=df)

plt.xlim(0, 1000000) # setting x-axis range to be consistent

plt.show()
# Violin plot of beds

sns.violinplot(df.beds)

plt.show()



# Violin plot of sqft

sns.violinplot(df.sqft)

plt.show()



# Violin plot of lot_size

sns.violinplot(df.lot_size)

plt.show()
# Sort df.lot_size and display the top 5 samples

df.lot_size.sort_values(ascending=False).head()
df[df.lot_size == df.lot_size.max()]
# Remove lot_size outliers

df = df[df.lot_size <= 500000]



# print length of df

print( len(df) )
# Sort df.lot_size and display the top 5 samples

df.lot_size.sort_values(ascending=False).head()
# Handle Missing Data (numeric)

# Display number of missing values by feature (numeric)

df.select_dtypes(exclude=['object']).isnull().sum()
# Display number of missing values by feature (numeric)

df.select_dtypes(exclude=['object']).isnull().sum()
# Handle missing data (Categorical)

# Display number of missing values by feature (categorical)

df.select_dtypes(include=['object']).isnull().sum()
# Fill missing values in exterior_walls with 'Missing'

df['exterior_walls'].fillna('Missing', inplace=True)

df['roof'].fillna('Missing', inplace=True)

# Fill missing categorical values

for column in df.select_dtypes(include=['object']):

    df[column].fillna('Missing', inplace=True)
# Display number of missing values by feature (categorical)

df.select_dtypes(include=['object']).isnull().sum()
# Save cleaned dataframe to new file

df.to_csv('cleaned_df.csv', index=None)
# Output data files are available in the "../output/" directory.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Feature engineering

# Create Interaction Features

# Create indicator variable for properties with 2 beds and 2 baths

df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int)
df.head()
# Display percent of rows where two_and_two == 1

print( df.two_and_two.mean() )
# Create a property age feature

df['property_age'] = df.tx_year - df.year_built
# Should not be less than 0

print( df.property_age.min() )
# Number of observations with 'property_age' < 0

print( sum(df.property_age < 0) )
# Remove rows where property_age is less than 0

df = df[df.property_age >= 0]



# Print number of rows in remaining dataframe

print( len(df) )
# Number of observations with 'property_age' < 0

print( sum(df.property_age < 0) )
# Create a school score feature that num_schools * median_school

df['school_score'] = df.num_schools * df.median_school
# Display median school score

df.school_score.median()
df.school_score.describe()
# Create indicator feature for transactions between 2010 and 2013, inclusive

df['during_recession'] = ((df.tx_year >= 2010) & (df.tx_year <= 2013)).astype(int)
# Set variable a as the earlier indicator variable (combining two masks)

a = ((df.tx_year >= 2010) & (df.tx_year <= 2013)).astype(int)



# Set variable b as the new indicator variable (using "between")

b = df.tx_year.between(2010, 2013).astype(int)



# Are a and b equivalent?

print( all(a == b) )
# Create indicator feature for transactions between 2010 and 2013, inclusive

df['during_recession'] = df.tx_year.between(2010, 2013).astype(int)
# Print percent of transactions where during_recession == 1

print( df.during_recession.mean() )
# Combine Sparse Class

# Bar plot for exterior_walls

sns.countplot(y='exterior_walls', data=df)

plt.show()
# Group 'Wood Siding' and 'Wood Shingle' with 'Wood'

df.exterior_walls.replace(['Wood Siding', 'Wood Shingle'], 'Wood', inplace=True)
# List of classes to group

other_exterior_walls = ['Concrete Block', 'Stucco', 'Masonry', 'Other', 'Asbestos shingle']



# Group other classes into 'Other'

df.exterior_walls.replace(other_exterior_walls, 'Other', inplace=True)
# Bar plot for exterior_walls

sns.countplot(y='exterior_walls', data=df)

plt.show()
print( df.exterior_walls.unique() )
# Display first 5 values of 'exterior_walls'

df.exterior_walls.head()
# Bar plot for roof

sns.countplot(y='roof', data=df)

plt.show()
# Group 'Composition' and 'Wood Shake/ Shingles' into 'Composition Shingle'

df.roof.replace(['Composition', 'Wood Shake/ Shingles'],

                'Composition Shingle', inplace=True)
# List of classes to group

other_roofs = ['Other', 'Gravel/Rock', 'Roll Composition', 'Slate', 'Built-up', 'Asbestos', 'Metal']



# Group other classes into 'Other'

df.roof.replace(other_roofs, 'Other', inplace=True)
# Bar plot for roof

sns.countplot(y='roof', data=df)

plt.show()
# Add dummy variables

# Get dummy variables and display first 5 observations

pd.get_dummies( df, columns=['exterior_walls'] ).head()
# Get dummy variables and display first 5 observations

pd.get_dummies( df, columns=['roof'] ).head()
# Create new dataframe with dummy features

abt = pd.get_dummies(df, columns=['exterior_walls', 'roof', 'property_type'])
abt.head()
print( len(abt.columns) )
# Drop 'tx_year' and 'year_built' from the dataset

abt.drop(['tx_year', 'year_built'], axis=1, inplace=True)
# Save analytical base table

abt.to_csv('analytical_base_table.csv', index=None)
# Output data files are available in the "../output/" directory.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Algorithm Selection

# Import Regularized Regression algos

from sklearn.linear_model import Lasso, Ridge, ElasticNet



# Import Tree Ensemble algos

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Load ABT from Module 3

df = pd.read_csv('analytical_base_table.csv')

print(df.shape)
# Function for splitting training and test set

from sklearn.model_selection import train_test_split
# Create separate object for target variable

y = df.tx_price



# Create separate object for input features

X = df.drop('tx_price', axis=1)
# Split X and y into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   test_size=0.2,

                                                   random_state=1234)
print( len(X_train), len(X_test), len(y_train), len(y_test) )
# Summary statistics of X_train

X_train.describe()
# Function for creating model pipelines

from sklearn.pipeline import make_pipeline
# For standardization

from sklearn.preprocessing import StandardScaler
# Pipeline with Standardization and Lasso Regression

make_pipeline(StandardScaler(), Lasso(random_state=123))
# Create pipelines dictionary

pipelines = {

    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),

    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123))

}
# Add a pipeline for Elastic-Net

pipelines['enet'] = make_pipeline(StandardScaler(), ElasticNet(random_state=123))
# List tuneable hyperparameters of our Lasso pipeline

pipelines['lasso'].get_params()
# Lasso hyperparameters

lasso_hyperparameters = { 

    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 

}



# Ridge hyperparameters

ridge_hyperparameters = { 

    'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  

}
# Elastic Net hyperparameters

enet_hyperparameters = { 

    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],                        

    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  

}
# Create hyperparameters dictionary

hyperparameters = {

    'lasso' : lasso_hyperparameters,

    'ridge' : ridge_hyperparameters,

    'enet' : enet_hyperparameters

}
# Helper for cross-validation

from sklearn.model_selection import GridSearchCV
# Create cross-validation object from Lasso pipeline and Lasso hyperparameters

model = GridSearchCV(pipelines['lasso'], hyperparameters['lasso'], cv=10, n_jobs=-1)
type(model)
# Ignore ConvergenceWarning messages

import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=ConvergenceWarning)
# Fit and tune model

model.fit(X_train, y_train)
# Create empty dictionary called fitted_models

fitted_models = {}



# Loop through model pipelines, tuning each one and saving it to fitted_models

for name, pipeline in pipelines.items():

    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    

    # Fit model on X_train, y_train

    model.fit(X_train, y_train)

    

    # Store model in fitted_models[name] 

    fitted_models[name] = model

    

    # Print '{name} has been fitted'

    print(name, 'has been fitted.')
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
# Display fitted random forest object

fitted_models['lasso']
# Predict test set using fitted random forest

pred = fitted_models['lasso'].predict(X_test)
# Calculate and print R^2 and MAE

print( 'R^2:', r2_score(y_test, pred ))

print( 'MAE:', mean_absolute_error(y_test, pred))
# Predict test set using fitted random forest

pred = fitted_models['ridge'].predict(X_test)
# Calculate and print R^2 and MAE

print( 'R^2:', r2_score(y_test, pred ))

print( 'MAE:', mean_absolute_error(y_test, pred))
# Predict test set using fitted random forest

pred = fitted_models['enet'].predict(X_test)
# Calculate and print R^2 and MAE

print( 'R^2:', r2_score(y_test, pred ))

print( 'MAE:', mean_absolute_error(y_test, pred))
from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import pickle
# Load ABT from Module 3

df = pd.read_csv('analytical_base_table.csv')

print(df.shape)
# Create separate object for target variable

y = df.tx_price



# Create separate object for input features

X = df.drop('tx_price', axis=1)
# Split X and y into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   test_size=0.2,

                                                   random_state=1234)
# Create pipelines dictionary

pipelines = {

    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),

    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123))

}



# Add a pipeline for Elastic-Net

pipelines['enet'] = make_pipeline(StandardScaler(), ElasticNet(random_state=123))
# Lasso hyperparameters

lasso_hyperparameters = { 

    'lasso__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 5, 10] 

}



# Ridge hyperparameters

ridge_hyperparameters = { 

    'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]  

}



# Elastic Net hyperparameters

enet_hyperparameters = { 

    'elasticnet__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10],                        

    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  

}
# Add a pipeline for 'rf'

pipelines['rf'] = make_pipeline(StandardScaler(),

                                RandomForestRegressor(random_state=123))



# Add a pipeline for 'gb'

pipelines['gb'] = make_pipeline(StandardScaler(),

                                GradientBoostingRegressor(random_state=123))
print( pipelines['rf'] )
print( type( pipelines['rf'] ) )
# Check that we have all 5 model families, and that they are all pipelines

for key, value in pipelines.items():

    print( key, type(value) )
# Random forest hyperparameters

rf_hyperparameters = { 

    'randomforestregressor__n_estimators' : [100, 200],

    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],

}
# Boosted tree hyperparameters

gb_hyperparameters = { 

    'gradientboostingregressor__n_estimators': [100, 200],

    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],

    'gradientboostingregressor__max_depth': [1, 3, 5]

}
# Create hyperparameters dictionary

hyperparameters = {

    'rf' : rf_hyperparameters,

    'gb' : gb_hyperparameters,

    'lasso' : lasso_hyperparameters,

    'ridge' : ridge_hyperparameters,

    'enet' : enet_hyperparameters

}
for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:

    if key in hyperparameters:

        if type(hyperparameters[key]) is dict:

            print( key, 'was found in hyperparameters, and it is a grid.' )

        else:

            print( key, 'was found in hyperparameters, but it is not a grid.' )

    else:

        print( key, 'was not found in hyperparameters')
# Create empty dictionary called fitted_models

fitted_models = {}



# Loop through model pipelines, tuning each one and saving it to fitted_models

for name, pipeline in pipelines.items():

    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    

    # Fit model on X_train, y_train

    model.fit(X_train, y_train)

    

    # Store model in fitted_models[name] 

    fitted_models[name] = model

    

    # Print '{name} has been fitted'

    print(name, 'has been fitted.')
# Check that we have 5 cross-validation objects

for key, value in fitted_models.items():

    print( key, type(value) )
from sklearn.exceptions import NotFittedError



for name, model in fitted_models.items():

    try:

        pred = model.predict(X_test)

        print(name, 'has been fitted.')

    except NotFittedError as e:

        print(repr(e))
for name, model in fitted_models.items():

    pred = model.predict(X_test)

    print( name )

    print( '--------' )

    print( 'R^2:', r2_score(y_test, pred ))

    print( 'MAE:', mean_absolute_error(y_test, pred))

    print()
rf_pred = fitted_models['rf'].predict(X_test)

plt.scatter(rf_pred, y_test)

plt.xlabel('predicted')

plt.ylabel('actual')

plt.show()
type(fitted_models['rf'])
type(fitted_models['rf'].best_estimator_)
fitted_models['rf'].best_estimator_
import pickle



with open('final_model.pkl', 'wb') as f:

    pickle.dump(fitted_models['rf'].best_estimator_, f)