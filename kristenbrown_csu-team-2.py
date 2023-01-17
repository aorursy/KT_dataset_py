# First import the appropriate libraries

import os

import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt
# Take a look at the data we have available in the input directory

print(os.listdir('../input/'))
# Read in the label and training data

states = pd.read_csv('../input/state_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

breeds = pd.read_csv('../input/breed_labels.csv')

train = pd.read_csv('../input/train/train.csv')
# Check out the contents

print("State Labels")

states.head()
print("Color Labels")

colors.head()
print("Breed Labels")

breeds.head()
print("Training Data")

train.head()
# Look at summary statistics of training data

train.describe()
# Variables we can look at from column names

var_name = list(train.columns)



# Data type of each based on first entry

data_type = train.dtypes



num_unique = pd.Series({x: len(train.loc[:,x].unique()) for x in var_name})



# Get summary stats from the describe table

min_val = train.describe().loc["mean", :]

max_val = train.describe().loc["max", :]



# Get median via function to return NaN for non-numeric objects

def get_median(x, df, dtype):

    if dtype[x] != 'object':

        return df.loc[:,x].median()

    else:

        return np.NaN

    

med_val = pd.Series({x: get_median(x,train,data_type) for x in var_name})



# Get number of missing values

num_missing = pd.Series({x: train.loc[:,x].isnull().sum() for x in var_name})





# Create a new data frame with some descriptive measures by merging these series

desc_measures = pd.DataFrame({

    "data_type": data_type,

    "num_unique": num_unique,

    "min": min_val,

    "max": max_val,

    "med": med_val,

    "num_missing": num_missing

}, index=var_name)



# Check contents

desc_measures
# Now to plot the number of animals being adopted based on categories of speed

sns.set_style('darkgrid')

ax = sns.countplot(train.AdoptionSpeed, palette='Set2')

ax.set_xticklabels(labels=("Same Day", 

                           "1-7 days", 

                           "8-30 days", 

                           "31-90 days", 

                           "Not adopted after 100 days"), rotation=60)

ax.set_xlabel("Adoption Speed")

ax.set_ylabel("Animals Adopted")
# Split the plot into categories, is there one category that is being adopted faster/slower than others?

# First check out the categories with only a few unique options 

fig, ax = plt.subplots(2,2)

sns.countplot(x="AdoptionSpeed", hue="Type", data=train, ax=ax[0,0], palette='Set2')

sns.countplot(x="AdoptionSpeed", hue="Gender", data=train, ax=ax[0,1], palette='Set2')

sns.countplot(x="AdoptionSpeed", hue="Health", data=train, ax=ax[1,0], palette='Set2')

sns.countplot(x="AdoptionSpeed", hue="MaturitySize", data=train, ax=ax[1,1], palette='Set2')
# Week 2 libraries to import - normally this should be at the very beginning, 

# but I am separating by week so you can see what libraries get added each week

# We do use libraries from week 1 though

import warnings

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz



# Ignore warnings from sklearn (omit this if you're still experimenting with code)

def warn(*args, **kwargs):

    pass

warnings.warn = warn
# pull out the outcomes we want to predict. Since the data is skewed, look at values

# greater than zero

train_g0 = train.where(train.loc[:,"AdoptionSpeed"] > 0)

cat_train_g0 = train.where(train.loc[:,"Type"] == 2)

dog_train_g0 = train.where(train.loc[:,"Type"] == 1)



# note: we dropped Breed1 and RescuerID, but need to revisit these

cat_train_g0_subset = cat_train_g0.loc[:,("Age", 'Gender', 'Vaccinated', 

                                          'Dewormed', 'Health', 'Fee', 'PhotoAmt',

                                          'Sterilized', 'FurLength', 'Color1',

                                          'MaturitySize', 'AdoptionSpeed')].dropna()



cat_outcomes_g0 = cat_train_g0_subset.loc[:,'AdoptionSpeed']



dog_train_g0_subset = dog_train_g0.loc[:,("Age", 'Gender', 'Vaccinated', 

                                          'Dewormed', 'Health', 'Fee', 'PhotoAmt',

                                          'Sterilized', 'FurLength', 'Color1',

                                          'MaturitySize', 'AdoptionSpeed')].dropna()



dog_outcomes_g0 = dog_train_g0_subset.loc[:,'AdoptionSpeed']
# Fix the data types to contain categorical variables

category_cols = {'Age': np.int32,

    'Gender': 'category',

    'Vaccinated': 'category',

    'Dewormed': 'category',

    'Health': 'category',

    'Fee': np.int32,

    'PhotoAmt': np.int32,

    'Sterilized': 'category',

    'FurLength': 'category',

    'Color1': 'category',

    'MaturitySize': 'category',

    'AdoptionSpeed': np.int32

    }



for col,dtype in category_cols.items():

    cat_train_g0_subset.loc[:,col] = cat_train_g0_subset.loc[:,col].astype(dtype)

    dog_train_g0_subset.loc[:,col] = dog_train_g0_subset.loc[:,col].astype(dtype)

    

cat_train_g0_subset.dtypes
# Run a random forest model to determine what is driving empirical models

# Helpful in feature engineering



# Used the titanic code Steven shared with us as a baseline 

# Split the data into training and testing sets (20% of the data for testing)

Xtrain, Xtest, ytrain, ytest = train_test_split(cat_train_g0_subset.drop('AdoptionSpeed',axis=1), cat_outcomes_g0, test_size=0.20, random_state=154)



# Train and score the classifier

classifier = RandomForestClassifier(criterion='gini', n_jobs=4, random_state=154, n_estimators=100, oob_score=True)

classifier.fit(Xtrain, ytrain)

scores = classifier.score(Xtest, ytest)



print('The score of this Random Forest Classifier is {:.3f}'.format(scores))

print('The OOB score of this Random Forest Classifier is {:.3f}'.format(classifier.oob_score_))



# Feature importance values

importances = list(classifier.feature_importances_)

x_values = list(range(len(importances)))



# Feature importance plot

plt.figure(num=None, figsize=(8, 6), dpi=80)

plt.bar(x_values, importances, orientation='vertical')

plt.xticks(x_values, Xtrain.columns, rotation='vertical')

plt.ylabel('Importance')

plt.xlabel('Feature')

plt.title('Feature Importances: Gini Criterion')