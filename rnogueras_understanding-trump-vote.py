# Data manipulation

import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# SQL

import sqlite3
# Data 

path = "../input/2016-us-election/" 

database = path + 'database.sqlite'

conn = sqlite3.connect(database)
pd.read_sql("""



SELECT *

FROM sqlite_master

WHERE type='table';



""", conn)
pd.read_sql("""



SELECT * 

FROM primary_results

LIMIT 5;



""", conn)
pd.read_sql("""



SELECT * 

FROM county_facts

LIMIT 5;



""", conn)
dictionary = pd.read_sql("""



SELECT *

FROM county_facts_dictionary;



""", conn)
pd.read_sql("""



SELECT DISTINCT candidate

FROM primary_results

WHERE party = 'Republican';



""", conn)
df_raw = pd.read_sql("""



SELECT p.fips,

       p.county,

       p.candidate,

       p.votes,

       p.fraction_votes,

       c.*

FROM (

    SELECT *

    FROM (

        SELECT fips,

               county,

               party,

               candidate, 

               votes, 

               fraction_votes,

               row_number() 

                   OVER (PARTITION BY county 

                         ORDER BY fraction_votes desc) as rank

        FROM primary_results

        WHERE party = 'Republican') 

    WHERE rank = 1) p

JOIN county_facts c

    ON p.fips = c.fips;



""", conn)



df_raw[0:5]
print(df_raw['candidate'].value_counts())
# Create df with social variables only

df = df_raw.iloc[:, 8:] 



# Create binary target

df['target'] = df_raw['candidate']!='Donald Trump'

df['target'] = df['target'].astype(int)



# Distribution of target

counts = df['target'].value_counts()

print(counts)



proportion = round(counts[0] / (counts[0] + counts[1]), 2)

print('proportion: ', proportion, '/', 1-proportion)

X = df.drop('target', axis=1)

y = df.target
# Classifier

from sklearn import tree



# Pre-processing

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE



# Tune and cross validation

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
# Define tools

clf = tree.DecisionTreeClassifier(random_state=0)

smote = SMOTE(random_state=1) 

cv=StratifiedKFold(n_splits=10)



# Pipeline

pipeline = Pipeline([ 

                    ('smote', smote),

                    ('model', clf)

                   ])



# Tune algorithm

param_grid = {

              'model__max_depth': [3, 4, 5],

              'model__min_samples_leaf': [0.04, 0.06, 0.08]

             }



grid = GridSearchCV(pipeline, 

                    param_grid=param_grid, 

                    cv=cv, 

                    scoring="accuracy", 

                    n_jobs= -1)



# Fit

grid.fit(X, y)

best_estimator = grid.best_estimator_



# Print details

print('best params: ', grid.best_params_)

print('best score: ', round(grid.best_score_, 3))
# Add feature importances to dictionary

score = pd.Series(best_estimator.named_steps['model'].feature_importances_)

cols = pd.Series(X.columns)

feature_importance = pd.concat([cols, 

                               dictionary.iloc[:,1], 

                               score], 

                               axis=1)



# Drop low importance features

threshold = 0.01

mask = score>threshold

important_features = feature_importance[mask]

important_features
# Add a new column for feature names

names = pd.Series(['Under 18', 'Native American', 'Pacific Islander', 'Latino', 'Multi-units density', 'Land Area'])

names.index = important_features.index

important_features = pd.concat([names, important_features], axis=1)



# Rename columns

important_features.columns = ['name', 'code', 'description', 'score']

important_features
ax = sns.barplot(x='name', y='score', data=important_features)

plt.xticks(rotation=70)
# Replace codes with new names

cols.update(important_features['name'])



# Plot

fig = plt.figure(figsize=(20, 10))

tree.plot_tree(best_estimator.named_steps['model'], 

               filled=True, 

               feature_names=cols, 

               class_names=['Trump', 'Other'],

               rounded=True)