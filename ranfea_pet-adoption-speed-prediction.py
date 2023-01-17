# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score

# Bread and Butter
import os
import numpy as np
import pandas as pd
train_df = pd.read_csv("../input/train/train.csv")
test_df = pd.read_csv("../input/test/test.csv")
train_df.describe()
train_df.head()
train_df.dtypes
plt.title('Species', fontsize='xx-large')
train_df['Type'].value_counts().rename(
    {1:'Dogs',
     2:'Cats'}).plot(kind='barh')
plt.xlabel('Count')
plt.title('Adoption Speed', fontsize='xx-large')
train_df['AdoptionSpeed'].value_counts().rename(
    {0:'Same Day',
     1:'1-7 Days',
     2:'8-30 Days',
     3:'31-90 Days',
     4:'No adoption after 100 Days'}).plot(kind='barh')
plt.xlabel('Count')
# define target variable and eliminate the less useful metrics
target = train_df['AdoptionSpeed']
clean_df = train_df.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
clean_test = test_df.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])
# calculate the length of the description
descr_len_train = train_df['Description'].str.len()
descr_len_test = test_df['Description'].str.len()
# add the length of the decriptions to the dataframe
clean_df = pd.concat([clean_df,descr_len_train.fillna(0).astype(np.int32)],axis=1)
clean_test = pd.concat([clean_test,descr_len_test.fillna(0).astype(np.int32)],axis=1)
clean_df['Description'].describe()
classifier = RandomForestClassifier()

# Define the grid
rand_forest_grid = {
    'bootstrap': [True],
    'max_depth': [75, 80, 85, 90],
    'max_features': ['auto'],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [5, 10, 15],
    'n_estimators': [150, 175, 200, 225]
}


# Search parameter space
rand_forest_gridsearch = GridSearchCV(estimator = classifier, 
                           param_grid = rand_forest_grid, 
                           cv = 2, 
                           verbose = 1,
                           n_jobs = -1)
# Fit the models
rand_forest_gridsearch.fit(clean_df, target)
# What are the best parameters for each model?
rand_forest_gridsearch.best_params_
# Measure of performance 
print('Random Forest score: ', cohen_kappa_score(rand_forest_gridsearch.predict(clean_df), 
                                target, weights='quadratic'))
test_predictions = rand_forest_gridsearch.predict(clean_test)
prediction_df = pd.DataFrame({'PetID' : test_df['PetID'],
                              'AdoptionSpeed' : test_predictions})
prediction_df.head()
prediction_df.to_csv('submission.csv', index=False)
