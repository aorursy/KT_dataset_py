# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
import numpy as np # linear algebra
import pandas as pd
secom_df= pd.read_csv("../input/secom_data.csv")
secom_df.head()
#percentage of rows with NaN values
((secom_df.shape[0] - secom_df.dropna().shape[0])/secom_df.shape[0]) *100
#every row has atleast 1 NaN value
#dropping the Time Column
secom_df = secom_df.drop(['Time'], axis = 1)
#imputing NaN with column mean
secom_df.fillna(secom_df.mean(), inplace=True)
secom_df.shape
secom_df.describe()
labels = np.array(secom_df['Pass/Fail'])
secom_df= secom_df.drop('Pass/Fail', axis = 1)
# Saving feature names for later use
feature_list = list(secom_df.columns)
# Convert to numpy array
features = np.array(secom_df)
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
 #Import the model we are using
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
#predictions = predictions.round()
# Calculate the absolute errors
errors = abs(predictions - test_labels)
print("Accuracy = "+str((errors.shape[0] - errors.sum())/ errors.shape[0]))
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# New random forest with only the two most important variables
rf_most_important = RandomForestClassifier(n_estimators= 1000, random_state=42)
# Extract the most important features (> 0 importance)
important_indices = [feature_list.index('16'), feature_list.index('25'), feature_list.index('38'), feature_list.index('40'), feature_list.index('59'),
                     feature_list.index('64'), feature_list.index('65'), feature_list.index('77'), feature_list.index('292'), feature_list.index('348')
                    , feature_list.index('426'), feature_list.index('441'), feature_list.index('539'), feature_list.index('561'), feature_list.index('562')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
predictions = rf_most_important.predict(test_important)
#predictions = predictions.round()
errors = abs(predictions - test_labels)
(errors.shape[0] - errors.sum())/ errors.shape[0]
