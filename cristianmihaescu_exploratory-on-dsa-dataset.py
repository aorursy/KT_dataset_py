import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np



# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split



from sklearn.tree import export_graphviz

import pydot



# Import the model we are using

from sklearn.ensemble import RandomForestRegressor



# Import tools needed for visualization

from sklearn.tree import export_graphviz

import pydot



from IPython.display import Image



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



features = pd.read_csv("../input/dsa-test-dataset/fisier.csv")

features.head()

#features.describe()


# Labels are the values we want to predict

labels = np.array(features['ExamGradeD'])



# Remove the labels from the features

# axis 1 refers to the columns

features = features.drop('ExamGradeD', axis = 1)



# Remove 'Failure' as this is a computed feature which may act as a target variable

features = features.drop('Failure', axis = 1)



# Saving feature names for later use

feature_list = list(features.columns)

# Convert to numpy array

features = np.array(features)
# Split the data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)





print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages

baseline_preds = test_features[:, feature_list.index('MeanTestsGradeD')]

# Baseline errors, and display average baseline error

baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'points.')
# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 10, random_state = 42, max_depth = 3)



# Train the model on training data

rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data

predictions = rf.predict(test_features)



# Calculate the absolute errors

errors = abs(predictions - test_labels)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'points.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / test_labels)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Pull out one tree from the forest

tree = rf.estimators_[5]



# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph

(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file

graph.write_png('tree.png')

Image("tree.png")
# Get numerical feature importances

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];