import pandas as pd

features = pd.read_csv("../input/temperature-data-seattle/temps.csv")
features.head(5)
print(features.shape)
# Descriptive statistics for each column

features.describe()
import matplotlib.pyplot as plt
with plt.style.context('ggplot'):

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)

    axes[0][0].plot(features['actual'])

    axes[0][0].set_title('Max Temp')

    axes[0][1].plot(features['temp_1'])

    axes[0][1].set_title('Previous Max Temp')

    axes[1][0].plot(features['temp_2'])

    axes[1][0].set_title('Prior Two Days Max Temp')

    axes[1][1].plot(features['friend'])

    axes[1][1].set_title('Friend Estimate')

    plt.plot()
# One-hot encode the data using pandas get_dummies

features = pd.get_dummies(features)
features.head()
import numpy as np

# Access the targets

labels = np.array(features['actual'])

# remove targets from the features

# axis 1 refers to the columns

features=features.drop('actual', axis=1)
# saving feature names for later use

feature_list = list(features.columns)
# Convert to numpy array

features = np.array(features)
# Suing Scikit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split



# Split the data into training adn testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print(f'Training Features Shape: {train_features.shape}')

print(f'Testing Features Shape: {test_features.shape}')

print(f'Training Labels Shape: {train_labels.shape}')

print(f'Testing Labels Shape: {test_labels.shape}')
# The baseline predictions ar eth historical averages

baseline_preds = test_features[:, feature_list.index('average')]

# Baseline errors, and display average baseline error

baseline_errors = abs(test_labels - baseline_preds)



print(f'Average baseline error (MAE): {round(np.mean(baseline_errors), 2)}')
# Improt the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators=1000, random_state=42)



# Train the model on training data

rf.fit(train_features, train_labels)
# Use the forests predict method on the test data

predictions = rf.predict(test_features)



# Calcuate the absoulte errors

errors = abs(test_labels - predictions)



# Print out the mean absolute error (MAE)

print(f'Mean Absolute Error: {round(np.mean(errors), 2)}')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors/test_labels)

# Calcualte and display accuracy

accuracy = 100 - np.mean(mape)

print(f'Accuracy: {round(accuracy, 2)}%.')
# Import tools needed for visualization

from sklearn.tree import export_graphviz

import pydot



# Pull out one tree from the forest

tree = rf.estimators_[5]



# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names=feature_list, rounded=True, precision = 1)
# Use dot file to create a graph

(graph, ) = pydot.graph_from_dot_file('tree.dot')



# Write graph to a png file

graph.write_png('tree.png')
from IPython.display import Image

Image(filename='tree.png')
# Limit the depth of the tree to 3 levels

rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)

rf_small.fit(train_features, train_labels)

# Extract the small tree

tree_small = rf_small.estimators_[5]



# Svae thre tree as a png image

export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list, rounded=True, precision=1)



(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png("small_tree.png")
Image(filename='small_tree.png')
# Get the numerical fature importances

importances = list(rf.feature_importances_)
list(zip(feature_list, importances))
# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances
# Sort hte feature importances by most important first

feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
feature_importances
# Print out

_ = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# New random forest with only the two most important variables

rf_most_important = RandomForestRegressor(n_estimators = 1000, random_state=42)
# Extract the two most important features

important_indices = [feature_list.index('temp_1'), feature_list.index('average')]

train_important = train_features[:, important_indices]

test_important = test_features[:, important_indices]



# Train the random forest

rf_most_important.fit(train_important, train_labels)



# Make predictions and determine the error

predictions = rf_most_important.predict(test_important)



errors = abs(predictions - test_labels)



# Display the performance metrics

print('Mean Absolute Error: ', round(np.mean(errors), 2), ' degrees.')



mape = np.mean(100*(errors/test_labels))

accuracy = 100 - mape



print('Accuracy: ', round(accuracy, 2), '%.')
# Set the style

plt.style.use('fivethirtyeight')

# List of x locations for plotting

x_values = list(range(len(importances)))



# Make a bar chart

plt.bar(x_values, importances, orientation='vertical')

# Tick labels for x axis

plt.xticks(x_values, feature_list, rotation='vertical')



# Axis labels and title

plt.ylabel('Importance')

plt.xlabel('Variable')

plt.title('Variable Importances')
# Use datetime for creating data objects for plotting

import datetime



# Dates of training values

months = features[:, feature_list.index('month')]

days =features[:, feature_list.index('day')]

years = features[:, feature_list.index('year')]



# List and then convert to datetime object

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# Dates of predictions

months = test_features[:, feature_list.index('month')]

days = test_features[:, feature_list.index('day')]

years = test_features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]



# Convert to datetime objects

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]



# Dataframe with predictions and dates

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})
# Plot the actual values

plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

# Plot the predicted values

plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')

plt.xticks(rotation='60')

plt.legend()

# Graph labels

plt.xlabel('Date')

plt.ylabel('Maximum temperature (F)')

plt.title('Actual and Predicted Values')
# Make the data accessible for plotting

true_data['temp_1'] = features[:, feature_list.index('temp_1')]

true_data['average'] = features[:, feature_list.index('average')]

true_data['friend'] = features[:, feature_list.index('friend')]



# Plot all the data as lines

plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual', alpha = 1.0)

plt.plot(true_data['date'], true_data['temp_1'], 'y-', label='temp_1', alpha=1.0)

plt.plot(true_data['date'], true_data['average'], 'y-', label='average', alpha = 0.8)

plt.plot(true_data['date'], true_data['friend'], 'r-', label='friend', alpha =0.3)

plt.legend()

plt.xticks(rotation='60')

plt.xlabel('Date')

plt.ylabel('Maximum Tewmperature (F)')

plt.title('Actual Max Temp and Variables')