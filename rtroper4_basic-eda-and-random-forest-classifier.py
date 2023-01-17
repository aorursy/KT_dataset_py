# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# See where I am relative to the data

os.getcwd()
# Load training data and print info

train_all = pd.read_csv('../input/learn-together/train.csv')

train_all.info()  # Info indicates that there is no missing data



# Load test data

test_all = pd.read_csv('../input/learn-together/test.csv')
# Statistical description of the data

train_all.describe()
# Note that some columns have no or very few non-zero values

(train_all > 0).sum(axis=0)
# Drop any columns with 10 or fewer non-zero values

# The assumption, here, is that such columns don't provide enough descriminatory information to be useful in a model

columns_to_use = list(train_all.columns[(train_all > 0).sum(axis=0) > 10])

columns_to_use.remove('Id')  # We do not want to use 'Id' as a feature



# By inspection of scatter plots, these were selected to be dropped

#to_remove = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 

#             'Wilderness_Area3', 'Vertical_Distance_To_Hydrology', 'Soil_Type4',

#             'Soil_Type10', 'Soil_Type11', 'Soil_Type16', 'Soil_Type17', 'Soil_Type21', 'Soil_Type27']



to_remove = ['Aspect', 'Soil_Type4']



#to_keep = ['Elevation', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 

#           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type3', 'Soil_Type6',

#           'Soil_Type11', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16', 'Soil_Type18', 'Soil_Type20',

#           'Soil_Type21', 'Soil_Type24', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',

#           'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']



to_keep = ['Elevation', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 

           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type3', 'Soil_Type6', 

           'Soil_Type18', 'Soil_Type20', 'Soil_Type21', 'Soil_Type24',

           'Soil_Type31', 'Soil_Type33', 'Soil_Type34', 'Soil_Type40', 'Cover_Type']



columns_to_use = [c for c in columns_to_use if not(c in to_remove)]

#columns_to_use = [c for c in columns_to_use if (c in to_keep)]



# Select subset of columns

train_all = train_all[columns_to_use]

train_all.describe()
train_all.columns
# Store target column in Y_train_all and drop from train_all

y_train_all = train_all.Cover_Type

X_train_all = train_all.drop('Cover_Type', axis=1)
len(X_train_all.columns)
# Get number of features

N = X_train_all.shape[1]

nrows = int(N/math.sqrt(N))

ncols = math.ceil(N/nrows)



# Set dimensions of subplots (plotting takes about 15-20 seconds)

fig, axs = plt.subplots(nrows, ncols, figsize=(20,24))

for i, column in enumerate(X_train_all.columns):

    r = int(i/ncols)

    c = i % ncols

    axs[r, c].scatter(X_train_all.iloc[:, i], y_train_all, alpha=0.2, s=9)

    axs[r, c].set_title(column[:min(20, len(column))])

    

plt.show()
# Split data into train and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.20, random_state=1)
# Function to fit RandomForestClassifier and calculate accuracy

def rfc_fit_and_predict(X_train, y_train, X_val, y_val, n_estim=100, max_depth=2, min_samp_leaf=5, rand=0):

    # Create and fit model

    model = RandomForestClassifier(n_estimators=n_estim, max_depth=max_depth, min_samples_leaf=min_samp_leaf, random_state=rand)

    model.fit(X_train, y_train)

    

    # Predict on validation set and get accuracy

    y_pred = model.predict(X_val)

    if(type(y_val) == type(None)):

        accuracy = None

    else:

        accuracy = accuracy_score(y_val, y_pred)

    

    # Return the model and accuracy score

    return (model, y_pred, accuracy)
# Specify range of estimator numbers and max depths to use

estimators = [2, 4, 8, 16, 32, 64]

depths = [2, 4, 8, 16, 32, 64]

min_samp_leaf = 5



# Create a dataframe to store accuracy scores

row_labels = [str(e) for e in estimators]

col_labels = [str(d) for d in depths]

accuracy_scores = pd.DataFrame(index=row_labels, columns=col_labels, dtype='float')



for i, num_estim in enumerate(estimators):

    if(i > 0):

        print('')

    for depth in depths:

        # Fit on training set and score on validation set

        rfc_model, _, accuracy = rfc_fit_and_predict(X_train, y_train, X_val, y_val, n_estim=num_estim, max_depth=depth, min_samp_leaf=min_samp_leaf)

        

        # Print accuracy to console

        print('estim: {0}\tdepth: {1}\taccuracy: {2}'.format(num_estim, depth, accuracy))

        

        # Store accuracy in dataframe

        accuracy_scores.loc[str(num_estim), str(depth)] = accuracy
# plot heatmap of accuracy scores

ax = sns.heatmap(accuracy_scores.T, annot=True, cmap="YlGnBu")

ax.set_xlabel('number of estimators')

ax.set_ylabel('max depth')

plt.show()
# Number of estimators and maximum depth to use

number_estimators = 8

maximum_depth = 16

min_samp_leaf = 5
# Generate scatter plots again, but only for validation set

# Color according to the correct predictions to visualize incorrect categorizations



# Get number of features

N = X_train_all.shape[1]

nrows = 1

ncols = 7



# Get predictions on validation set

rfc_model, y_pred_val, accuracy = rfc_fit_and_predict(X_train, y_train, X_val, y_val, n_estim=number_estimators, max_depth=maximum_depth, min_samp_leaf=min_samp_leaf)
# Generate scatter plots again, but only for validation set

# Color according to the correct predictions to visualize incorrect categorizations



# Get number of features

N = X_train_all.shape[1]

nrows = 7

ncols = 1



# Get predictions on validation set

rfc_model, y_pred_val, accuracy = rfc_fit_and_predict(X_train, y_train, X_val, y_val, n_estim=number_estimators, max_depth=maximum_depth, min_samp_leaf=min_samp_leaf)



# Set dimensions of subplots

fig, axs = plt.subplots(nrows, ncols, figsize=(6,24))

for i in range(7):

    r = int(i/ncols)

    c = i % ncols

    axs[i].hist(y_val[y_pred_val == i+1], bins=7)

    axs[i].set_xlim((1,7))

    axs[i].set_ylim((0,400))

    axs[i].set_title('Predicted: {0}'.format(i+1))

    

plt.show()
# Prepare test data

X_test = test_all[columns_to_use[:-1]]



# Generate results for submission to competition

rfc_model, y_pred, _ = rfc_fit_and_predict(X_train_all, y_train_all, X_test, None, n_estim=number_estimators, max_depth=maximum_depth, min_samp_leaf=min_samp_leaf)



# Create dataframe with proper columns

results = pd.DataFrame({'Id':test_all.Id, 'Cover_Type':y_pred})
# Write results to file

results.to_csv('submission.csv', index=False)