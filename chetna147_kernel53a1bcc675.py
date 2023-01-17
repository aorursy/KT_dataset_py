# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



# Read in track metadata with genre labels

tracks = pd.read_csv("../input/fma-rock-vs-hiphop.csv")



# Read in track metrics with the features

echonest_metrics = pd.read_json("../input/echonest-metrics.json", precise_float=True)



# Merge the relevant columns of tracks and echonest_metrics

echo_tracks = pd.merge(echonest_metrics, tracks[["track_id", "genre_top"]], on="track_id")



# Inspect the resultant dataframe

echo_tracks.info()
# Create a correlation matrix

corr_metrics = echo_tracks.corr()

corr_metrics.style.background_gradient()
# Define our features 

features = echo_tracks.drop(["genre_top","track_id"], axis=1)



# Define our labels

labels = echo_tracks["genre_top"]



# Import the StandardScaler

from sklearn.preprocessing import StandardScaler



# Scale the features and set the values to a new variable

scaler = StandardScaler()

scaled_train_features = scaler.fit_transform(features)
# This is just to make plots appear in the notebook

%matplotlib inline



# Import our plotting module, and PCA class

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



# Get our explained variance ratios from PCA using all features

pca = PCA()

pca.fit(scaled_train_features)

exp_variance = pca.explained_variance_ratio_



print(pca.explained_variance_ratio_)

print(pca.n_components_)



# plot the explained variance using a barplot

fig, ax = plt.subplots()

ax.bar(range(8), exp_variance)

ax.set_xlabel('Principal Component #')
# Import numpy

import numpy as np



# Calculate the cumulative explained variance

cum_exp_variance = np.cumsum(exp_variance)



# Plot the cumulative explained variance and draw a dashed line at 0.90.

fig, ax = plt.subplots()

ax.plot(range(8), cum_exp_variance)

ax.axhline(y=0.9, linestyle='--')

n_components = 6



# Perform PCA with the chosen number of components and project data onto components

pca = PCA(n_components, random_state=10)

pca.fit(scaled_train_features)

pca_projection = pca.transform(scaled_train_features)
# Import train_test_split function and Decision tree classifier

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



# Split our data

train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)



# Train our decision tree

tree = DecisionTreeClassifier(random_state=10)

tree.fit(train_features, train_labels)



# Predict the labels for the test data

pred_labels_tree = tree.predict(test_features)
# Import LogisticRegression

from sklearn.linear_model import LogisticRegression



# Train our logistic regression and predict labels for the test set

logreg = LogisticRegression(random_state=10)

logreg.fit(train_features, train_labels)

pred_labels_logit = logreg.predict(test_features)



# Create the classification report for both models

from sklearn.metrics import classification_report

class_rep_tree = classification_report(test_labels, pred_labels_tree)

class_rep_log = classification_report(test_labels, pred_labels_logit)



print("Decision Tree: \n", class_rep_tree)

print("Logistic Regression: \n", class_rep_log)
# Subset only the hip-hop tracks, and then only the rock tracks

hop_only = echo_tracks.loc[echo_tracks["genre_top"] == "Hip-Hop"]



# sample the rocks songs to be the same number as there are hip-hop songs

rock_only = echo_tracks.loc[echo_tracks["genre_top"] == "Rock"].sample(len(hop_only), random_state=10)



# concatenate the dataframes rock_only and hop_only

rock_hop_bal = pd.concat([rock_only, hop_only])



# The features, labels, and pca projection are created for the balanced dataframe

features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 

labels = rock_hop_bal['genre_top']

pca_projection = pca.fit_transform(scaler.fit_transform(features))



# Redefine the train and test set with the pca_projection from the balanced data

train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)
# Train our decision tree on the balanced data

tree = DecisionTreeClassifier(random_state=10)

tree.fit(train_features, train_labels)

pred_labels_tree = tree.predict(test_features)



# Train our logistic regression on the balanced data

logreg = LogisticRegression(random_state=10)

logreg.fit(train_features, train_labels)

pred_labels_logit = logreg.predict(test_features)



# Compare the models

print("Decision Tree: \n", classification_report(test_labels, pred_labels_tree))

print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))
from sklearn.model_selection import KFold, cross_val_score



# Set up our K-fold cross-validation

kf = KFold(n_splits=10, random_state=10)



tree = DecisionTreeClassifier(random_state=10)

logreg = LogisticRegression(random_state=10)



# Train our models using KFold cv

tree_score = cross_val_score(tree, pca_projection, labels, cv=kf)

logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)



# Print the mean of each array of scores

print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))