# Import libraries necessary for this project

import numpy as np

import pandas as pd



from IPython.display import display # Allows the use of display() for DataFrames



# Show matplotlib plots inline (nicely formatted in the notebook)

%matplotlib inline



input_file = '../input/Dataset_spine.csv'

input_data = pd.read_csv(input_file)



renamed_columns = {

    'Col1': 'pelvic_incidence',

    'Col2': 'pelvic_tilt',

    'Col3': 'lumbar_lordosis_angle',

    'Col4': 'sacral_slope',

    'Col5': 'pelvic_radius',

    'Col6': 'degree_spondylolisthesis',

    'Col7': 'pelvic_slope',

    'Col8': 'direct_tilt',

    'Col9': 'thoracic_slope',

    'Col10': 'cervical_tilt',

    'Col11': 'sacrum_angle',

    'Col12': 'scoliosis_slope',

    'Class_att' : 'classification'

}

input_data.rename(columns=renamed_columns, inplace=True)

input_data.drop(input_data.columns[13], axis=1, inplace=True)

display(input_data.head())

display(input_data.tail())

display(input_data.describe())
# take a sample size of the data, 1.0 if the data is not too large.

import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames



sample_size = 1.0

data = input_data.sample(frac=sample_size)



# Randomize the data

data = data.sample(frac=1.0).reset_index(drop=True)



display(data.head())



# Extract the classification column from the data

classification = data[['classification']]

display(classification.describe())

display(classification.head())

classification.describe().to_csv("classification_stats.csv", float_format='%.6f', index=True)
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn import preprocessing



# Scale the data using the natural logarithm

#preprocessed_data = reduced_feature_data.apply(np.log)



# Scale the data using preprocessing function in sklearn

feature_cols = list(data.columns[:-1])

preprocessed_data = data.copy()



suggested_features_to_drop = []

for feature in feature_cols:

    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature

    # Extract the values of the dataframe to be used for the regression

    new_data = preprocessed_data.drop([feature], axis = 1, inplace = False)

    remaining_cols = list(new_data.columns[:-1])

    new_data_values = new_data[remaining_cols].values

    target_label = data[feature].values



    # Split the data into training and testing sets using the given feature as the target

    X_train, X_test, y_train, y_test = train_test_split(new_data_values, target_label, test_size=0.20, random_state=42)



    # Create a decision tree regressor and fit it to the training set

    regressor = tree.DecisionTreeRegressor(random_state=42)

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)



    # Report the score of the prediction using the testing set

    score = regressor.score(X_test, y_test)

    print (feature, score)

    if score < 0.0:

        suggested_features_to_drop.append(feature)

        



print ("\nSuggested features to drop:")

for feature in suggested_features_to_drop: print (feature)
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames



# Initial set of features to drop by using the features with negative scores above

dropped_features = ['pelvic_radius', 'pelvic_slope', 'direct_tilt', 'thoracic_slope', 'cervical_tilt', 'sacrum_angle', 'scoliosis_slope']

 

reduced_feature_data = data.copy()

for feature in dropped_features:

    print ("Manually dropping feature:", feature)

    reduced_feature_data.drop([feature], axis = 1, inplace = True)

    

display(reduced_feature_data.head())

reduced_feature_data.to_csv("reduced_feature_data.csv", float_format='%.6f', index=False)
import pandas as pd



reduced_feature_data = pd.read_csv("reduced_feature_data.csv")

feature_cols = list(reduced_feature_data.columns[:-1])



# Visulalize the original data

pd.scatter_matrix(reduced_feature_data[feature_cols], alpha = 0.3, figsize = (8,10), diagonal = 'kde');
import pandas as pd

from sklearn import preprocessing

import numpy as np



feature_cols = list(reduced_feature_data.columns[:-1])

target_col = reduced_feature_data.columns[-1] 

preprocessed_data = reduced_feature_data.copy()



# Scale the data using the natural logarithm



if True:

    preprocessed_data = preprocessed_data[feature_cols].apply(np.log)

    preprocessed_data = pd.concat([preprocessed_data, reduced_feature_data[target_col]], axis = 1)



# Scale the data using preprocessing function in sklearn



if False:

    scaler=preprocessing.StandardScaler()

    scaler.fit(reduced_feature_data[feature_cols])

    preprocessed_data[feature_cols] = scaler.transform(reduced_feature_data[feature_cols])



    display(preprocessed_data.describe())



# Drop any rows that contain NA values due to the scaling

preprocessed_data.dropna(subset=feature_cols, inplace=True)



# Display the percentage of the dataset kept.

percent_kept = (float(len(preprocessed_data)) / float(len(reduced_feature_data))) * 100.00

print ("Percentage kept:  {:2.2f}%".format(percent_kept))
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames



# Output data set so that the analysis can be restarted w/out having to replay the entire notebook

preprocessed_data.to_csv("scaled_data.csv", float_format='%.6f', index=False)



# Visualize the scaled data

display(preprocessed_data.head())

pd.scatter_matrix(preprocessed_data, alpha = 0.3, figsize = (8,10), diagonal = 'kde');
import pandas as pd

import numpy as np

from collections import Counter

from IPython.display import display # Allows the use of display() for DataFrames



# Get the scaled data data set so that the analysis can be restarted w/out having to replay the entire notebook

scaled_data = pd.read_csv("scaled_data.csv")



# For each feature find the data points with extreme high or low values

outlierCounts = np.array([])



feature_cols = list(scaled_data.columns[:-1])



for feature in feature_cols:

    

    # Calculate Q1 (25th percentile of the data) for the given feature

    Q1 = np.percentile(scaled_data[feature].values, 25)

    

    # Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = np.percentile(scaled_data[feature].values, 75)

    

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = 1.5 * (Q3 - Q1)

    

    # Display the outliers

    print ("Data points considered outliers for the feature '{}':".format(feature))

    is_outlier = ~((scaled_data[feature] >= Q1 - step) & (scaled_data[feature] <= Q3 + step))

    display(scaled_data[~((scaled_data[feature] >= Q1 - step) & (scaled_data[feature] <= Q3 + step))])

    outlierCounts = np.append(outlierCounts, scaled_data[~((scaled_data[feature] >= Q1 - step) & (scaled_data[feature] <= Q3 + step))].index)



# OPTIONAL: Select the indices for data points you wish to remove

outlierCounts = outlierCounts.astype(int)

outlierCounted = Counter(outlierCounts)



print ("Number of data points that have one or more outlier features: ", len(outlierCounted))



outliers = [key for key,val in outlierCounted.items()]

print ("Data points with outliers: ", outliers)

# Remove the outliers, if any were specified

good_scaled_data = scaled_data.drop(scaled_data.index[outliers]).reset_index(drop = True)



display(good_scaled_data.describe())



# Output good scaled data set so that the analysis can be restarted w/out having to replay the entire notebook

good_scaled_data.to_csv("good_scaled_data.csv", float_format='%.6f', index=False)
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.cluster import AffinityPropagation

from sklearn.mixture import GaussianMixture

from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_score



# Get the scaled data data set so that the analysis can be restarted w/out having to replay the entire notebook

good_scaled_data = pd.read_csv("good_scaled_data.csv")



feature_cols = list(good_scaled_data.columns[:-1])

feature_data = good_scaled_data[feature_cols]



kmeans = KMeans(n_clusters=3)

afprop =  AffinityPropagation(preference=-50)

gmm_spherical = GaussianMixture(n_components=3, covariance_type='spherical', random_state = 42)



clusterers = [kmeans, afprop, gmm_spherical]



best = -1.0

for clusterer in clusterers:

    clusterer.fit(feature_data)



    # Predict the cluster for each data point

    preds = clusterer.predict(feature_data)



    # Calculate the mean silhouette coefficient for the number of clusters chosen

    score = silhouette_score(feature_data, preds)

    

    print ("Clusterer {}      score {:0.4f}.".format(type(clusterer).__name__, score))



    if best < score:

        best = score

        best_clusterer = clusterer

        best_preds = preds



print ("\nThe best score was {:0.4f} using {} clusterer.".format(best, type(best_clusterer).__name__))



predictions = pd.DataFrame(best_preds, columns = ['cluster'])

predictions.to_csv("predictions.csv", float_format='%.6f', index=False)



print ("\n", predictions.groupby(['cluster']).size())



print ("\nClustering complete.")
import matplotlib.pyplot as plt

import matplotlib.cm as cm

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from IPython.display import display # Allows the use of display() for DataFrames



%matplotlib inline



good_scaled_data = pd.read_csv('good_scaled_data.csv')

predictions = pd.read_csv('predictions.csv')

feature_data = good_scaled_data.drop(['classification'], axis = 1)



pca_components = 2

pca = PCA(n_components=pca_components).fit(feature_data)

reduced_data = pca.transform(feature_data)



# Create a DataFrame for the reduced data

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])



# Output data set

reduced_data.to_csv("reduced_data.csv", float_format='%.6f', index=False)

plot_data = pd.concat([predictions, reduced_data], axis = 1)



# Generate the cluster plot

fig, ax = plt.subplots(figsize = (8,8))



# Color map

cmap = cm.get_cmap('gist_rainbow')



clusters = plot_data['cluster'].unique()



# Color the points based on assigned cluster

for i, cluster in plot_data.groupby('cluster'):   

    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \

                 color = cmap((i)*1.0/(len(clusters)-1)), label = 'Cluster %i'%(i), s=200);



# Set plot title

ax.set_title("Cluster Learning on PCA-Reduced Data");
import pandas as pd



good_scaled_data = pd.read_csv('good_scaled_data.csv')

reduced_data = pd.read_csv('reduced_data.csv')

predicitons = pd.read_csv('predictions.csv')



full_data = pd.concat([good_scaled_data, predictions, reduced_data], axis = 1)



full_data.to_csv('full_processed_data.csv', index=False)

display(full_data.head())
import pandas as pd



full_data = pd.read_csv('full_processed_data.csv')



sample_data = full_data.sample(frac=0.0)

frac_samples_per_cluster = 0.20 / float(len(full_data['cluster'].unique()))

samples_per_cluster = 20



clusters = full_data['cluster'].unique()



for cluster_number in clusters:

    cluster_data = full_data[full_data['cluster'] == cluster_number].sample(n=samples_per_cluster)

    sample_data = sample_data.append(cluster_data)



print ("Chosen samples segments dataset:")

#sample_data = full_data.sample(n=20)

display(sample_data.sort_values(by=['cluster']))



sample_data.to_csv('sample_data.csv', index=False)
import pandas as pd

from sklearn.metrics.cluster import homogeneity_score

from sklearn.metrics.cluster import v_measure_score

from sklearn.metrics import adjusted_rand_score



sample_data = pd.read_csv('sample_data.csv')



print("Homogeneity Score is:       %.2f" % homogeneity_score(sample_data['cluster'].values, sample_data['classification'].values))

print("V Measure Score is:         %.2f" % v_measure_score(sample_data['cluster'].values, sample_data['classification'].values))

print("Adjusted Rand Score is:     %.2f" % adjusted_rand_score(sample_data['cluster'].values, sample_data['classification'].values))
import pandas as pd

from scipy.stats import mode



abnormal_clusters = []

sample_data = pd.read_csv('sample_data.csv')

clusters = sample_data['cluster'].unique()



for cluster_number in clusters:

    cluster_data = sample_data[sample_data['cluster'] == cluster_number]

    

    cluster_size = len(cluster_data)

    print ("Cluster", cluster_number, "has", cluster_size, "elements")

    

    cluster_classification = mode(cluster_data[['classification']])[0][0][0]

    

    print ("Cluster", cluster_number, "will be classified as", cluster_classification)

    

    if cluster_classification == 'Abnormal':

        abnormal_clusters.append(cluster_number)





abnormal_preds = full_data[(full_data['cluster'].isin(abnormal_clusters))]

normal_preds = full_data[(~full_data['cluster'].isin(abnormal_clusters))]

print ("Abnormal Preds:            {}".format(len(abnormal_preds.index)))

print ("Normal Preds:              {}".format(len(normal_preds.index)))
import matplotlib.pyplot as plt

import matplotlib.cm as cm

import pandas as pd

import numpy as np

from IPython.display import display # Allows the use of display() for DataFrames



%matplotlib inline



# Visualize the cluster data

full_data = pd.read_csv('full_processed_data.csv')

sample_data = pd.read_csv('sample_data.csv')



pca_samples = sample_data[['Dimension 1', 'Dimension 2', 'classification']]



plot_data = full_data[['cluster', 'Dimension 1', 'Dimension 2']]

# Generate the cluster plot

fig, ax = plt.subplots(figsize = (8,8))



# Color map

cmap = cm.get_cmap('gist_rainbow')



centers = plot_data['cluster'].unique()



# Color the points based on assigned cluster

for i, cluster in plot_data.groupby('cluster'):   

    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \

                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=200);



# Plot transformed abnormal sample points 

abnormal_samples = pca_samples[(pca_samples['classification'] == 'Abnormal')].values

ax.scatter(x = abnormal_samples[:,0], y = abnormal_samples[:,1], \

           s = 30, linewidth = 1, color = 'black', marker = 'v');



# Plot transformed normal sample points 

abnormal_samples = pca_samples[(pca_samples['classification'] == 'Normal')].values

ax.scatter(x = abnormal_samples[:,0], y = abnormal_samples[:,1], \

           s = 30, linewidth = 1, color = 'blue', marker = '^');

# Set plot title

ax.set_title("Cluster Learning on PCA-Reduced Data - Sample Classifcations Marked [N = ^, A = v]");
import pandas as pd

import numpy as np

from IPython.display import display # Allows the use of display() for DataFrames



full_data = pd.read_csv('full_processed_data.csv')

training_data = full_data.drop(['classification', 'Dimension 1', 'Dimension 2'], axis=1)



training_data.loc[training_data['cluster'].isin(abnormal_clusters), 'project_classification'] = 'Abnormal'

training_data.loc[~training_data['cluster'].isin(abnormal_clusters), 'project_classification'] = 'Normal'



training_data.drop(['cluster'], axis=1, inplace=True)



# Randomize the data

training_data = training_data.sample(frac=1.0)



training_data.to_csv('training_data.csv', float_format='%.6f', index=False)



display(training_data.head())

print ("Training data created.")
import pandas as pd



training_data = pd.read_csv('training_data.csv')



# Extract feature columns (first set)

feature_cols = list(training_data.columns[:-1])



# Extract target column 'false_positive' (last column)

target_col = training_data.columns[-1] 



# Show the list of columns

print ("Feature columns:\n{}".format(feature_cols))

print ("\nTarget column: {}".format(target_col))



# Separate the data into feature data and target data (X_all and y_all, respectively)

X_all = training_data[feature_cols]

y_all = training_data[target_col]
# Import the three supervised learning models from sklearn

import pandas as pd

from sklearn import tree

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.model_selection import cross_val_score



# Initialize the three models

clf_A = tree.DecisionTreeClassifier(random_state=42)

clf_A_parameters = {'min_samples_split':(2, 200, 2000), 'min_samples_leaf': (2, 4, 6, 8 , 10), 'splitter': ('best', 'random')}



clf_B = SVC(random_state=42)

clf_B_parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'gamma': (1, 100, 1000, 10000, 100000), 'C':(1.0, 2.0, 3.0)}



clf_C = GaussianNB()

clf_C_parameters = {}



classifiers = [clf_A, clf_B, clf_C]

parameters = [clf_A_parameters, clf_B_parameters, clf_C_parameters]



# Loop through each classifier, train and test. Keep the one with the best test score

best_score = 0.0

for index in range(len(classifiers)):

    clf = classifiers[index]

    print ("\n{}:".format(clf.__class__.__name__))

    

    # Using Cross Validation

    

    scores = cross_val_score(clf, X_all, y_all, cv=10)

    print ("F1 mean score for CV training set: {:.4f}.".format(scores.mean()))



    if best_score < scores.mean():

        best_score = scores.mean()

        best_clf = clf

        best_parms = parameters[index]



print ("\nThe best testing score was : {:.4f}.".format(best_score))

print ("The best classifier was: {}".format(best_clf.__class__.__name__))

print ("The optional parameteres for this classifier was: {}".format(best_parms))
import pandas as pd

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV



# Perform grid search on the classifier using the f1_scorer as the scoring method

grid_obj = GridSearchCV(best_clf, param_grid=best_parms, cv=10)



# Fit the grid search object to the training data and find the optimal parameters

grid_obj.fit(X_all, y_all)



print ("Best parameters set found on development set:")

print (grid_obj.best_params_)

print ()

# Get the estimator

clf = grid_obj.best_estimator_

print (clf)



print ("\nBest score: {:.4f}.".format(grid_obj.best_score_))
from sklearn.externals import joblib

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import train_test_split

import math



# Shuffle and split the dataset into the number of training and testing points above

X_train, X_test, y_train, y_test = train_test_split(

    X_all, y_all, train_size=0.75, test_size=0.25, random_state=42)



# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape[0]))

print ("Testing set has {} samples.".format(X_test.shape[0]))



joblib.dump(clf, 'classification_model.pkl')

loaded_clf = joblib.load('classification_model.pkl')



testing_pred = loaded_clf.predict(X_test)

# Get Score based on training data

f1_score_testing_data = f1_score(y_test.values, testing_pred, pos_label="Abnormal")

print ("F1 score for test set: {:.4f}.".format(f1_score_testing_data))
# Finally score the original classification to the one derived by the model

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



full_data = pd.read_csv('full_processed_data.csv')

validation_data = full_data.drop(['cluster', 'Dimension 1', 'Dimension 2'], axis=1)



# Extract feature columns (first set)

feature_cols = list(validation_data.columns[:-1])



# Extract target column 'false_positive' (last column)

target_col = validation_data.columns[-1] 



# Show the list of columns

print ("Feature columns:\n{}".format(feature_cols))

print ("\nTarget column: {}".format(target_col))



# Separate the data into feature data and target data (X_all and y_all, respectively)

X_all = validation_data[feature_cols]

y_all = validation_data[target_col]



loaded_clf = joblib.load('classification_model.pkl')



validation_pred = loaded_clf.predict(X_all)

# Get Score based on training data

validation_score = accuracy_score(y_all.values, validation_pred)

print ("\nValidation accuracy score: {:.4f}".format(validation_score))



target_names = ['Abnormal', 'Normal']



report = classification_report(y_all.values, validation_pred, target_names)

cf_matrix = confusion_matrix(y_all.values, validation_pred, target_names)



confusion = pd.crosstab(y_all, validation_pred, rownames=['Actual'], colnames=['Classified As'])



display(confusion)





cf_matrix = confusion.values

sensitivity = cf_matrix[0][0] / float(np.sum(cf_matrix[0][0:]))

specificity =  cf_matrix[1][1] / float(np.sum(cf_matrix[1][0:]))



print ("\nSensitivity:             {:.4f}".format(sensitivity))

print ("\nSpecificity:             {:.4f}".format(specificity))

print ("\nClassification Report\n", report)