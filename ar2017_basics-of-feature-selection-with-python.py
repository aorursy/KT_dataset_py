import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")



import warnings

warnings.filterwarnings('ignore')
# Load Iris dataset from Scikit-learn

from sklearn.datasets import load_iris



# Create input and output features

feature_names = load_iris().feature_names

X_data = pd.DataFrame(load_iris().data, columns=feature_names)

y_data = load_iris().target



# Show the first five rows of the dataset

X_data.head()
# Import f_classif from Scikit-learn

from sklearn.feature_selection import f_classif
# Create f_classif object to calculate F-value

f_value = f_classif(X_data, y_data)



# Print the name and F-value of each feature

for feature in zip(feature_names, f_value[0]):

    print(feature)
# Create a bar chart for visualizing the F-values

plt.figure(figsize=(4,4))

plt.bar(x=feature_names, height=f_value[0], color='tomato')

plt.xticks(rotation='vertical')

plt.ylabel('F-value')

plt.title('F-value Comparison')

plt.show()
# Import VarianceThreshold from Scikit-learn

from sklearn.feature_selection import VarianceThreshold
# Create VarianceThreshold object to perform variance thresholding

selector = VarianceThreshold()



# Perform variance thresholding

selector.fit_transform(X_data)



# Print the name and variance of each feature

for feature in zip(feature_names, selector.variances_):

    print(feature)
# Create a bar chart for visualizing the variances

plt.figure(figsize=(4,4))

plt.bar(x=feature_names, height=selector.variances_, color='tomato')

plt.xticks(rotation='vertical')

plt.ylabel('Variance')

plt.title('Variance Comparison')



plt.show()
# Create VarianceThreshold object to perform variance thresholding

selector = VarianceThreshold(threshold=0.2)



# Transform the dataset according to variance thresholding

X_data_new = selector.fit_transform(X_data)



# Print the results

print('Number of features before variance thresholding: {}'.format(X_data.shape[1]))

print('Number of features after variance thresholding: {}'.format(X_data_new.shape[1]))
# Import mutual_info_classif from Scikit-learn

from sklearn.feature_selection import mutual_info_classif
# Create mutual_info_classif object to calculate mutual information

MI_score = mutual_info_classif(X_data, y_data, random_state=0)



# Print the name and mutual information score of each feature

for feature in zip(feature_names, MI_score):

    print(feature)
# Create a bar chart for visualizing the mutual information scores

plt.figure(figsize=(4,4))

plt.bar(x=feature_names, height=MI_score, color='tomato')

plt.xticks(rotation='vertical')

plt.ylabel('Mutual Information Score')

plt.title('Mutual Information Score Comparison')



plt.show()
# Import SelectKBest from Scikit-learn

from sklearn.feature_selection import SelectKBest
# Create a SelectKBest object

skb = SelectKBest(score_func=f_classif, # Set f_classif as our criteria to select features

                  k=2)                  # Select top two features based on the criteria



# Train and transform the dataset according to the SelectKBest

X_data_new = skb.fit_transform(X_data, y_data)



# Print the results

print('Number of features before feature selection: {}'.format(X_data.shape[1]))

print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))
# Print the name of the selected features

for feature_list_index in skb.get_support(indices=True):

    print('- ' + feature_names[feature_list_index])
# Import ExhaustiveFeatureSelector from Mlxtend

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
# Import logistic regression from Scikit-learn

from sklearn.linear_model import LogisticRegression
# Create a logistic regression classifier

lr = LogisticRegression()



# Create an EFS object

efs = EFS(estimator=lr,        # Use logistic regression as the classifier/estimator

          min_features=1,      # The minimum number of features to consider is 1

          max_features=4,      # The maximum number of features to consider is 4

          scoring='accuracy',  # The metric to use to evaluate the classifier is accuracy 

          cv=5)                # The number of cross-validations to perform is 5



# Train EFS with our dataset

efs = efs.fit(X_data, y_data)



# Print the results

print('Best accuracy score: %.2f' % efs.best_score_) # best_score_ shows the best score 

print('Best subset (indices):', efs.best_idx_)       # best_idx_ shows the index of features that yield the best score 

print('Best subset (corresponding names):', efs.best_feature_names_) # best_feature_names_ shows the feature names 

                                                                     # that yield the best score
# Transform the dataset

X_data_new = efs.transform(X_data)



# Print the results

print('Number of features before transformation: {}'.format(X_data.shape[1]))

print('Number of features after transformation: {}'.format(X_data_new.shape[1]))
# Show the performance of each subset of features

efs_results = pd.DataFrame.from_dict(efs.get_metric_dict()).T

efs_results.sort_values(by='avg_score', ascending=True, inplace=True)

efs_results
# Create a horizontal bar chart for visualizing 

# the performance of each subset of features

fig, ax = plt.subplots(figsize=(12,9))

y_pos = np.arange(len(efs_results))

ax.barh(y_pos, 

        efs_results['avg_score'],

        xerr=efs_results['std_dev'],

        color='tomato')

ax.set_yticks(y_pos)

ax.set_yticklabels(efs_results['feature_names'])

ax.set_xlabel('Accuracy')

plt.show()
# Import SequentialFeatureSelector from Mlxtend

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# Create a logistic regression classifier

lr = LogisticRegression()



# Create an SFS object

sfs = SFS(estimator=lr,       # Use logistic regression as our classifier

          k_features=(1, 4),  # Consider any feature combination between 1 and 4

          forward=True,       # Set forward to True when we want to perform SFS

          scoring='accuracy', # The metric to use to evaluate the classifier is accuracy 

          cv=5)               # The number of cross-validations to perform is 5



# Train SFS with our dataset

sfs = sfs.fit(X_data, y_data)



# Print the results

print('Best accuracy score: %.2f' % sfs.k_score_)   # k_score_ shows the best score 

print('Best subset (indices):', sfs.k_feature_idx_) # k_feature_idx_ shows the index of features 

                                                    # that yield the best score

print('Best subset (corresponding names):', sfs.k_feature_names_) # k_feature_names_ shows the feature names 

                                                                  # that yield the best score
# Transform the dataset

X_data_new = sfs.transform(X_data)



# Print the results

print('Number of features before transformation: {}'.format(X_data.shape[1]))

print('Number of features after transformation: {}'.format(X_data_new.shape[1]))
# Show the performance of each subset of features considered by SFS

sfs_results = pd.DataFrame.from_dict(sfs.subsets_).T 

sfs_results
# Create a horizontal bar chart for visualizing 

# the performance of each subset of features

fig, ax = plt.subplots(figsize=(6,2))

y_pos = np.arange(len(sfs_results))

ax.barh(y_pos, 

        sfs_results['avg_score'], 

        color='tomato')

ax.set_yticks(y_pos)

ax.set_yticklabels(sfs_results['feature_names'])

ax.set_xlabel('Accuracy')

plt.show()
# Create a logistic regression classifier

lr = LogisticRegression()



# Create an SBS object

sbs = SFS(estimator=lr,       # Use logistic regression as our classifier

          k_features=(1, 4),  # Consider any feature combination between 1 and 4

          forward=False,      # Set forward to False when we want to perform SBS

          scoring='accuracy', # The metric to use to evaluate the classifier is accuracy 

          cv=5)               # The number of cross-validations to perform is 5



# Train SBS with our dataset

sbs = sbs.fit(X_data.values, y_data, custom_feature_names=feature_names)



# Print the results

print('Best accuracy score: %.2f' % sbs.k_score_)   # k_score_ shows the best score 

print('Best subset (indices):', sbs.k_feature_idx_) # k_feature_idx_ shows the index of features 

                                                    # that yield the best score

print('Best subset (corresponding names):', sbs.k_feature_names_) # k_feature_names_ shows the feature names 

                                                                  # that yield the best score
# Transform the dataset

X_data_new = sbs.transform(X_data)



# Print the results

print('Number of features before transformation: {}'.format(X_data.shape[1]))

print('Number of features after transformation: {}'.format(X_data_new.shape[1]))
# Show the performance of each subset of features considered by SBS

sbs_results = pd.DataFrame.from_dict(sbs.subsets_).T

sbs_results
# Create a horizontal bar chart for visualizing 

# the performance of each subset of features

fig, ax = plt.subplots(figsize=(6,2))

y_pos = np.arange(len(sbs_results))

ax.barh(y_pos, 

        sbs_results['avg_score'], 

        color='tomato')

ax.set_yticks(y_pos)

ax.set_yticklabels(sbs_results['feature_names'])

ax.set_xlabel('Accuracy')

plt.show()
# Compare the selection generated by EFS, SFS, and SBS

print('Best subset by EFS:', efs.best_feature_names_)

print('Best subset by SFS:', sfs.k_feature_names_)

print('Best subset by SBS:', sbs.k_feature_names_)
# Import RandomForestClassifier from Scikit-learn

from sklearn.ensemble import RandomForestClassifier
# Import train_test_split from Scikit-learn

from sklearn.model_selection import train_test_split



# Split the dataset into 30% test and 70% training

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
# Create a random forest classifier

rfc = RandomForestClassifier(random_state=0, 

                             criterion='gini') # Use gini criterion to define feature importance



# Train the classifier

rfc.fit(X_train, y_train)



# Print the name and gini importance of each feature

for feature in zip(feature_names, rfc.feature_importances_): 

    print(feature)
from sklearn.feature_selection import SelectFromModel
# Create a random forest classifier

rfc = RandomForestClassifier(random_state=0, 

                             criterion='gini') # Use gini criterion to define feature importance



# Create a SelectFromModel object 

sfm = SelectFromModel(estimator=rfc, # Use random forest classifier to identify features

                      threshold=0.2) # that have an importance of more than 0.2



# Train the selector

sfm = sfm.fit(X_train, y_train)



# Print the names of the most important features

print('The most important features based on random forest classifier:')

for feature_list_index in sfm.get_support(indices=True):

    print('- ' + feature_names[feature_list_index])
# Transform the dataset

X_important_train = sfm.transform(X_train)

X_important_test = sfm.transform(X_test)



# Print the results

print('Number of features before transformation: {}'.format(X_train.shape[1]))

print('Number of features after transformation: {}'.format(X_important_train.shape[1]))
# Import accuracy_score from Scikit-learn

from sklearn.metrics import accuracy_score
# Create a random forest classifier

rfc_full = RandomForestClassifier(random_state=0, criterion='gini')



# Train the classifier using dataset with full features

rfc_full.fit(X_train, y_train)



# Make predictions

pred_full = rfc_full.predict(X_test)



# Generate accuracy score

print('The accuracy of classifier with full features: {:.2f}'.format(accuracy_score(y_test, pred_full)))
# Create a random forest classifier

rfc_lim = RandomForestClassifier(random_state=0, criterion='gini')



# Train the classifier with limited features

rfc_lim.fit(X_important_train, y_train)



# Make predictions

pred_lim = rfc_lim.predict(X_important_test)



# Generate accuracy score

print('The accuracy of classifier with limited features: {:.2f}'.format(accuracy_score(y_test, pred_lim)))