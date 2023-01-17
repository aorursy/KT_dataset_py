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
# import libraries

import boto3, re, sys, math, json, os, urllib.request

import numpy as np                                

import pandas as pd                               

import matplotlib.pyplot as plt                   

from IPython.display import Image                 

from IPython.display import display               

from time import gmtime, strftime                 
try:

  model_data = pd.read_csv('../input/machine_event.csv',index_col=0)

  print('Success: Data loaded into dataframe.')

except Exception as e:

    print('Data load error: ',e)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Imported Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections





# Other Libraries

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")



model_data.head()
model_data.describe()
# Good No Null Values!

model_data.isnull().sum().max()
model_data.columns
print('No event', round(model_data['event'].value_counts()[0]/len(model_data) * 100,2), '% of the dataset')

print('event', round(model_data['event'].value_counts()[1]/len(model_data) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]



sns.countplot('event', data=model_data, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
# find the corelation between inputs



num_cols = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9']



corr = model_data[num_cols].corr()



# plot heatmap

sns.heatmap(corr, 

            xticklabels=corr.columns.values, yticklabels=corr.columns.values,

            cmap=sns.light_palette("navy"),

           )

plt.show()
# We can see that feature 1 and feature 2 are strongly correlated and feature 6 and feature 9 are strongly correlated.

# We will keep only 1 of the correlated datasets.



num_cols_2 = ['event','feature1','feature3','feature4','feature5','feature6','feature7','feature8']

              

model_data = model_data[num_cols_2]
fig, ax = plt.subplots(1, 2, figsize=(18,4))



feature3_val = model_data['feature3'].values

feature4_val = model_data['feature4'].values



sns.distplot(feature3_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Feature3', fontsize=14)

ax[0].set_xlim([min(feature3_val), max(feature3_val)])



sns.distplot(feature4_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Feature4', fontsize=14)

ax[1].set_xlim([min(feature4_val), max(feature4_val)])



plt.show()
# We are going to scale feature_3 and feature_4 which need to be scaled to the other features.



# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



model_data['scaled_feature3'] = rob_scaler.fit_transform(model_data['feature3'].values.reshape(-1,1))

model_data['scaled_feature4'] = rob_scaler.fit_transform(model_data['feature4'].values.reshape(-1,1))



model_data.drop(['feature3','feature4'], axis=1, inplace=True)

scaled_feature3 = model_data['scaled_feature3']

scaled_feature4 = model_data['scaled_feature4']



# feature3 and feature4 are Scaled!



model_data.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No event', round(model_data['event'].value_counts()[0]/len(model_data) * 100,2), '% of the dataset')

print('event', round(model_data['event'].value_counts()[1]/len(model_data) * 100,2), '% of the dataset')



X = model_data.drop('event', axis=1)

y = model_data['event']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



model_data = model_data.sample(frac=1)



# amount of event classes in 492 rows.

event_df = model_data.loc[model_data['event'] == 1]

non_event_df = model_data.loc[model_data['event'] == 0][:492]



normal_distributed_df = pd.concat([event_df, non_event_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()


print('Distribution of the Classes in the subsample dataset')

print(new_df['event'].value_counts()/len(new_df))



sns.countplot('event', data=new_df, palette=colors)

plt.title('Equally Distributed event', fontsize=14)

plt.show()
# New_df is from the random undersample data (fewer instances)

X = new_df.drop('event', axis=1)

y = new_df['event']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))



# PCA Implementation

t0 = time.time()

X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("PCA took {:.2} s".format(t1 - t0))



# TruncatedSVD

t0 = time.time()

X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

t1 = time.time()

print("Truncated SVD took {:.2} s".format(t1 - t0))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))

# labels = ['No Fraud', 'Fraud']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

red_patch = mpatches.Patch(color='#AF0000', label='Fraud')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])





# PCA scatter plot

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax2.set_title('PCA', fontsize=14)



ax2.grid(True)



ax2.legend(handles=[blue_patch, red_patch])



# TruncatedSVD scatter plot

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax3.set_title('Truncated SVD', fontsize=14)



ax3.grid(True)



ax3.legend(handles=[blue_patch, red_patch])



plt.show()
# Undersampling before cross validating (prone to overfit)

X = new_df.drop('event', axis=1)

y = new_df['event']
# Our data is already scaled we should split our training and test sets

from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
events = y_test

predictions = clf.predict(X_test)
from sklearn.metrics import classification_report



y_true = events

y_pred = predictions



print(classification_report(y_true, y_pred))
# Class count

count_class_0, count_class_1 = new_df.event.value_counts()



# Divide by class

df_class_0 = new_df[new_df['event'] == 0]

df_class_1 = new_df[new_df['event'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)



print('Random under-sampling:')

print(df_test_under.event.value_counts())



df_test_under.event.value_counts().plot(kind='bar', title='Count (event)');
df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)



print('Random over-sampling:')

print(df_test_over.event.value_counts())



df_test_over.event.value_counts().plot(kind='bar', title='Count (target)');
import imblearn
from sklearn.datasets import make_classification



X, y = make_classification(

    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],

    n_informative=3, n_redundant=1, flip_y=0,

    n_features=20, n_clusters_per_class=1,

    n_samples=100, random_state=10

)



df = pd.DataFrame(X)

df['target'] = y

df.target.value_counts().plot(kind='bar', title='Count (target)');
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X = pca.fit_transform(X)



plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(return_indices=True)

X_rus, y_rus, id_rus = rus.fit_sample(X, y)



print('Removed indexes:', id_rus)



plot_2d_space(X_rus, y_rus, 'Random under-sampling')
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)



print(X_ros.shape[0] - X.shape[0], 'new random picked points')



plot_2d_space(X_ros, y_ros, 'Random over-sampling')
from imblearn.under_sampling import TomekLinks



tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, id_tl = tl.fit_sample(X, y)



print('Removed indexes:', id_tl)



plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')
from imblearn.under_sampling import ClusterCentroids



cc = ClusterCentroids(ratio={0: 10})

X_cc, y_cc = cc.fit_sample(X, y)



plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)



plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, y)



plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
# Undersampling before cross validating (prone to overfit)

X = X_smt

y = y_smt
# Our data is already scaled we should split our training and test sets

from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
clf.score(X,y)
events = y_test

predictions = clf.predict(X_test)
from sklearn.metrics import classification_report



y_true = events

y_pred = predictions



print(classification_report(y_true, y_pred))
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs',

                       multi_class='multinomial').fit(X, y)

clf.score(X, y)