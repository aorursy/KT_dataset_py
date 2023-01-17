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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

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

from imblearn.datasets import fetch_datasets

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")





df = pd.read_csv('../input/creditcard.csv')

df.head()
df.describe()
df.isnull().sum().max()
df.columns
# The classes are heavily skewed we need to solve this issue later.

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]



sns.countplot('Class', data=df, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = df['Amount'].values

time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()


index = 1

for k in range(1, 9):

    fig, ax = plt.subplots(1, 3, figsize=(18,4))



    for i in range(1, 4):

        

        currentFeature = 'V' + str(index)



        feature = df[currentFeature].values



        sns.distplot(feature, ax=ax[i-1], color='r')

        ax[i-1].set_title('Distribution ' + currentFeature, fontsize=14)

        ax[i-1].set_xlim([min(feature), max(feature)])

        index += 1

    plt.show()

    
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = df['Amount'].values

featurePositiveAmount = df[df.Class == 0]['Amount'].values

featureNegativeAmount = df[df.Class == 1]['Amount'].values

    

time_val = df['Time'].values

featurePositiveTime = df[df.Class == 0]['Time'].values

featureNegativeTime = df[df.Class == 1]['Time'].values



sns.distplot(featurePositiveAmount, ax=ax[0], color='r')

sns.distplot(featureNegativeAmount, ax=ax[0], color='b')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(featurePositiveTime, ax=ax[1], color='r')

sns.distplot(featureNegativeTime, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()
index = 1

for k in range(1, 9):

    fig, ax = plt.subplots(1, 3, figsize=(18,4))



    for i in range(1, 4):

        

        currentFeature = 'V' + str(index)

        feature = df[currentFeature].values

        featurePositive = df[df.Class == 0][currentFeature].values

        featureNegative = df[df.Class == 1][currentFeature].values



        sns.distplot(featurePositive, ax=ax[i-1], color='r')

        sns.distplot(featureNegative, ax=ax[i-1], color='b')

        ax[i-1].set_title('Distribution ' + currentFeature, fontsize=14)

        ax[i-1].set_xlim([min(feature), max(feature)])

        index += 1

    plt.show()
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Time','Amount'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

xtrain = original_Xtrain.values

xtest = original_Xtest.values



ytrain = original_ytrain.values

ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(ytest, return_counts=True)

print('-' * 100)

print('Label Distributions: \n')

print('-' * 100)

print('Train distribution:')

print(train_unique_label, train_counts_label )

print(train_counts_label/ len(original_ytrain))

print('')

print('Test distribution:')

print(test_unique_label, test_counts_label)

print(test_counts_label/ len(original_ytest))

print('-' * 100)







classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
for key, classifier in classifiers.items():

    classifier.fit(xtrain, ytrain)

    ypred = classifier.predict(xtest)

    print("Classifiers: ", classifier.__class__.__name__, " has the following scores")

    print('Recall Score: {:.2f}'.format(recall_score(ytest, ypred)))

    print('Precision Score: {:.2f}'.format(precision_score(ytest, ypred)))

    print('F1 Score: {:.2f}'.format(f1_score(ytest, ypred)))

    print('Accuracy Score: {}'.format(accuracy_score(ytest, ypred)))

    

    fig, ax = plt.subplots(1, 1,figsize=(22,12))

        



    cf = confusion_matrix(ytest, ypred)



    sns.heatmap(cf, annot=True, cmap=plt.cm.copper)

    plt.show()
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



df = df.sample(frac=1) # Return a random sample of items from an axis of object.



# amount of fraud classes 492 rows.

fraud_df_train = df.loc[df['Class'] == 1][0:394]

non_fraud_df_train = df.loc[df['Class'] == 0][0:394]



fraud_df_test = df.loc[df['Class'] == 1][394:]

non_fraud_df_test = df.loc[df['Class'] == 0][394:394+int(len(df)/5)]



normal_distributed_df = pd.concat([fraud_df_train, non_fraud_df_train])



# Shuffle dataframe rows

new_df_train = normal_distributed_df.sample(frac=1, random_state=42)

new_df_train.head()





normal_distributed_df_test = pd.concat([fraud_df_test, non_fraud_df_test])

# Shuffle dataframe rows

new_df_test = normal_distributed_df_test.sample(frac=1, random_state=42)

new_df_test.head()
print(len(new_df_test))
print('Distribution of the Classes in the subsample dataset')

print(new_df_train['Class'].value_counts()/len(new_df_train))





sns.countplot('Class', data=new_df_train, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
print('Distribution of the Classes in the subsample testset dataset')

print(new_df_test['Class'].value_counts()/len(new_df_test))





sns.countplot('Class', data=new_df_train, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
Xtrain = new_df_train.drop('Class', axis=1)

ytrain = new_df_train['Class']



Xtest = new_df_test.drop('Class', axis=1)

ytest = new_df_test['Class']
# Turn the values into an array for feeding the classification algorithms.

Xtrain = Xtrain.values

Xtest = Xtest.values

ytrain = ytrain.values

ytest = ytest.values
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
# Wow our scores are getting even high scores even when applying cross validation.



for key, classifier in classifiers.items():

    classifier.fit(Xtrain, ytrain)

    ypred = classifier.predict(Xtest)

    print("Classifiers: ", classifier.__class__.__name__, " has the following scores")

    print('Recall Score: {:.2f}'.format(recall_score(ytest, ypred)))

    print('Precision Score: {:.2f}'.format(precision_score(ytest, ypred)))

    print('F1 Score: {:.2f}'.format(f1_score(ytest, ypred)))

    print('Accuracy Score: {}'.format(accuracy_score(ytest, ypred)))

    

    fig, ax = plt.subplots(1, 1,figsize=(22,12))

        



    cf = confusion_matrix(ytest, ypred)



    sns.heatmap(cf, annot=True, cmap=plt.cm.copper)

    plt.show()
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(Xtrain, ytrain)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_

print(log_reg)



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(Xtrain, ytrain)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_

print(knears_neighbors)



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(Xtrain, ytrain)



# SVC best estimator

svc = grid_svc.best_estimator_

print(svc)



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(Xtrain, ytrain)



# tree best estimator

tree_clf = grid_tree.best_estimator_

print(tree_clf)



trainedClassifiers = [log_reg, knears_neighbors, svc, tree_clf]
for classifier in trainedClassifiers:

 

    print(classifier)

    ypred = classifier.predict(Xtest)

    

    print('Recall Score: {:.2f}'.format(recall_score(ytest, ypred)))

    print('Precision Score: {:.2f}'.format(precision_score(ytest, ypred)))

    print('F1 Score: {:.2f}'.format(f1_score(ytest, ypred)))

    print('Accuracy Score: {}'.format(accuracy_score(ytest, ypred)))

    

    fig, ax = plt.subplots(1, 1,figsize=(22,12))

        



    cf = confusion_matrix(ytest, ypred)



    sns.heatmap(cf, annot=True, cmap=plt.cm.copper)

    plt.show()

     
from imblearn.over_sampling import SMOTE

 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

xtrain = original_Xtrain.values

xtest = original_Xtest.values

ytrain = original_ytrain.values

ytest = original_ytest.values
sm = SMOTE(random_state=42, ratio='minority')

xtrainSmote, ytrainSmote = sm.fit_resample(xtrain, ytrain)
train_unique_label, train_counts_label = np.unique(ytrainSmote, return_counts=True)

test_unique_label, test_counts_label = np.unique(ytest, return_counts=True)

print('-' * 100)

print('Label Distributions: \n')

print('-' * 100)

print('Train distribution:')

print(train_unique_label, train_counts_label )

print(train_counts_label/ len(ytrainSmote))

print('')

print('Test distribution:')

print(test_unique_label, test_counts_label)

print(test_counts_label/ len(ytest))

print('-' * 100)
classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
for key, classifier in classifiers.items():

    classifier.fit(xtrainSmote, ytrainSmote)

    ypred = classifier.predict(xtest)

    print("Classifiers: ", classifier.__class__.__name__, " has the following scores")

    print('Recall Score: {:.2f}'.format(recall_score(ytest, ypred)))

    print('Precision Score: {:.2f}'.format(precision_score(ytest, ypred)))

    print('F1 Score: {:.2f}'.format(f1_score(ytest, ypred)))

    print('Accuracy Score: {}'.format(accuracy_score(ytest, ypred)))

    

    fig, ax = plt.subplots(1, 1,figsize=(22,12))

        



    cf = confusion_matrix(ytest, ypred)



    sns.heatmap(cf, annot=True, cmap=plt.cm.copper)

    plt.show()
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy
n_inputs = xtrainSmote.shape[1]



oversample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
oversample_model.fit(xtrainSmote, ytrainSmote, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
ypred = oversample_model.predict(xtest, batch_size=200, verbose=0)
ypred = ypred.argmax(axis=1)


print('Recall Score: {:.2f}'.format(recall_score(ytest, ypred)))

print('Precision Score: {:.2f}'.format(precision_score(ytest, ypred)))

print('F1 Score: {:.2f}'.format(f1_score(ytest, ypred)))

print('Accuracy Score: {}'.format(accuracy_score(ytest, ypred)))

    

fig, ax = plt.subplots(1, 1,figsize=(22,12))

        



cf = confusion_matrix(ytest, ypred)



sns.heatmap(cf, annot=True, cmap=plt.cm.copper)

plt.show()