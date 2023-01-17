import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import itertools

import sklearn

import numpy as np
train_df = pd.read_csv("../input/data_set_ALL_AML_train.csv")

test_df = pd.read_csv("../input/data_set_ALL_AML_independent.csv")

validation_df = pd.read_csv("../input/actual.csv")

train_df.head()
# removing all call columns from data frame

train_columns = [col for col in train_df if "call" not in col]

test_columns = [col for col in test_df if "call" not in col]

train_adjusted = train_df[train_columns]

test_adjusted = test_df[test_columns]

train_adjusted.head()
#transposing data frames

transposed_train = train_adjusted.T

transposed_test = test_adjusted.T

transposed_train.head()
predictors = pd.concat([transposed_train, transposed_test], axis = 0)

predictors = predictors.drop(['Gene Description', 'Gene Accession Number'])

predictors.columns = transposed_train.iloc[0]
#resetting indices of both predictor and validation data frames so they can be combined

vd = validation_df.reset_index(drop = True)

pr = predictors.reset_index(drop = True)

#combining validation and predictor data frames

combined = pd.concat([pr, vd], axis = 1)

#finding most expressed genes in combined data dataframe

outcomes = combined.groupby('cancer').size()

outcomes.plot(kind = 'bar')
highest = combined.mean().abs().sort_values(ascending = False)

plt.figure(figsize=(10, 8))

highest.head(10).plot(kind = 'bar')

plt.title('10 Genes Highest Expression Levels Both Cancer Types')

plt.ylabel('Expression Levels (au)')
c_ALL = combined[combined.cancer == 'ALL']

highest_ALL = c_ALL.mean().abs().sort_values(ascending = False)

plt.figure(figsize=(10, 8))

highest_ALL.head(10).plot(kind = 'bar')

plt.title('10 Genes Highest Expression Levels ALL')

plt.ylabel('Expression Levels (au)')
c_AML = combined[combined.cancer == 'AML']

highest_AML = c_AML.mean().abs().sort_values(ascending = False)

plt.figure(figsize=(10, 8))

highest_AML.head(10).plot(kind = 'bar')

plt.title('10 Genes Highest Expression Levels AML')

plt.ylabel('Expression Levels (au)')
train_no_acc = transposed_train.drop(["Gene Accession Number","Gene Description"]).apply(pd.to_numeric)

test_no_acc = transposed_test.drop(["Gene Accession Number", "Gene Description"]).apply(pd.to_numeric)

predictors_no_acc = predictors.drop(['Gene Accession Number'], axis = 1).apply(pd.to_numeric)
#resetting indices for test and train data frames

train_no_acc = train_no_acc.reset_index(drop = True)

test_no_acc = test_no_acc.reset_index(drop = True)
#creating data frames for both test and train data validation

validation_train = validation_df[validation_df.patient <= 38].reset_index(drop = True)

validation_test = validation_df[validation_df.patient > 38].reset_index(drop = True)

validation_test.head()
# combining predictor and validation set data

train = pd.concat([validation_train, train_no_acc], axis = 1)

test = pd.concat([validation_test, test_no_acc], axis = 1)
#creating sample data frames from original for model creation

train_sample = train.iloc[:,2:].sample(n=200, axis=1)

test_sample = test.iloc[:,2:].sample(n=200, axis=1)

test_sample.head()
train_sample.plot(kind="hist", legend=None, bins=20, color='k')

train_sample.plot(kind="kde", legend=None);
from sklearn import preprocessing

scaled = pd.DataFrame(preprocessing.scale(train_sample))

scaled.plot(kind="hist", legend=None, bins=20, color='k')

scaled.plot(kind="kde", legend=None);
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



sample_scaled = StandardScaler().fit_transform(train_sample)

pca = PCA(n_components = 30)

pca.fit(sample_scaled)



cum_sum = pca.explained_variance_ratio_.cumsum()

cum_sum = cum_sum*100



fix, ax = plt.subplots(figsize = (8, 8))

plt.bar(range(30), cum_sum, color = 'r',alpha=0.5)

plt.title('PCA Analysis')

plt.ylabel('cumulative explained variance')

plt.xlabel('number of components')

plt.locator_params(axis='y', nbins=20)

#training and test samples are created and scaled for model creation

X_train = StandardScaler().fit_transform(train_no_acc)

X_test = StandardScaler().fit_transform(test_no_acc)

y_train = validation_train['cancer']

y_test = validation_test['cancer']
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score





def pipeline_PCA_GLM(components):

    accuracy_chart = []

    for i in components:

        steps = [('pca', PCA(n_components = i)),

        ('estimator', LogisticRegression())]

        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)

        predictions = pipe.predict(X_test)

        accuracy_chart.append(accuracy_score(y_test,predictions))

    return accuracy_chart

n_components = range(1,30)

accuracy_chart = pipeline_PCA_GLM(n_components)
plt.figure(figsize=(10, 8))

plt.bar(n_components, accuracy_chart)

plt.ylim(0,1)

plt.xlim(0,30)

plt.locator_params(axis='y', nbins=20)

plt.locator_params(axis = 'x', nbins = 30)

plt.ylabel("Accuracy")

plt.xlabel("Number of Components")
#KNN

from sklearn.neighbors import KNeighborsClassifier

def knn_pred(train_predictors, train_outcome, k_range, test_predictors):

    #train_predictors and train_outcome should both be from training split while test_predictors should be from test split

    y_pred = []

    for i in k_range:

        knn = KNeighborsClassifier(n_neighbors = i)

        knn.fit(train_predictors, train_outcome)

        y_pred.append(knn.predict(test_predictors))

    return y_pred





#function compares KNN accuracy at different levels of K

def knn_accuracy(pred, k_range, test_outcome):

    #pred represents predicted values while test_outcome represents the values from the test set

    accuracy_chart = []

    for i in range(len(k_range)):

        accuracy_chart.append((sklearn.metrics.accuracy_score(test_outcome, pred[i])))

    return accuracy_chart

        
train_range = range(2, 20, 2)

sample_pred = knn_pred(X_train, y_train, train_range, X_test)

accuracy = knn_accuracy(sample_pred, train_range, y_test)

plt.figure(figsize=(10, 8))

plt.bar(train_range, accuracy)

plt.ylim(0,1)

plt.xlim(0,20)

plt.locator_params(axis='y', nbins=20)

plt.locator_params(axis = 'x', nbins = 10)

plt.ylabel("Accuracy")

plt.xlabel("Number of Neighborhoods")