# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries

%matplotlib inline

import scipy.stats as stats

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#shape the dataset

print('This data frame has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))
#peek at dataset

df.sample(20)
#info dataset

df.info()
#numerical summary

pd.set_option('precision', 3)

df.loc[:, ['Time', 'Amount']].describe()
#visualizations of dataset - time and amount

plt.figure(figsize=(10,8))

plt.title('Distribution of Time Feature')

sns.distplot(df.Time)
# Distribution of monetory & value functions

plt.figure(figsize=(10,8))

plt.title('Distribution of monetory & value functions')

sns.distplot(df.Amount)
# fraud transactions vs. normal transactions 

counts = df.Class.value_counts()

normal = counts[0]

fraud = counts[1]

perc_normal = (normal/(normal+fraud))*100

perc_fraud = (fraud/(normal+fraud))*100

print('There were {} non-fraud transactions ({:.3f}%) and {} fraud transactions ({:.3f}%).'.format(normal, perc_normal, fraud, perc_fraud))
plt.figure(figsize=(8,6))

sns.barplot(x=counts.index, y=counts)

plt.title('Fraud Count vs. Normal Transactions')

plt.ylabel('Count')

plt.xlabel('Class (0:Normal, 1:Fraud)')
corr = df.corr()

corr
#heatmap of correlation

Correlation = df.corr()

plt.figure(figsize=(12,10))

heat = sns.heatmap(data=Correlation)

plt.title('Heatmap of Correlation')
#skewness of Correlation

skew_ = df.skew()

skew_
# Importing Sklearn

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler2 = StandardScaler()
#scaling time

scaled_time = scaler.fit_transform(df[['Time']])

flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]

scaled_time = pd.Series(flat_list1)
#scaling the amount of columns

scaled_amount = scaler2.fit_transform(df[['Amount']])

flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]

scaled_amount = pd.Series(flat_list2)
#concatenating newly created columns with original delta function

df = pd.concat([df, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)

df.sample(20)
#dropping dataset of old amount and time columns

df.drop(['Amount', 'Time'], axis=1, inplace=True)
# train - test split using numpy

mask = np.random.rand(len(df)) < 0.9

train = df[mask]

test = df[~mask]

print('Training Shape: {}\nTest Shape: {}'.format(train.shape, test.shape))
train.reset_index(drop=True, inplace=True)

test.reset_index(drop=True, inplace=True)
# random samples from normal transactions to fraud transactions

no_of_frauds = train.Class.value_counts()[1]

print('There are {} fraud transactions in the train data.'.format(no_of_frauds))
#randomly selecting 708 normal transactions

non_fraud = train[train['Class'] == 0]

fraud = train[train['Class'] == 1]
selected = non_fraud.sample(no_of_frauds)

selected.head(20)
#concatenating both into a subsample data set with equal class distribution

selected.reset_index(drop=True, inplace=True)

fraud.reset_index(drop=True, inplace=True)
subsample = pd.concat([selected, fraud])

len(subsample)
#shuffling our whole dataset

subsample = subsample.sample(frac=1).reset_index(drop=True)

subsample.head(20)
# Count of Fraud vs. Normal Transactions

new_counts = subsample.Class.value_counts()

plt.figure(figsize=(8,6))

sns.barplot(x=new_counts.index, y=new_counts)

plt.title('Count of Fraud vs. Normal Transactions')

plt.ylabel('Count')

plt.xlabel('Class (0:Normal, 1:Fraud)')
#taking a look at correlations once more

correlations = subsample.corr()

correlations = correlations

correlations
#negative correlations smaller than -0.5

correlations[correlations.Class < -0.5]
#positive correlations greater than 0.5

corr[corr.Class > 0.5]
#visualizing the features with high negative correlation

f, axes = plt.subplots(nrows=2, ncols=4, figsize=(26,16))

f.suptitle('Features With High Negative Correlation', size=35)

sns.boxplot(x="Class", y="V3", data=subsample, ax=axes[0,0])

sns.boxplot(x="Class", y="V9", data=subsample, ax=axes[0,1])

sns.boxplot(x="Class", y="V10", data=subsample, ax=axes[0,2])

sns.boxplot(x="Class", y="V12", data=subsample, ax=axes[0,3])

sns.boxplot(x="Class", y="V14", data=subsample, ax=axes[1,0])

sns.boxplot(x="Class", y="V16", data=subsample, ax=axes[1,1])

sns.boxplot(x="Class", y="V17", data=subsample, ax=axes[1,2])

f.delaxes(axes[1,3])
#visualizing the features with high positive correlation

f, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))

f.suptitle('Features With High Positive Correlation', size=20)

sns.boxplot(x="Class", y="V4", data=subsample, ax=axes[0])

sns.boxplot(x="Class", y="V11", data=subsample, ax=axes[1])
# removing extreme outliers

Q1 = subsample.quantile(0.25)

Q3 = subsample.quantile(0.75)

IQR = Q3 - Q1

df2 = subsample[~((subsample < (Q1 - 2.5 * IQR)) |(subsample > (Q3 + 2.5 * IQR))).any(axis=1)]
from sklearn.manifold import TSNE

X = df2.drop('Class', axis=1)

y = df2['Class']
#t-SNE

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
# t-SNE scatter plot

import matplotlib.patches as mpatches

f, ax = plt.subplots(figsize=(24,16))

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax.set_title('t-SNE', fontsize=14)

ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])
# Removing warnings

def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn
# train - test split dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Dividing the dataset into training and testing

X_train = X_train.values

X_validation = X_test.values

y_train = y_train.values

y_validation = y_test.values
# CVisualizing shape ans size of training and testing dataset into matrix

print('X_shapes:\n', 'X_train:', 'X_validation:\n', X_train.shape, X_validation.shape, '\n')

print('Y_shapes:\n', 'Y_train:', 'Y_validation:\n', y_train.shape, y_validation.shape)
# Importing necessery libraries

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
# Spot-Checking Algorithms

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('SVM', SVC()))

models.append(('XGB', XGBClassifier()))

models.append(('RF', RandomForestClassifier()))
#testing models

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')

    results.append(cv_results)

    names.append(name)

    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

    print(msg)
#Compare Algorithms result

fig = plt.figure(figsize=(12,10))

plt.title('Comparison of Classification Algorithms')

plt.xlabel('Algorithm')

plt.ylabel('ROC-AUC Score')

plt.boxplot(results)

ax = fig.add_subplot(111)

ax.set_xticklabels(names)

plt.show()
#visualizing RF

model = RandomForestClassifier(n_estimators=10)
# Training RF

model.fit(X_train, y_train)
# Extract single tree

estimator = model.estimators_[8]
from sklearn.tree import export_graphviz

# Export dot file of tree

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = X.columns.tolist(),

                class_names = ['0',' 1'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)
# Convert to png

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')