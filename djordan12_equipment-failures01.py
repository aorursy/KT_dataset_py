#### IMPORT LIBARIES



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle tabular data and matricies

import numpy as np

import pandas as pd



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

#### IMPORT DATA

train = pd.read_csv('../input/equipfailstest/equip_failures_training_set.csv', na_values='na')

test = pd.read_csv('../input/equipfailstest/equip_failures_test_set.csv', na_values='na')



# Add a new column 'data' to identify which data frame it came from

train['dataset'] = 'train'

test['dataset'] = 'test'



# Combine both train and test data sets

data = pd.concat([train, test], copy=False)
data.shape
#### EXPLORE DATA SET

data.head()
data.info()
data.describe(include='all')
data.isna().sum()
# Remove columns that have more than 70% of its data missing.

data = data.dropna(thresh=0.7*len(data), axis=1)

data.shape
# Don't fill dataset, id, or target columns

for col in data.columns:

    if col != ('dataset' or 'id' or 'target'):

        data[col]=data[col].fillna(data[col].median())
# Check if still missing data

data.isna().sum()
from sklearn import model_selection
# Split data set into X and y for training / testing purposes.

X = data[data.dataset=='train'].drop(['dataset', 'id', 'target'], axis=1)

y = data[data.dataset=='train']['target']



# X submit test is for submission file

X_submit_test = data[data.dataset=='test'].drop(['dataset', 'id', 'target'], axis=1)
# Splitting the final dataset into training and testing datasets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score
# Step 1) Define Model.

DTC_model = DecisionTreeClassifier(random_state=1)



# Step 2) Fit Model.

DTC_model.fit(X_train, y_train)
# Step 3) Model Predictions

DTC_model_pred = DTC_model.predict(X_test)



# Step 4) Evaulate

print('Decision Tree Classifier accuracy: {}'.format(accuracy_score(y_test, DTC_model_pred)))
### DTC Model

# output = DTC_model.predict(X_submit_test).astype(int)

# df_output = pd.DataFrame()

# df_output['id'] = test['id']

# df_output['target'] = output

# df_output.to_csv('submission.csv', index=False)
fig, axes = plt.subplots(1, 1, sharex="all", figsize=(15,15))

name = "DTC_Model"

classifier = DTC_model

indices = np.argsort(classifier.feature_importances_)[::-1][:40]

g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes)

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title(name + " feature importance")

# Import libraries

from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Step 1) Define Model.

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)



# Step 2) Fit Model.

adaDTC.fit(X_train, y_train)



# Step 3) Model Predictions

adaDTC_pred = adaDTC.predict(X_test)



# Step 4) Evaulate

print('AdaBoost Classifier accuracy: {}'.format(accuracy_score(y_test, adaDTC_pred)))
RFC = RandomForestClassifier()

RFC.fit(X_train, y_train)

RFC_pred = RFC.predict(X_test)

print('RFC accuracy: {}'.format(accuracy_score(y_test, RFC_pred)))
### RFC Model

output = RFC.predict(X_submit_test).astype(int)

df_output = pd.DataFrame()

df_output['id'] = test['id']

df_output['target'] = output

df_output.to_csv('rfc_submission.csv', index=False)