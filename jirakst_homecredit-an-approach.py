# Basic

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Common tools

import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from collections import Counter

'''

# Advanced visualization

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)

'''

# Model

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score

from sklearn.ensemble import VotingClassifier



# Configure Defaults

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.max_colwidth', -1)
# List all files

print(os.listdir('../input/'))
train = pd.read_csv('../input/application_train.csv')
train.shape
test = pd.read_csv('../input/application_test.csv')
test.shape
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.shape
sns.countplot(x='TARGET', data=train)

print(train.TARGET.sum()/train.TARGET.count())
train.info()
train.head()
# Number of categories within a categorical feature

train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Missing values

n = train.isnull().sum() / len(train)

n.sort_values(ascending=False).head(10)
# Number of features exceeding 1/6 of missing values.

sum(i>0.1667 for i in n)
# Age

(train['DAYS_BIRTH'] / -365).describe()
(train['DAYS_EMPLOYED'] / 365).describe()
train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
# Out of curiosity...

anom = train[train['DAYS_EMPLOYED'] == 365243]

non_anom = train[train['DAYS_EMPLOYED'] != 365243]

print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))

print('There are %d anomalous days of employment' % len(anom))
# Create new feature

train['DAYS_EMPLOYED_BOOL'] = train['DAYS_EMPLOYED'] == 365243

train['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace=True)
# Check orginal feature

train['DAYS_EMPLOYED'].plot.hist()
def detect_outliers(df,n,features):

    

    outlier_indices = []

    

    for col in features:

        Q1 = df[col].quantile(0.02)

        Q3 = df[col].quantile(0.98)

        IQR = Q3 - Q1

        

        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR )].index

        outlier_indices.extend(outliers)

        

    # Select observations with more than n outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   
# Return only float64 features

numerical_feature_mask = train.dtypes==float

numerical_cols = train.columns[numerical_feature_mask].tolist()
# Detect outliers

Outliers_to_drop = detect_outliers(train,2,numerical_cols)
# Number of outliers to drop

len(Outliers_to_drop)
# Remove outliers

train.drop(Outliers_to_drop, inplace=True)
# Save Id for the submission at the very end.

Id = test['SK_ID_CURR']
# Get marker

split = len(train)
# Merge into one dataset

data =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# We don't need the Id anymore now.

data.drop('SK_ID_CURR', axis=1, inplace=True)
data.shape
# Remove mostly sparse features

for f in data:

   if data[f].isnull().sum() / data.shape[0] >= 0.5: del data[f] # Or do a boolean flag here
# Check for TARGET data type

data['TARGET'].dtype
# Select columns due to theirs data type

float_col = data.select_dtypes('float').drop(['TARGET'], axis=1)

int_col = data.select_dtypes('int')

object_col = data.select_dtypes('object')
# Remove and impute numerical features

for f in float_col:

   if data[f].isnull().sum() / data.shape[0] > 0.1667: del data[f] # Remove 1/6+ of NANs

   else: data[f] = data[f].fillna(data[f].mean()) # Impute others with a mean value
# Impute default value into a numerical category

for i in int_col:

   data[i] = data[i].fillna(-1)
# Impute object type with a default

for o in object_col:

   data[o] = data[o].fillna('Unknown')
# Check

data.isnull().sum().sort_values(ascending=False).head(5)
data = pd.get_dummies(data, prefix_sep='_', drop_first=True) # Drop originall feature to avoid multi-collinearity
'''# Categorical mask

categorical_feature_mask = train.dtypes==object

# Get categorical columns

categorical_cols = train.columns[categorical_feature_mask].tolist()'''
'''# Instantiate LE

le = LabelEncoder()'''
'''# Apply LE

train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))'''
'''# Check

train[categorical_cols].head(10)'''
'''# Instantiate OHE

ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) #Can be enabled True for higher preformance'''
'''# Apply OHE

train_ohe = ohe.fit_transform(train) #an numpy array'''
#Split data

train_c = data[:split]

test_c = data[split:].drop(['TARGET'], axis=1)
from sklearn.model_selection import train_test_split



# Get variables for a model

x = train_c.drop(["TARGET"], axis=1)

y = train_c["TARGET"]



#Do train data splitting

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.22, random_state=101)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train) # Fit on training set only.



# Apply transform to both the training set and the test set.

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.decomposition import PCA



pca = PCA(.95)

pca.fit(X_train)
X_train = pca.transform(X_train)

X_test = pca.transform(X_test)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(solver = 'lbfgs').fit(X_train, y_train)

pred = lr.predict(X_test)

acc = lr.score(X_test, y_test)



print("Accuracy: ", acc)
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



predicts = cross_val_predict(lr, X_train, y_train, cv=3)

confusion_matrix(y_train, predicts)
from sklearn.metrics import precision_score, recall_score



print("Precision:", precision_score(y_train, predicts))

print("Recall:",recall_score(y_train, predicts))
from sklearn.metrics import f1_score



f1_score(y_train, predicts)
from sklearn.metrics import precision_recall_curve



y_scores = lr.predict_proba(X_train)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(y_train, y_scores)



def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
from sklearn.metrics import roc_curve



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)



def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
from sklearn.metrics import roc_auc_score
y_scores = lr.predict_proba(X_train)

y_scores = y_scores[:,1]
auroc = roc_auc_score(y_train, y_scores)

print("ROC-AUC Score:", auroc)