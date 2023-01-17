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
# Import useful libraries



import time

import re

import string

from numpy import mean

from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest, f_classif



import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from sklearn.utils.multiclass import type_of_target



from catboost import CatBoostClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from collections import Counter

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler



import warnings

warnings.filterwarnings('ignore')
# Read dataset



train_data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test_data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)

print('Test Data Shape: ', test_data.shape)

train_data.head()
train_data.isnull().sum()
train_data['response'].value_counts()
train_data.nunique()
fig, axes = plt.subplots(ncols = 2, figsize = (13, 3), dpi = 100)

plt.tight_layout()



train_data.groupby('response').count()['id'].plot(kind = 'pie', ax = axes[0], labels = ['Interested (87.7%)', 'Not Interested (12.1%)'])

sns.countplot(x = train_data['response'], hue = train_data['response'], ax = axes[1])



axes[0].set_ylabel('')

axes[1].set_ylabel('')

axes[1].set_xticklabels(['Interested (87.7%)', 'Not Interested (12.1%)'])

axes[0].tick_params(axis = 'x', labelsize = 8)

axes[0].tick_params(axis = 'y', labelsize = 8)

axes[1].tick_params(axis = 'x', labelsize = 8)

axes[1].tick_params(axis = 'y', labelsize = 8)



axes[0].set_title('Label Distribution in Training Set', fontsize = 8)

axes[1].set_title('Label Count in Training Set', fontsize =8)



plt.show()
# looking at the frequency of records by age



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 66))

train_data['age'].value_counts().head(66).plot.bar(color = color)

plt.title('Age distribution (Most policy holders are young. Age is highly skewed)', fontsize = 15)

plt.xticks(rotation = 90)

plt.show()
train_data['type'] = 'train'

test_data['type'] = 'test'



master_data = pd.concat([train_data, test_data])
plt.figure(figsize = (8, 5))

sns.distplot(master_data['annual_premium'])

plt.title('Annual Premium distribution (Highly skewed to the right)', fontsize = 15)

plt.show()
plt.figure(figsize = (15, 6))

sns.distplot(master_data.loc[(master_data['gender'] == 'Male'), 'age'], kde_kws = {"color": "b", "lw": 1, "label": "Male"})

sns.distplot(master_data.loc[(master_data['gender'] == 'Female'), 'age'], kde_kws = {"color": "r", "lw": 1, "label": "Female"})

plt.title('Age distribution by Gender', fontsize = 15)

plt.show()
plt.figure(figsize = (15, 6))

sns.distplot(master_data.loc[(master_data['gender'] == 'Male'), 'annual_premium'], kde_kws = {"color": "b", "lw": 1, "label": "Male"})

sns.distplot(master_data.loc[(master_data['gender'] == 'Female'), 'annual_premium'], kde_kws = {"color": "r", "lw": 1, "label": "Female"})

plt.title('Annual Premium distribution by Gender', fontsize = 15)

plt.show()
plt.figure(figsize = (15, 6))

sns.distplot(master_data.loc[(master_data['driving_license'] == 0), 'age'], kde_kws = {"color": "b", "lw": 1, "label": "Not Licensed for driving"})

sns.distplot(master_data.loc[(master_data['driving_license'] == 1), 'age'], kde_kws = {"color": "r", "lw": 1, "label": "Licensed for Driving"})

plt.title('Age distribution by Driving License', fontsize = 15)

plt.show()
plt.figure(figsize = (15, 6))

sns.distplot(master_data.loc[(master_data['driving_license'] == 0), 'annual_premium'], kde_kws = {"color": "b", "lw": 1, "label": "Not Licensed for driving"})

sns.distplot(master_data.loc[(master_data['driving_license'] == 1), 'annual_premium'], kde_kws = {"color": "r", "lw": 1, "label": "Licensed for Driving"})

plt.title('Annual Premium distribution by Driving License', fontsize = 15)

plt.show()
plt.figure(figsize = (18, 5))

sns.boxplot(master_data['annual_premium'])

plt.title('Annual Premium distribution (Highly skewed to the right)', fontsize = 15)

plt.show()
plt.figure(figsize = (8, 5))

sns.distplot(master_data['vintage'])

plt.title('No. of days customer was associated with the company', fontsize = 15)

plt.show()
# looking at the frequency of records by age



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 50))

train_data['policy_sales_channel'].value_counts().head(50).plot.bar(color = color)

plt.title('Top Policy Sales Channels', fontsize = 15)

plt.xticks(rotation = 90)

plt.show()
# looking at the frequency of records by sales channel



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 53))

train_data['region_code'].value_counts().head(53).plot.bar(color = color)

plt.title('Customers count by top regions', fontsize = 15)

plt.xticks(rotation = 90)

plt.show()
fig, axes = plt.subplots(ncols = 2, figsize = (13, 3), dpi = 100)

plt.tight_layout()



train_data.groupby('previously_insured').count()['id'].plot(kind = 'pie', ax = axes[0], labels = ['Insured Customers (54.1%)', 'Not Insured Customers (45.9%)'])

sns.countplot(x = train_data['previously_insured'], hue = train_data['previously_insured'], ax = axes[1])



axes[0].set_ylabel('')

axes[1].set_ylabel('')

axes[1].set_xticklabels(['Insured Customers (54.1%)', 'Not Insured Customers (45.9%)'])

axes[0].tick_params(axis = 'x', labelsize = 8)

axes[0].tick_params(axis = 'y', labelsize = 8)

axes[1].tick_params(axis = 'x', labelsize = 8)

axes[1].tick_params(axis = 'y', labelsize = 8)



axes[0].set_title('Label Distribution in Training Set', fontsize = 8)

axes[1].set_title('Label Count in Training Set', fontsize =8)



plt.show()
sns.countplot(data = master_data, x = 'driving_license', hue = 'gender')

plt.ylabel('Count')

plt.show()
# Unique values for all the columns

for col in train_data.columns[~(train_data.columns.isin(['age', 'id', 'region_code', 'annual_premium', 'policy_sales_channel', 'vintage']))].tolist():

    print(" Unique Values --> " + col, ':', len(train_data[col].unique()), ': ', train_data[col].unique())
gender = {'Male': 0, 'Female': 1}

driving_license = {0: 0, 1: 1}

previously_insured = {0: 1, 1: 0}

vehicle_age = {'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0}

vehicle_damage = {'Yes': 1, 'No': 0}



master_data['gender'] = master_data['gender'].map(gender)

master_data['driving_license'] = master_data['driving_license'].map(driving_license)

master_data['previously_insured'] = master_data['previously_insured'].map(previously_insured)

master_data['vehicle_age'] = master_data['vehicle_age'].map(vehicle_age)

master_data['vehicle_damage'] = master_data['vehicle_damage'].map(vehicle_damage)



master_data['policy_sales_channel'] = master_data['policy_sales_channel'].apply(lambda x: np.int(x))

master_data['region_code'] = master_data['region_code'].apply(lambda x: np.int(x))



master_data.head()
corrMatrix = master_data.corr()

sns.heatmap(corrMatrix, annot = True)

plt.show()
# Numerical columns

numerical_cols = ['age', 'vintage']



# categorical column 

cat_col = ['gender', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 'vehicle_damage', 'policy_sales_channel']



#master_data['policy_sales_channel'] = master_data['policy_sales_channel'].map(master_data['policy_sales_channel'].value_counts())

#master_data['region_code'] = master_data['region_code'].map(master_data['region_code'].value_counts())



ss = StandardScaler()

master_data[numerical_cols] = ss.fit_transform(master_data[numerical_cols])



mm = MinMaxScaler()

master_data[['annual_premium']] = mm.fit_transform(master_data[['annual_premium']])



master_data.head()
train_data = master_data.loc[(master_data['type'] == 'train')]

test_data = master_data.loc[(master_data['type'] == 'test')]



train_data = train_data.drop(['id', 'type'], axis = 1)

train_data['response'] = train_data['response'].apply(lambda x: np.int(x))



testIDs = test_data['id']

test_data = test_data.drop(['id', 'type', 'response'], axis = 1)

train_data.head()
for column in cat_col:

    test_data[column] = test_data[column].astype('str')
for column in cat_col:

    train_data[column] = train_data[column].astype('str')



train_data = train_data.drop(['vintage'], axis = 1)

test_data = test_data.drop(['vintage'], axis = 1)



X = train_data.drop(['response'], axis = 1)#.values

y = train_data['response']#.values



cat_cols = [0, 2, 3, 4, 5, 6, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 22, stratify = y, shuffle = True)



modelC = CatBoostClassifier()

modelC = modelC.fit(X_train, y_train, cat_features = cat_col, eval_set = (X_test, y_test), early_stopping_rounds = 10, verbose = 100)



predictions = [pred[1] for pred in modelC.predict_proba(X_test)]

print('Validation ROC AUC Score:', roc_auc_score(y_test, predictions, average = 'weighted'))
cat_pred = [pred[1] for pred in modelC.predict_proba(test_data)]

submissionC = pd.DataFrame(data = {'id': testIDs, 'Response': cat_pred})

submissionC.to_csv("catboost_v1.csv", index = False)

submissionC.head()
X = train_data.drop(['response'], axis = 1).values

y = train_data['response'].values
kfold, scores = KFold(n_splits = 5, shuffle = True, random_state = 22), list()

for train, test in kfold.split(X):

    X_train, X_test = X[train], X[test]

    y_train, y_test = y[train], y[test]



    model = LGBMClassifier(random_state = 22, max_depth = 7, n_estimators = 110, reg_lambda = 1.2, reg_alpha = 1.2, min_child_weight = 1, 

                           learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc', is_higher_better = 1, plot = True)

    model.fit(X_train, y_train)

    preds = [pred[1] for pred in model.predict_proba(X_test)]

    score = roc_auc_score(y_test, preds, average = 'weighted')

    scores.append(score)

    print('Validation ROC AUC:', score)

print("Average Validation ROC AUC: ", sum(scores)/len(scores))
yTest = model.predict(X_test)



fpr, tpr, thresholds = roc_curve(yTest.ravel(), y_test)

roc_auc = auc(fpr, tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label = 'AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
"""

model = LGBMClassifier(random_state = 22)



param_grid = {"learning_rate"    : [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],

              "max_depth"        : [4, 5, 6, 7, 8, 9, 10],

              "min_child_weight" : [1, 3, 5, 7],

              "gamma"            : [0.0, 0.1, 0.2 , 0.3, 0.4],

              "colsample_bytree" : [0.3, 0.4, 0.5 , 0.7],

              "n_estimators"     : [50, 70, 90, 100, 120, 150, 200, 250, 300, 350, 400, 450],

              'reg_alpha'        : [1,1.2],

              'reg_lambda'       : [1,1.2,1.4]

              }



kfold = KFold(n_splits = 6, shuffle = True, random_state = 22)



grid_search = RandomizedSearchCV(model, param_distributions = param_grid, scoring = "accuracy", n_jobs  = -1, cv = kfold, verbose = 1)

grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



"""
bestLGB = LGBMClassifier(random_state = 22, max_depth = 7, n_estimators = 110, reg_lambda = 1.2, reg_alpha = 1.2, min_child_weight = 1,

                         learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5)

bestLGB.fit(X_train, y_train)

y_pred = bestLGB.predict_proba(X_test)
Preds = [predClass[1] for predClass in model.predict_proba(test_data.values)]
submission = pd.DataFrame(data = {'id': testIDs, 'Response': Preds})

submission.to_csv('cross_sell_v8.csv', index = False)

submission.head()