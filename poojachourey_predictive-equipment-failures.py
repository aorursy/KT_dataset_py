#importing the required libraries



import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline

import warnings

from collections import Counter

warnings.filterwarnings('ignore')
#loading the given training dataset



train_set = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')
#exploring first few rows of training data



train_set.head()
#reading the given test data



test_set = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')
#exploring first few rows of test data



test_set.head()
# Counting the occurence of each target value - 0 and 1

g = sns.countplot(x='target', data = train_set)

plt.xlabel('Target')

plt.ylabel('Number of Records')
# Concatenating train_set and test_set to clean both the sets together



data = pd.concat(objs=[train_set, test_set], axis=0).reset_index(drop=True)
# Length of training data so that later we can split our training and test data



train_len = len(train_set)
# Replacing string 'na' with NaN values



data = data.replace('na', np.NaN)
# Converting all measures to numerical data type

for col in data.columns:

    if col not in ['id', 'target']:

        data[col] = data[col].astype(np.float)
# Checking for unique values in each column in full dataset



unique_data = data.nunique().reset_index()

unique_data.columns = ['Name','Unique_Count']
# Checking the columns which have less than 2 unique values across the training and test data sets because columns with 

# constant values do not support our model's prediction



unique_data[unique_data.Unique_Count < 2]
# Checking for null values across each column in dataset



null_df = data.isna().sum().reset_index()

null_df.columns = ['Name', 'Unique_Count']
# sorting the columns having null values



null_df.sort_values('Unique_Count', ascending=False).head(5)
#Outlier detection - Finding rows with more than two outlier values in columns



def outliers(df, n, features):

    

    outlier_indices = []

    

    for col in features:

        

        #1st quartile

        Q1 = np.percentile(df[col], 25)

        

        #3rd quartile

        Q3 = np.percentile(df[col], 75)

        

        #inter quartile range

        IQR = Q3 - Q1

        

        #identify index of outlier rows

        outlierlist = df[(df[col] < (Q1 - (1.5 * IQR))) | (df[col] > (Q3 + (1.5 * IQR)))].index

        

        outlier_indices.extend(outlierlist)

        

    #selecting rows with more than two outliers

    outlier_indices = Counter(outlier_indices)

    multipleoutliers = list(k for k,v in outlier_indices.items() if v > n)

    

    return multipleoutliers



#detect outliers from Age, SibSp, Fare and Parch

finaloutliers = outliers(data, 2, data.columns)
#outlier detection

data.iloc[finaloutliers]
# Dropping sensor54_measure as it has constant values + nulls



data.drop(columns=['sensor54_measure'], axis=1, inplace=True)
# Removing columns which mostly have null values - more than 75%



data.drop(columns=['sensor43_measure', 'sensor42_measure', 'sensor41_measure', 'sensor40_measure', 'sensor2_measure', \

                  'sensor39_measure', 'sensor38_measure', 'sensor68_measure'], axis=1, inplace = True)
# Replacing NaN with median values



for col in data.columns:

    if col not in ['id','target']:

        data[col] = data[col].fillna(data[col].median())
data.isnull().sum().sort_values(ascending=False).head(5)
# Correlation matrix between highly correlated varaibles and target value

fig, ax = plt.subplots(figsize = (18, 18))

g = sns.heatmap(data[['sensor104_measure','sensor103_measure','sensor10_measure','sensor11_measure','sensor12_measure','sensor13_measure',

     'sensor14_measure','sensor15_measure','sensor46_measure','sensor27_measure','sensor31_measure',

     'sensor32_measure','sensor33_measure',

      'sensor44_measure','sensor48_measure','sensor49_measure','sensor59_measure',

     'sensor53_measure', 'sensor78_measure', 'sensor72_measure','sensor87_measure','sensor88_measure','sensor89_measure',

     'sensor8_measure', 'sensor90_measure', 'sensor91_measure', 'sensor94_measure', 'sensor95_measure', 'target']].corr(), annot=True, ax=ax)
data['sensor_1415'] = data['sensor14_measure'] - data['sensor15_measure']

data['sensor_7872'] = data['sensor78_measure'] - data['sensor72_measure']

data['sensor3214'] = data['sensor32_measure'] - data['sensor14_measure']

data['sensor148'] = data['sensor14_measure'] - data['sensor8_measure']

data['sensor4615'] = data['sensor46_measure'] - data['sensor15_measure']

data['sensor815'] = data['sensor15_measure'] - data['sensor8_measure']

data['sensor468'] = data['sensor46_measure'] - data['sensor8_measure']

data['sensor278'] = data['sensor27_measure'] - data['sensor8_measure']

data['sensor8933'] = data['sensor89_measure'] - data['sensor33_measure']

data['sensor9495'] = data['sensor94_measure'] - data['sensor95_measure']

data['sensor1427'] = data['sensor14_measure'] - data['sensor27_measure']
# Dropping sensor32_measure as it mostly duplicates sensor8_measure

data.drop(columns=['sensor32_measure'], axis=1, inplace=True)
data.head()
# Dropping id feature as it does not affect model performance

data.drop(columns=['id'], axis=1, inplace=True)
# train_set and test_set split

train_set = data[:train_len]

test_set = data[train_len:]
# X and y split

X = train_set.drop(labels=['target'], axis=1)

y = train_set['target']
# Train and val split



from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)
# RF classifier



rfr = RandomForestClassifier(n_estimators = 100, random_state=0, n_jobs=4, class_weight={0:1,1:2}, verbose=1)

rfr.fit(X_train,y_train)
#Predicting validation set results

y_pred = rfr.predict(X_val)
#Checking f1 score based on validation set results

from sklearn.metrics import f1_score

f1_score(y_val, y_pred)
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, loss='deviance', verbose=1)
# Fitting with train values and prediciting for validation set

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_val)
# Checking f1 score for GBDT model using validation set results

from sklearn.metrics import f1_score

f1_score(y_val, y_pred_gb)
# Ada Boost classifier

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

ada.fit(X_train, y_train)
# Predicting validation set results

y_pred_ada = ada.predict(X_val)



#Checking f1 score

f1_score(y_val, y_pred_ada)
test_set.drop(columns=['target'], axis=1, inplace=True)
def finalpred(testset):

    finalpred = rfr.predict(testset)

    

    prediction = pd.Series(finalpred, name = 'target')

    test_id = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')

    submission = pd.concat([test_id['id'], prediction], axis = 1)

    submission['target'] = submission['target'].astype(np.int)

    submission.to_csv('finalsub.csv', index=False)
finalpred(test_set)