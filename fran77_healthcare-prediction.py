# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')
train.head()
len(train.Hospital_code.unique())
plt.figure(figsize=(12, 6))

sns.countplot(train.Hospital_type_code)
plt.figure(figsize=(12, 6))

sns.countplot(train.City_Code_Hospital)
plt.figure(figsize=(12, 6))

sns.countplot(train.Hospital_region_code)
plt.figure(figsize=(12, 6))

sns.countplot(train['Available Extra Rooms in Hospital']) 
plt.figure(figsize=(12, 6))

sns.countplot(train.Department) 
plt.figure(figsize=(12, 6))

sns.countplot(train.Ward_Type) 
plt.figure(figsize=(12, 6))

sns.countplot(train.Ward_Facility_Code) 
plt.figure(figsize=(12, 6))

sns.countplot(train['Bed Grade'])
len(train.patientid.unique())
plt.figure(figsize=(12, 6))

sns.countplot(train['Type of Admission'])
plt.figure(figsize=(12, 6))

sns.countplot(train['Severity of Illness'])
plt.figure(figsize=(12, 6))

sns.countplot(train['Visitors with Patient'])
plt.figure(figsize=(12, 6))

train.groupby('Age')['patientid'].count().plot()
plt.figure(figsize=(12, 6))

sns.distplot(train.Admission_Deposit)
# Almost perfect normal distribution
plt.figure(figsize=(15, 6))

sns.countplot(train.Stay)
train.isnull().sum()
# we will remove the City_Code_Patient column and replace rows where Bed Grade is null by most frequent
sns.catplot(x="Stay", y="Bed Grade", kind="bar", data=train, aspect=2.5)
# Not a lot of differences for between the stay days for the bed grades
# Convert categorical feature for observation



type_ad = train[['Type of Admission', 'Stay']]



type_ad['Type of Admission_cat'] = type_ad['Type of Admission'].astype('category').cat.codes
type_ad[['Type of Admission', 'Type of Admission_cat']].drop_duplicates()
type_ad = type_ad.sort_values('Stay')
sns.catplot(x="Stay", y="Type of Admission_cat", kind="bar", data=type_ad, aspect=2.5)
# More emergencies for patients staying 0-10 days, otherwise quite homogene
# Convert categorical feature for observation



ill = train[['Severity of Illness', 'Stay']]



# Order of severity

ill['Severity of Illness_cat'] = ill['Severity of Illness'].map({'Minor':0, 'Moderate':1, 'Extreme':2})
ill[['Severity of Illness', 'Severity of Illness_cat']].drop_duplicates()
ill = ill.sort_values('Stay')
sns.catplot(x="Stay", y="Severity of Illness_cat", kind="bar", data=ill, aspect=2.5)
# Number of stay days increase with the severity of the illness 
# Convert categorical feature for observation



age = train[['Age', 'Stay']]



age['Age_cat'] = age['Age'].astype('category').cat.codes
age[['Age', 'Age_cat']].drop_duplicates()
age = age.sort_values('Stay')
sns.catplot(x="Stay", y="Age_cat", kind="bar", data=age, aspect=2.5)
# Number of stay days increase with the age 
train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')
test = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')
test.isnull().sum()
# Replace rows with null values

train['Bed Grade'] = train['Bed Grade'].fillna(train['Bed Grade'].mode().values[0])

test['Bed Grade'] = test['Bed Grade'].fillna(train['Bed Grade'].mode().values[0])
# Remove City_Code_Patient

train = train.drop('City_Code_Patient', axis=1)

test = test.drop('City_Code_Patient', axis=1)
# Convert categorical features



# TRAIN

train['Hospital_type_code'] = train.Hospital_type_code.astype('category').cat.codes

train['City_Code_Hospital'] = train.City_Code_Hospital.astype('category').cat.codes

train['Hospital_region_code'] = train.Hospital_region_code.astype('category').cat.codes

train['Department'] = train.Department.astype('category').cat.codes

train['Ward_Type'] = train.Ward_Type.astype('category').cat.codes

train['Ward_Facility_Code'] = train.Ward_Facility_Code.astype('category').cat.codes

train['Type of Admission'] = train['Type of Admission'].astype('category').cat.codes

train['Severity of Illness'] = train['Severity of Illness'].astype('category').cat.codes

train['Age'] = train['Age'].astype('category').cat.codes

train['Stay'] = train['Stay'].astype('category').cat.codes



#TEST

test['Hospital_type_code'] = test.Hospital_type_code.astype('category').cat.codes

test['City_Code_Hospital'] = test.City_Code_Hospital.astype('category').cat.codes

test['Hospital_region_code'] = test.Hospital_region_code.astype('category').cat.codes

test['Department'] = test.Department.astype('category').cat.codes

test['Ward_Type'] = test.Ward_Type.astype('category').cat.codes

test['Ward_Facility_Code'] = test.Ward_Facility_Code.astype('category').cat.codes

test['Type of Admission'] = test['Type of Admission'].astype('category').cat.codes

test['Severity of Illness'] = test['Severity of Illness'].astype('category').cat.codes

test['Age'] = test['Age'].astype('category').cat.codes
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
train.head()
train.columns
a = train[['Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

       'Type of Admission', 'Severity of Illness',

       'Visitors with Patient', 'Age', 'Admission_Deposit', 'Stay']]
X = a.drop("Stay", axis=1)

Y = a["Stay"]
# Split 20% test, 80% train



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression(max_iter=100)

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_val)

acc_log = accuracy_score(Y_pred_log, Y_val)

acc_log
# Logistic RegressionCV



logcv = LogisticRegressionCV(cv=10, random_state=0)

logcv.fit(X_train, Y_train)

Y_pred_log = logcv.predict(X_val)

acc_logcv = accuracy_score(Y_pred_log, Y_val)

acc_logcv
neigh = KNeighborsClassifier(n_neighbors=11) # 11 different values of Stay

neigh.fit(X_train, Y_train)

Y_pred_neigh = neigh.predict(X_val)

# get the accuracy score

acc_neigh = accuracy_score(Y_pred_neigh, Y_val)

print(acc_neigh)
clf_rf = RandomForestClassifier(n_estimators=200, max_depth=15)



clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_val)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_val)

print(acc_rf)
clf_xgb = XGBClassifier(learning_rate=0.02, n_estimators=200, max_depth = 15)



clf_xgb.fit(X_train, Y_train)



Y_pred_xgb = clf_xgb.predict(X_val)

# get the accuracy score

acc_xgb = accuracy_score(Y_pred_xgb, Y_val)

print(acc_xgb)
# LGBM Classifier



lgbm = LGBMClassifier(random_state=0)

lgbm.fit(X_train, Y_train)

Y_pred_lgbm = lgbm.predict(X_val)

acc_lgbm = accuracy_score(Y_pred_lgbm, Y_val)

acc_lgbm
test = test[['Available Extra Rooms in Hospital', 'Department', 'Ward_Type',

       'Ward_Facility_Code', 'Bed Grade', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit']]
# Best accuracy with LGBM



lgbm = LGBMClassifier(random_state=0)

lgbm.fit(X_train, Y_train)

Y_test = lgbm.predict(test)
Y_test