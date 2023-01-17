import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

from statistics import mode
# Read train data

train = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/train_fNxu4vz.csv')



# Have a first look at train data

print('Train shape:', train.shape)
# Have a look at first 5 data observations

train.head()
# Have a look at last 5 data observations

train.tail()
train['Length_Employed'] = train['Length_Employed'].str.replace('<','')

train['Length_Employed'] = train['Length_Employed'].str.replace('+','')

train['Length_Employed'] = train['Length_Employed'].str.replace('year','')

train['Length_Employed'] = train['Length_Employed'].str.replace('s','')
train.isnull().mean().sort_values(ascending = False)
# Read test data

test = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/test_fjtUOL8.csv')



# Have a first look at test data

print('Test shape:', test.shape)
# Have a look at train and test columns

print('Train columns:', train.columns.tolist())

print('Test columns:', test.columns.tolist())
# Load our plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns
# Countplot for 'Interest_Rate' variable

sns.countplot(train['Interest_Rate'])
# Let's calculate the mean of our target

round(np.mean(train['Interest_Rate']), 2)
# Convert to numeric

train["Loan_Amount_Requested"] = train["Loan_Amount_Requested"].str.replace(",", "")

train["Loan_Amount_Requested"] = pd.to_numeric(train["Loan_Amount_Requested"])

test["Loan_Amount_Requested"] = test["Loan_Amount_Requested"].str.replace(",", "")

test["Loan_Amount_Requested"] = pd.to_numeric(test["Loan_Amount_Requested"])
test['Length_Employed'] = test['Length_Employed'].str.replace('<','')

test['Length_Employed'] = test['Length_Employed'].str.replace('+','')

test['Length_Employed'] = test['Length_Employed'].str.replace('year','')

test['Length_Employed'] = test['Length_Employed'].str.replace('s','')
# Predict!

#test['Interest_Rate'] = logisticRegression.predict(pd.get_dummies(test['Gender']))
# Write test predictions for final submission

#test[['Loan_ID', 'Interest_Rate']].to_csv('First_Prediction.csv', index = False)
# Excuse me, can we have a plot please?!

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'plasma')
train.describe(include = 'all')
fig = plt.figure(figsize=(12,8))

sns.countplot(train['Length_Employed'])
fig = plt.figure(figsize=(12,8))

sns.countplot(train['Home_Owner'])
fig = plt.figure(figsize=(12,8))

sns.countplot(train['Income_Verified'])
fig = plt.figure(figsize=(15,8))

sns.countplot(train['Purpose_Of_Loan'])
# Plot 'Debt_To_Income' histogram

train['Debt_To_Income'].hist(bins = 50, color = 'blue')
# Plot 'Annual_Income' histogram

train['Annual_Income'].hist(bins = 50, color = 'blue')
# Plot 'Months_Since_Deliquency' histogram

train['Months_Since_Deliquency'].hist(bins = 50, color = 'blue')
# Plot 'Inquiries_Last_6Mo' histogram

train['Inquiries_Last_6Mo'].hist(bins = 50, color = 'blue')
plt.figure(figsize = (15,10))

ax = sns.heatmap(train.corr(), annot = True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
feature_num = train.select_dtypes(include=[np.number])

feature_num.columns
feature_cat = train.select_dtypes(include=[np.object])

feature_cat.columns
df = [train, test]
for dataset in df:

    dataset.drop(['Loan_ID'], axis=1, inplace=True)
train["Length_Employed"].fillna('NaN', inplace=True)

test["Length_Employed"].fillna('NaN', inplace=True)



train["Home_Owner"].fillna('NaN', inplace=True)

test["Home_Owner"].fillna('NaN', inplace=True)



train["Income_Verified"].fillna('NaN', inplace=True)

test["Income_Verified"].fillna('NaN', inplace=True)



train["Purpose_Of_Loan"].fillna('NaN', inplace=True)

test["Purpose_Of_Loan"].fillna('NaN', inplace=True)



train["Gender"].fillna('NaN', inplace=True)

test["Gender"].fillna('NaN', inplace=True)



train["Annual_Income"].fillna(train["Annual_Income"].mean(), inplace=True)

test["Annual_Income"].fillna(test["Annual_Income"].mean(), inplace=True)



train["Months_Since_Deliquency"].fillna(0, inplace=True)

test["Months_Since_Deliquency"].fillna(0, inplace=True)
test.isnull().sum().sort_values(ascending = False)
test.shape
test.isnull().sum().sort_values(ascending = False)
train.shape
full = pd.concat([train, test])

full.shape
full = pd.get_dummies(full, drop_first=True)

full.shape
train = full.iloc[: 164309, :]

test = full.iloc[164309: , :]

test
from sklearn.model_selection import train_test_split



# Here is out local validation scheme!

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Interest_Rate'], axis = 1), 

                                                    train['Interest_Rate'], test_size = 0.2, 

                                                    random_state = 2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# We'll use a logistic regression model again, but we'll go to something more fancy soon! 

from sklearn.linear_model import LogisticRegression

logisticRegression = LogisticRegression(max_iter = 10000)

logisticRegression.fit(X_train, y_train)

# Predict!

predictions = logisticRegression.predict(X_test)

# Print our preditions

print(predictions)
# Check mean

round(np.mean(predictions), 2)
from sklearn.model_selection import KFold



# Set our robust cross-validation scheme!

kf = KFold(n_splits = 5, random_state = 2)
from sklearn.model_selection import cross_val_score



# Print our CV accuracy estimate:

#print(cross_val_score(logisticRegression, X_test, y_test, cv = kf).mean())

cross_val_score(logisticRegression, train.drop(['Interest_Rate'], axis = 1),train['Interest_Rate'], cv = kf).mean()
from sklearn.ensemble import RandomForestClassifier



#Initialize randomForest

randomForest = RandomForestClassifier(random_state = 2)
# Define our optimal randomForest algo

randomForestFinalModel = RandomForestClassifier(random_state = 2, criterion = 'gini', 

                                                max_depth = 7, max_features = 'auto', n_estimators = 300)
# Fit the model to the training set

randomForestFinalModel.fit(X_train, y_train)
# Predict!

predictions = randomForestFinalModel.predict(X_test)
# Predict!

test['Interest_Rate'] = randomForestFinalModel.predict(test.drop(['Interest_Rate'], axis = 1))
submission_sample = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/sample_submission_HSqiq1Q.csv')
dataset = pd.DataFrame({

    'Loan_ID': submission_sample['Loan_ID'],

    'Interest_Rate': test['Interest_Rate']

})
dataset.to_csv('output.csv', index = False)