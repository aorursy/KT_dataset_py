#Import libary



import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import os

%matplotlib inline

#display file name



print(os.listdir("../input/"))
# Training Data



train = pd.read_csv('../input/application_train.csv')

test = pd.read_csv('../input/application_test.csv')

train.head()
#Missing data



sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#visualizing some more of the data



sns.set_style('whitegrid')

sns.countplot(x='NAME_CONTRACT_TYPE',data=train,palette='RdBu_r')
#visualizing some more of the data



sns.set_style('whitegrid')

sns.countplot(x='NAME_CONTRACT_TYPE',hue='CODE_GENDER',data=train,palette='RdBu_r')
#check the anomaly data



train_labels = train['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

train, test = train.align(test, join = 'inner', axis = 1)



# Add the target back in

train['TARGET'] = train_labels



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)
#describe



train['DAYS_EMPLOYED'].describe()
#Count anamalies data



anom = train[train['DAYS_EMPLOYED'] == 365243]

non_anom = train[train['DAYS_EMPLOYED'] != 365243]

print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))

print('There are %d anomalous days of employment' % len(anom))
#count anomali data



test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243

test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)



print('There are %d anomalies in the test data out of %d entries' % (test["DAYS_EMPLOYED_ANOM"].sum(), len(test)))
# Find correlations with the target and sort

correlations = train.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
# Plot the distribution of ages in years



plt.style.use('fivethirtyeight')

plt.hist(train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)

plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
# Find the correlation of the positive days since birth and target



train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])

train['DAYS_BIRTH'].corr(train['TARGET'])
# Age information into a separate dataframe



age_data = train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365



# Bin the age data



age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_data.head(10)
# Group by the bin and calculate averages



age_groups  = age_data.groupby('YEARS_BINNED').mean()

age_groups
# Create a label encoder object



le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in train:

    if train[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(train[col].unique())) <= 2:

            # Train on the training data

            le.fit(train[col])

            # Transform both training and testing data

            train[col] = le.transform(train[col])

            test[col] = le.transform(test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

train = pd.get_dummies(train)

test = pd.get_dummies(test)



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)
train_labels = train['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

train, test = train.align(test, join = 'inner', axis = 1)



# Add the target back in

train['TARGET'] = train_labels



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)
from sklearn.preprocessing import MinMaxScaler, Imputer



# Drop the target from the training data

if 'TARGET' in train:

    train1 = train.drop(columns = ['TARGET'])

else:

    train1 = train.copy()

    

# Feature names

features = list(train1.columns)



# Copy of the testing data

test1 = test.copy()



# Median imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0, 1))



# Fit on the training data

imputer.fit(train1)



# Transform both training and testing data

train1 = imputer.transform(train1)

test1 = imputer.transform(test)



# Repeat with the scaler

scaler.fit(train1)

train1 = scaler.transform(train1)

test1 = scaler.transform(test1)



print('Training data shape: ', train1.shape)

print('Testing data shape: ', test1.shape)
from sklearn.linear_model import LogisticRegression



# Make the model with the specified regularization parameter

log_reg = LogisticRegression(C = 0.0001)



# Train on the training data

log_reg.fit(train1, train_labels)
log_reg_pred = log_reg.predict_proba(test1)[:, 1]
# Submission dataframe



submit = test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



submit.head()
#Create File



submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier



# Make the random forest classifier

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 65, verbose = 1, n_jobs = -2)
# Train on the training data

random_forest.fit(train1, train_labels)



# Extract feature importances

feature_importance_values = random_forest.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})



# Make predictions on the test data

predictions = random_forest.predict_proba(test1)[:, 1]