import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from time import time



from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from sklearn.model_selection import train_test_split



from sklearn import metrics
%%time

# dataset_path = '../datasets/fraud_datasets/' # Local Notebook

dataset_path = '../input/ieee-fraud-detection/' # Kaggle Notebook



# sample_submission_df = pd.read_csv(f'{dataset_path}sample_submission.csv')



train_transaction_df = pd.read_csv(f'{dataset_path}train_transaction.csv')

test_transaction_df = pd.read_csv(f'{dataset_path}test_transaction.csv')



train_id_df = pd.read_csv(f'{dataset_path}train_identity.csv')

test_id_df = pd.read_csv(f'{dataset_path}test_identity.csv')
%%time

train = train_transaction_df.merge(train_id_df, on='TransactionID', how='left', left_index=True, right_index=True)

test = test_transaction_df.merge(test_id_df, on='TransactionID', how='left', left_index=True, right_index=True)



# Renaming columns for better description

names = {

    'addr1': 'billing zipcode',

    'addr2': 'country codes',

    'P_emaildomain': 'Purchaser_email.dom',

    'R_emaildomain': 'Retailer_email.dom'

}



train.rename(columns=names, inplace=True)

test.rename(columns=names, inplace=True)
del train_transaction_df, train_id_df, test_transaction_df, test_id_df
# # This is for local environment

# eda_output_path = 'eda_output/'

# feat_over_file = 'feature_overview.txt'



# if os.path.exists(f'{eda_output_path}{feat_over_file}'):

#     os.remove(f'{eda_output_path}{feat_over_file}')



# with open(f'{eda_output_path}{feat_over_file}', 'a') as overview_file:

#     for col, values in train.iteritems():

#         overview_file.write(f'{col}: {values.nunique()} ({values.dtypes})\n')

#         overview_file.write(str(values.unique()[:100]))

#         overview_file.write(

#             '\n\n###########################################################\n\n')
train_contains_na = train.isna().any().sum()

test_contains_na = test.isna().any().sum()
print(f'{train_contains_na} out of {len(train.columns)} columns contain missing values in the train data')

print(f'{test_contains_na} out of {len(test.columns)} columns contain missing values in the test data')
# Calculating percentage of missing values in each column

train_missing_values = train.isna().mean().round(2)

test_missing_values = test.isna().mean().round(2)



# Keeping only columns that contain more than 5% missing values

train_missing_values_5 = train_missing_values[train_missing_values.values > 0.05]

test_missing_values_5 = test_missing_values[test_missing_values.values > 0.05]
print(f'{len(train_missing_values_5)} out of {len(train.columns)} columns in the train data contain more than 5% missing values')

print(f'{len(test_missing_values_5)} out of {len(test.columns)} columns in the test data contain more than 5% missing values')
# Keeping only columns that contain more than 50% missing values

train_missing_values_50 = train_missing_values[train_missing_values.values > 0.5]

test_missing_values_50 = test_missing_values[test_missing_values.values > 0.5]
print(f'{len(train_missing_values_50)} out of {len(train.columns)} columns in the train data contain more than 50% missing values')

print(f'{len(test_missing_values_50)} out of {len(test.columns)} columns in the test data contain more than 50% missing values')
isFraud = 'isFraud'
# Plot Function

def PlotFunction(df, feature, title, xLable, yLabel, vertical=False, percentLabels=False, size=[10, 7]):

    plt.style.use('ggplot')



    f = plt.figure(figsize=size)

    ax = f.add_subplot(1, 1, 1)

    ax.title.set_text(title)

    ax.set_ylabel(yLabel)

    ax.set_xlabel(xLable)



    plot = ax.bar([str(i) for i in df[feature].value_counts(dropna=False, normalize=True).index],

                  df[feature].value_counts(dropna=False, normalize=True), 0.40,

                  color=['cornflowerblue', 'darkorange', 'green', 'brown', 'black'])

    if vertical:

        plt.xticks(rotation=90)



    # Add counts above the two bar graphs

    if percentLabels:

        percentages = (df[feature].value_counts(

            dropna=False, normalize=True)*100).round(3)

        i = 0

        for rect in plot:

            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2.0, height,

                     f'{percentages[i]}%', ha='center', va='bottom')

            i += 1
print(f'These are the two types of values for fraud: {train[isFraud].unique()}')
print(f'isFraud contains {train[isFraud].isna().any() * 1} missing values')
### 'Fraud' Feature

PlotFunction(train, isFraud, 'isFaud percentages', 'Not fraud | Fraud', 'Number of occurances', percentLabels=True)    
# 'TransactionAmt' feature

plt.figure(figsize=[12, 5])

plt.title('Transaction Amounts')

plt.xlabel('Amount')

plt.ylabel('Number of transactions')

_ = plt.hist(train['TransactionAmt'], bins=100)
# 'TransactionAmt' feature

plt.figure(figsize=[12, 5])

plt.title('Transaction Amounts')

plt.xlabel('Amount')

plt.ylabel('Number of transactions')

_ = plt.hist(train['TransactionAmt'].apply(np.log), bins=100)
# 'TransactionAmt' feature

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))



# Not Fraud

ax1.title.set_text('Non-Fraudulent Transaction Amounts')

ax1.set_xlabel('LgAmount')

ax1.set_ylabel('Number of transactions')

_ = ax1.hist(train['TransactionAmt']

             [train['isFraud'] == 0].apply(np.log), bins=100)



# Fraud

ax2.title.set_text('Fraudulent Transaction Amounts')

ax2.set_xlabel('LgAmount')

ax2.set_ylabel('Number of transactions')

_ = ax2.hist(train['TransactionAmt']

             [train['isFraud'] == 1].apply(np.log), bins=100)
### 'ProductCD' feature

PlotFunction(train, 'ProductCD', 'Product codes for each transaction', 'Product Codes', 'Count', percentLabels=True)
f = plt.figure(figsize=[15, 5])

ax = f.add_subplot(1, 1, 1)

ax.title.set_text('Percentage of products from fraudulent transactions')

ax.set_ylabel('Percent')

ax.set_xlabel('Products')



plot = ax.bar([str(i) for i in train['ProductCD'][train['isFraud'] == 1].value_counts(dropna=False, normalize=True).index],

              train['ProductCD'][train['isFraud'] == 1].value_counts(

                  dropna=False, normalize=True), 0.40,

              color=['cornflowerblue', 'darkorange', 'green', 'brown', 'black'])



# Add counts above the two bar graphs

percentages = (train['ProductCD'][train['isFraud'] == 1].value_counts(

    dropna=False, normalize=True)*100).round(3)

i = 0

for rect in plot:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height,

             f'{percentages[i]}%', ha='center', va='bottom')

    i += 1
# 'card1' feature

print('Number of unique values:', len(train['card1'].unique()))

train['card1'].value_counts(dropna=False, normalize=True).head()
### 'card2' feature

print('Number of unique values:', len(train['card2'].unique()))

train['card2'].value_counts(dropna=False, normalize=True).head()
### 'card3' feature

print('Number of unique values:', len(train['card3'].unique()))

train['card3'].value_counts(dropna=False, normalize=True).head()
### 'card5' feature

print('Number of unique values:', len(train['card5'].unique()))

train['card5'].value_counts(dropna=False, normalize=True).head()
### 'card4' feature

PlotFunction(train, 'card4', 'Number of card types', 'Card types', 'Number of card occurrences', percentLabels=True)



print('Number of unique values:', len(train['card4'].unique()))

train['card4'].value_counts(dropna=False, normalize=True).head(10)
### 'card6' feature

PlotFunction(train, 'card6', 'Number of account types', 'Account types', 'Number of account type occurrences', percentLabels=True)



print('Number of unique values:', len(train['card6'].unique()))

train['card6'].value_counts(dropna=False, normalize=True).head(10)
### 'billing zipcode' feature

print('Number of unique values:', len(train['billing zipcode'].unique()))

train['billing zipcode'].value_counts(dropna=False, normalize=True).head(10)
### 'country codes' feature

print('Number of unique values:', len(train['country codes'].unique()))

train['country codes'].value_counts(dropna=False, normalize=True).head()
### 'Purchaser Email' feature

PlotFunction(train, 'Purchaser_email.dom', 'Number of Purchaser email types', 'Email types',

             'Number of email type occurrences', vertical=True, percentLabels=False)



print('Number of unique values:', len(train['Purchaser_email.dom'].unique()))

train['Purchaser_email.dom'].value_counts(dropna=False, normalize=True).head()
# 'Purchaser Email' feature

PlotFunction(train, 'Retailer_email.dom', 'Number of Purchaser email types', 'Email types',

             'Number of email type occurrences', vertical=True, percentLabels=False)



print('Number of unique values:', len(train['Purchaser_email.dom'].unique()))

train['Retailer_email.dom'].value_counts(dropna=False, normalize=True).head()
feature_names = ['M1', 'M2', 'M4', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']



fig, axes = plt.subplots(3, 3, figsize=(17, 10))



j = 0

for row in axes:

    for ax in row:

        ax.set_title(feature_names[j])

        ax.set_ylabel('Count')

        ax.set_xlabel(feature_names[j])



        plot = ax.bar([str(i) for i in train[feature_names[j]].value_counts(dropna=False, normalize=True).index],

                      train[feature_names[j]].value_counts(

                          dropna=False, normalize=True), 0.40,

                      color=['cornflowerblue', 'darkorange', 'green', 'brown', 'black'])

        j += 1



plt.tight_layout()
y = train['isFraud'].copy()

X = train.drop(['isFraud', 'TransactionID'], axis=1) # Dropping TransactionID as it is a useless feature

del train
%%time

X = X.fillna(-999)

test = test.fillna(-999)
%%time

for feature in X.columns:

    if X[feature].dtype == 'object' or test[feature].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(X[feature].values) + list(test[feature].values))

        X[feature] = le.transform(list(X[feature].values))

        test[feature] = le.transform(list(test[feature].values))
# from sklearn.model_selection import RandomizedSearchCV



# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)



# # Parameter grid

# parameter_grid = {

#     'n_estimators': list(np.linspace(10, 300, dtype=int)),

#     'max_depth': None + list(np.linspace(1, 30, dtype=int)),

#     'max_features': ['auto', 'sqrt']

# }



# model = RandomForestRegressor(random_state=42)

# rscv = RandomizedSearchCV(model, parameter_grid, cv=5, n_jobs = -1, random_state=42)

# rscv.fit(X_train, y_train)



# print(rscv.best_params_)
# y_pred = rscv.best_estimator_.predict(X_test)



# model_measures(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



model = RandomForestRegressor(n_estimators=258, max_features='sqrt', 

                               max_depth=18, n_jobs=-1, verbose=1)



model.fit(X, y)
print("Roc Auc Score:",metrics.roc_auc_score(y_test,model.predict(X_test)))
# y_pred_rf = rscv.best_estimator_.predict(test[features]) # RSCV

y_pred_rf = model.predict(test.drop('TransactionID', axis=1)) # Single random forest



# result_path = 'results/' # Local Notebook



finished_df_rf = pd.DataFrame(test['TransactionID'])

finished_df_rf['isFraud'] = y_pred_rf



finished_df_rf.to_csv('predictions_rf.csv', index=False)