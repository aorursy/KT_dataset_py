# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from catboost import Pool, CatBoostClassifier

from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
from statsmodels.stats.proportion import proportion_confint
from imblearn.under_sampling import RandomUnderSampler
# Read all the data
all_data = pd.read_csv(
    '../input/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv',
    parse_dates=['issue_d'], infer_datetime_format=True)
# Only process data in a 1 year period from beginning of 2018 to beginning of 2019
all_data = all_data[(all_data.issue_d >= '2017-01-01 00:00:00') & (all_data.issue_d < '2019-01-01 00:00:00')]
all_data.head()
# Read in the LCDataDictionary.xls file which contains all the features available to investors and their descriptions
dataset_fields = pd.read_excel('../input/lending-club-loan-data/LCDataDictionary.xlsx',sheet_name=1)

# Read in the features file
dataset_features = dataset_fields['BrowseNotesFile'].dropna().values

# Reformat the feature names to match the ones in all_data
dataset_features = [re.sub('(?<![0-9_])(?=[A-Z0-9])', '_', x).lower().strip() for x in dataset_features]

# There are some features in both datasets that mean the same thing are different words
# are used for them. Change those feautre names accordingly
old_name = ['is_inc_v', 'mths_since_most_recent_inq', 'mths_since_oldest_il_open',
         'mths_since_recent_loan_delinq', 'verified_status_joint']
new_name = ['verification_status', 'mths_since_recent_inq', 'mo_sin_old_il_acct',
           'mths_since_recent_bc_dlq', 'verification_status_joint']

dataset_features = np.setdiff1d(dataset_features, old_name)
dataset_features = np.append(dataset_features, new_name)

# Remove all the feauture columns from all_data that are not part of dataset_features
all_data_feaures = all_data.columns.values
availble_features = np.intersect1d(dataset_features, all_data_feaures)
availble_features = np.append(availble_features, ['loan_status']);
data_with_available_features = all_data[availble_features].copy()

# Get info regarding the dataframe to confirm appropriate features are removed
data_with_available_features.info()
# Change appropriate columns to float type and remove rows where string cannot be converted to float
data_with_available_features['emp_length'] = data_with_available_features['emp_length'].replace({'< 1 year': '0 years', '10+ years': '11 years'})
data_with_available_features['emp_length'] = data_with_available_features['emp_length'].str.extract('(\d+)').astype('float')

# Change appropriate columns to datetime type
data_with_available_features['earliest_cr_line'] = pd.to_datetime(data_with_available_features['earliest_cr_line'], infer_datetime_format=True)
data_with_available_features['sec_app_earliest_cr_line'] = pd.to_datetime(data_with_available_features['sec_app_earliest_cr_line'], infer_datetime_format=True)

# Remove columns with no values
data_with_available_features = data_with_available_features.drop(['desc', 'member_id', 'id'], axis=1, errors='ignore')

# Get info regarding the dataframe to confirm appropriate features are removed
data_with_available_features.info()
# Define columns that need to be filled with an empty value, max, or min value
columns_to_fill_empty = ['emp_title', 'verification_status_joint']
columns_to_fill_max = ['bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',
            'mths_since_last_major_derog', 'mths_since_last_record',
            'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
            'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
            'pct_tl_nvr_dlq','sec_app_mths_since_last_major_derog']
columns_to_fill_min = np.setdiff1d(data_with_available_features.columns.values, np.append(columns_to_fill_empty, columns_to_fill_max))

data_with_available_features[columns_to_fill_empty] = data_with_available_features[columns_to_fill_empty].fillna('')
data_with_available_features[columns_to_fill_max] = data_with_available_features[columns_to_fill_max].fillna(data_with_available_features[columns_to_fill_max].max())
data_with_available_features[columns_to_fill_min] = data_with_available_features[columns_to_fill_min].fillna(data_with_available_features[columns_to_fill_min].min())
# Remove all features that only hold one value or all unqiue values
data_with_available_features = data_with_available_features.drop(['num_tl_120dpd_2m', 'url', 'emp_title'], axis=1, errors='ignore')

# Calculate the correlation between all pairs of numeric features
# Pearson's R correlation coefficient was used
features_with_num_type = data_with_available_features.select_dtypes('number').columns.values
combination_of_features_with_num_type = np.array(list(combinations(features_with_num_type, 2)))
correlation_of_features_with_num_type = np.array([])
for comb in combination_of_features_with_num_type:
    corr = pearsonr(data_with_available_features[comb[0]], data_with_available_features[comb[1]])[0]
    correlation_of_features_with_num_type = np.append(correlation_of_features_with_num_type, corr)

# Drop the first one in the pair with coefficient >= 0.9
high_corr_num = combination_of_features_with_num_type[np.abs(correlation_of_features_with_num_type) >= 0.9]
data_with_available_features = data_with_available_features.drop(np.unique(high_corr_num[:, 0]), axis=1, errors='ignore')

# Calculate the correlation between all pairs of categorical features
# Cramer's V correlation coefficient was used
cat_feat = data_with_available_features.select_dtypes('object').columns.values
comb_cat_feat = np.array(list(combinations(cat_feat, 2)))
corr_cat_feat = np.array([])
for comb in comb_cat_feat:
    table = pd.pivot_table(data_with_available_features, values='loan_amnt', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
    corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )
    corr_cat_feat = np.append(corr_cat_feat, corr)

# Drop the second one in the pair with coefficient >= 0.9
high_corr_cat = comb_cat_feat[corr_cat_feat >= 0.9]
data_with_available_features = data_with_available_features.drop(np.unique(high_corr_cat[:, 1]), axis=1, errors='ignore')
# Create the expected_output dataframe
expected_output = data_with_available_features['loan_status'].copy()
expected_output = expected_output.isin(['Current', 'Fully Paid', 'In Grace Period']).astype('int')
undersampler = RandomUnderSampler(sampling_strategy='majority')

# remove the int_rate field since its highly correlated with the grading field and loan_status since we saved it in expected_output
dataset = data_with_available_features.drop(['int_rate', 'loan_status'], axis=1, errors='ignore')
expected_results = expected_output[dataset.index]

# Split the dataset and the expected_result set into training and testing
dataset_train, dataset_test, expected_results_train, expected_results_test = train_test_split(dataset, expected_results, stratify=expected_results, random_state=0)

# Further split the train datasets into training and validation dataset
dataset_train, dataset_validate, expected_results_train, expected_results_validate = train_test_split(dataset_train, expected_results_train, stratify=expected_results_train, random_state=0)
# Transform the datasets into CatBoost objects
cat_feat_ind = (dataset_train.dtypes == 'object').nonzero()[0]
pool_train = Pool(dataset_train, expected_results_train, cat_features=cat_feat_ind)
pool_val = Pool(dataset_validate, expected_results_validate, cat_features=cat_feat_ind)
pool_test = Pool(dataset_test, expected_results_test, cat_features=cat_feat_ind)

# Build the model and fit the data
n = expected_results_train.value_counts()
model = CatBoostClassifier(learning_rate=0.03,
                           iterations=1000,
                           early_stopping_rounds=100,
                           class_weights=[1, n[0] / n[1]],
                           verbose=False,
                           random_state=0)
model.fit(pool_train, eval_set=pool_val, plot=True);
# Predict the test dataset with the model
actual_results = model.predict(pool_test)

# Get the acuracy, precision (false positives), and recall (false negatives)
acc_test = accuracy_score(expected_results_test, actual_results)
prec_test = precision_score(expected_results_test, actual_results)
rec_test = recall_score(expected_results_test, actual_results)
print(f'''Accuracy (test): {acc_test:.3f}
Precision (test): {prec_test:.3f}
Recall (test): {rec_test:.3f}''')

# Generate the confusion matrix
cm = confusion_matrix(expected_results_test, actual_results)
ax = sns.heatmap(cm, cmap='viridis_r', annot=True, fmt='d', square=True)
ax.set_xlabel('Predicted')
ax.set_ylabel('True');
# Plot the precision (false positives) and the recall (false negatvies). Want to maximize this graph for the precision.
expected_output_proba_validate = model.predict_proba(pool_val)[:, 1]
p_val, r_val, t_val = precision_recall_curve(expected_results_validate, expected_output_proba_validate)
plt.plot(r_val, p_val)
plt.ylabel('Precision');
plt.xlabel('Recall')
# precision of 1 results in recall of 0 which is too low
max_precision = p_val[p_val != 1].max()

# Find the threshold at which precision is maximized
t_all = np.insert(t_val, 0, 0)
t_adj_val = t_all[p_val == max_precision]
expected_results_adj_validate = (expected_output_proba_validate > t_adj_val).astype(int)
p_adj_val = precision_score(expected_results_validate, expected_results_adj_validate)
print(f'Adjusted precision (validation): {p_adj_val:.3f}')
expected_results_proba_test = model.predict_proba(pool_test)[:, 1]
expected_results_adj_test = (expected_results_proba_test > t_adj_val).astype(int)
a_adj_test = accuracy_score(expected_results_test, expected_results_adj_test)
p_adj_test = precision_score(expected_results_test, expected_results_adj_test)
r_adj_test = recall_score(expected_results_test, expected_results_test)
print(f'''Adjusted Accuracy (test): {a_adj_test:.3f}
Adjusted Precision (test): {p_adj_test:.3f}
Adjusted Recall (test): {r_adj_test:.3f}''')

cm_test = confusion_matrix(expected_results_test, expected_results_adj_test)
ax = sns.heatmap(cm_test, cmap='viridis_r', annot=True, fmt='d', square=True)
ax.set_xlabel('Predicted')
ax.set_ylabel('True');