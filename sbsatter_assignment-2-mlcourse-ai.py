# Import libraries and set desired options

import pickle

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
# Read the training and test data sets, change paths if needed

times = ['time%s' % i for i in range(1, 11)]

train_df = pd.read_csv('../input/train_sessions.csv',

                       index_col='session_id', parse_dates=times)

test_df = pd.read_csv('../input/test_sessions.csv',

                      index_col='session_id', parse_dates=times)



# Sort the data by time

train_df = train_df.sort_values(by='time1')



# Look at the first rows of the training set

train_df.head()
# Change site1, ..., site10 columns type to integer and fill NA-values with zeros

sites = ['site%s' % i for i in range(1, 11)]

train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)

test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)



# Load websites dictionary

with open(r"../input/site_dic.pkl", "rb") as input_file:

    site_dict = pickle.load(input_file)



# Create dataframe for the dictionary

sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), 

                          columns=['site'])

print(u'Websites total:', sites_dict.shape[0])

sites_dict.head()
sites_dict.loc[25075]
train_df.loc[train_df['target'] == 1, sites].head()
pd.Series(train_df.loc[train_df['target'] == 1, sites].values.flatten())#.value_counts()
# Top websites in the training data set

top_sites = pd.Series(train_df[sites].values.flatten()

                     ).value_counts().sort_values(ascending=False).head(5)

print(top_sites)

sites_dict.loc[top_sites.drop(0).index]
# You code here

top_visited_by_alice = train_df.loc[train_df['target'] == 1, sites].melt(value_vars=['site%s' % i for i in range(1, 11)]).groupby('value').count().sort_values(by='variable', ascending=False).iloc[:5]

sites_dict.loc[top_visited_by_alice.index]

# top_visited_by_alice.
sites_dict.loc[77]
sites_dict.head(1)
# Create a separate dataframe where we will work with timestamps

time_df = pd.DataFrame(index=train_df.index)

time_df['target'] = train_df['target']



# Find sessions' starting and ending

time_df['min'] = train_df[times].min(axis=1)

time_df['max'] = train_df[times].max(axis=1)



# Calculate sessions' duration in seconds

time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')



time_df.head()
# CASE 5: False

time_df.loc[time_df['target'] == 1, 'seconds'].quantile(0.75)
# CASE 3: True (considering 37 seconds as not significant)

# CASE 4: False

time_df.groupby('target')['seconds'].agg(['min', 'max', 'std'])
# CASE 2: False

time_df.loc[:, 'target'].mean()
# You code here

# CASE 1: True

time_df.groupby(['target'])['seconds'].mean()
# Our target variable

y_train = train_df['target']



# United dataframe of the initial data 

full_df = pd.concat([train_df.drop('target', axis=1), test_df])



# Index to split the training and test data sets

idx_split = train_df.shape[0]
# Dataframe with indices of visited websites in session

full_sites = full_df[sites]

full_sites.head()
# sequence of indices

sites_flatten = full_sites.values.flatten()



# and the matrix we are looking for 

# (make sure you understand which of the `csr_matrix` constructors is used here)

# a further toy example will help you with it

full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],

                                sites_flatten,

                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]
range(0, sites_flatten.shape[0]  + 10, 10)
full_sites_sparse.shape
336358*48371/1e9
# How much memory does a sparse matrix occupy?

print('{0} elements * {1} bytes = {2} bytes'.format(full_sites_sparse.count_nonzero(), 8, 

                                                    full_sites_sparse.count_nonzero() * 8))

# Or just like this:

print('sparse_matrix_size = {0} bytes'.format(full_sites_sparse.data.nbytes))
# data, create the list of ones, length of which equal to the number of elements in the initial dataframe (9)

# By summing the number of ones in the cell, we get the frequency,

# number of visits to a particular site per session

data = [1] * 9



# To do this, you need to correctly distribute the ones in cells

# Indices - website ids, i.e. columns of a new matrix. We will sum ones up grouping them by sessions (ids)

indices = [1, 0, 0, 1, 3, 1, 2, 3, 4]



# Indices for the division into rows (sessions)

# For example, line 0 is the elements between the indices [0; 3) - the rightmost value is not included

# Line 1 is the elements between the indices [3; 6)

# Line 2 is the elements between the indices [6; 9) 

indptr = [0, 3, 6, 9]



# Aggregate these three variables into a tuple and compose a matrix

# To display this matrix on the screen transform it into the usual "dense" matrix

csr_mat = csr_matrix((data, indices, indptr))

dense = csr_mat.todense()

dense
# Your code is here

csr_mat = csr_mat[:, 1:]

non_zero = csr_mat.count_nonzero()

total = csr_mat.shape[0] * csr_mat.shape[1]

non_zero/total

# ANSWER: 50%
def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):

    # Split the data into the training and validation sets

    idx = int(round(X.shape[0] * ratio))

    # Classifier training

    lr = LogisticRegression(C=C, random_state=seed, solver='liblinear').fit(X[:idx, :], y[:idx])

    # Prediction for validation set

    y_pred = lr.predict_proba(X[idx:, :])[:, 1]

    # Calculate the quality

    score = roc_auc_score(y[idx:], y_pred)

    

    return score
%%time

# Select the training set from the united dataframe (where we have the answers)

X_train = full_sites_sparse[:idx_split, :]



# Calculate metric on the validation set

print(get_auc_lr_valid(X_train, y_train))
# Function for writing predictions to a file

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
# Train the model on the whole training data set

# Use random_state=17 for repeatability

# Parameter C=1 by default, but here we set it explicitly

lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear').fit(X_train, y_train)



# Make a prediction for test data set

X_test = full_sites_sparse[idx_split:,:]

y_test = lr.predict_proba(X_test)[:, 1]



# Write it to the file which could be submitted

write_to_submission_file(y_test, 'baseline_1.csv')
# Your code is here

full_df['time1'].apply(lambda ts: ts.year).unique()
# Dataframe for new features

full_new_feat = pd.DataFrame(index=full_df.index)



# Add start_month feature

full_new_feat['start_month'] = full_df['time1'].apply(lambda ts: 

                                                      100 * ts.year + ts.month).astype('float64')
# Your code is here

# train_df.shape, test_df.shape, full_df.shape, full_new_feat.shape

full_df_with_start_month = pd.concat([full_df, full_new_feat], axis=1)

alice_sessions = full_df_with_start_month[:idx_split][train_df['target'] == 1]

countplot = sns.countplot('start_month', data=alice_sessions)

countplot.set_xticklabels(alice_sessions.start_month.unique(), rotation=30)
# Add the new feature to the sparse matrix

tmp = full_new_feat[['start_month']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))



# Compute the metric on the validation set

print(get_auc_lr_valid(X_train, y_train))
# Add the new standardized feature to the sparse matrix

tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))



# Compute metric on the validation set

print(get_auc_lr_valid(X_train, y_train))
# Your code is here

# full_new_feat['n_unique_sites'] = full_new_feat[sites]

# full_new_feat.head()

full_df_with_start_month['n_unique_sites'] = full_df_with_start_month[sites].nunique(axis=1, dropna=True)

full_df_with_start_month
tmp2 = full_df_with_start_month[['n_unique_sites']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split, :], tmp2[:idx_split, :]]))

print(get_auc_lr_valid(X_train, y_train))
# Your code is here

full_df_with_start_month['start_hour'] = full_df_with_start_month['time1'].dt.hour
tmp3 = full_df_with_start_month[['start_hour']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :], tmp[:idx_split, :], tmp3[:idx_split, :]]))

print(get_auc_lr_valid(X_train, y_train))
full_df_with_start_month['morning'] = full_df_with_start_month['time1'].dt.hour <= 11
tmp4 = full_df_with_start_month[['morning']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :], tmp[:idx_split, :], tmp4[:idx_split, :]]))

print(get_auc_lr_valid(X_train, y_train))
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :], tmp[:idx_split, :], tmp3[:idx_split, :], tmp4[:idx_split, :]]))

print(get_auc_lr_valid(X_train, y_train))

#Both gave an improvement
# Compose the training set

tmp_scaled = StandardScaler().fit_transform(full_df_with_start_month[['start_month', 

                                                           'start_hour', 

                                                           'morning']])

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 

                             tmp_scaled[:idx_split,:]]))



# Capture the quality with default parameters

score_C_1 = get_auc_lr_valid(X_train, y_train)

print(score_C_1)
from tqdm import tqdm



# List of possible C-values

Cs = np.logspace(-3, 1, 10)

scores = []

for C in tqdm(Cs):

    scores.append(get_auc_lr_valid(X_train, y_train, C=C))
scores
plt.plot(Cs, scores, 'ro-')

plt.xscale('log')

plt.xlabel('C')

plt.ylabel('AUC-ROC')

plt.title('Regularization Parameter Tuning')

# horizontal line -- model quality with default C value

plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed') 

plt.show()
# Your code is here

Cs[np.argmax(scores)]
# Prepare the training and test data

tmp_scaled = StandardScaler().fit_transform(full_df_with_start_month[['start_month', 'start_hour', 

                                                           'morning']])

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 

                             tmp_scaled[:idx_split,:]]))

X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:], 

                            tmp_scaled[idx_split:,:]]))



# Train the model on the whole training data set using optimal regularization parameter

lr = LogisticRegression(C=0.1668100537200059, random_state=17, solver='liblinear').fit(X_train, y_train)



# Make a prediction for the test set

y_test = lr.predict_proba(X_test)[:, 1]



# Write it to the submission file

write_to_submission_file(y_test, 'baseline_2.csv')
!ls

!cat baseline_2.csv
full_new_feat_copy = full_new_feat.copy(deep=True)

full_new_feat = full_df_with_start_month.copy(deep=True)

full_new_feat.head()
# full_new_feat['avg_time_on_each_site'] = 

full_new_feat[times].apply(lambda t: t.time1.year)