# Load necessary modules
import numpy as np 
import pandas as pd
import os
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
print(os.listdir("../input"))
# Load the Data
# Read data frame
df = pd.read_csv('../input/Financial Distress.csv', index_col=False,
                 dtype={
                     'Company': np.uint16,
                     'Time': np.uint8,
                     'Financial Distress': np.double
                 })
# Look at the Data
print("Number of unique companies:", df.Company.unique().shape[0])  # 422 companies
print("Number of time periods per company:")
print(pd.crosstab(df.Company, df.Time.sum()))  # Some companies have < 5 time periods
# Take a look at the data based on Groups per Company
grouped_company = df.groupby('Company')

# Take first 5 groups
group_gen = ((name, group) for name, group in grouped_company)
for name, group in itertools.islice(group_gen, 5):
    # For each group, print and show the data
    print('-------------------------------------')
    print("Data of Company", name)
    print(group.head(15))
# Dummy Variables
dummy_cols = pd.get_dummies(df[['x80']], prefix='dummy', columns=['x80'], drop_first=True)

print(dummy_cols.head())

# Combine dummy_cols back with original data set
x_cols = [col for col in df.columns if all([col.startswith('x'), col != 'x80'])]
df_transformed = pd.concat([df[['Company', 'Time', 'Financial Distress'] + x_cols].reset_index(drop=True),
                            pd.DataFrame(data=dummy_cols)], axis=1)

df_transformed.head()
# Helper function to create lagged features
def lagged_features(df_long, lag_features, window=2, lag_prefix='lag', lag_prefix_sep='_'):
    """
    Function calculates lagged features (only for columns mentioned in lag_features)
    based on time_feature column. The highest value of time_feature is retained as a row
    and the lower values of time_feature are added as lagged_features
    :param df_long: Data frame (longitudinal) to create lagged features on
    :param lag_features: A list of columns to be lagged
    :param window: How many lags to perform (0 means no lagged feature will be produced)
    :param lag_prefix: Prefix to name lagged columns.
    :param lag_prefix_sep: Separator to use while naming lagged columns
    :return: Data Frame with lagged features appended as columns
    """
    if not isinstance(lag_features, list):
        # So that while sub-setting DataFrame, we don't get a Series
        lag_features = [lag_features]

    if window <= 0:
        return df_long

    df_working = df_long[lag_features].copy()
    df_result = df_long.copy()
    for i in range(1, window+1):
        df_temp = df_working.shift(i)
        df_temp.columns = [lag_prefix + lag_prefix_sep + str(i) + lag_prefix_sep + x
                           for x in df_temp.columns]
        df_result = pd.concat([df_result.reset_index(drop=True),
                               df_temp.reset_index(drop=True)],
                               axis=1)

    return df_result


# Now split Data Set into groups (based on Company) and create lagged features for each group
grouped_company = df_transformed.groupby('Company')
cols_to_lag = [col for col in df_transformed.columns if col.startswith('x')]
df_cross = pd.DataFrame()

for name, group in grouped_company:
    # For each group, calculate lagged features and rbind to df_cross
    print('----------------------------------------------------')
    print('Working on group:', name, 'with shape', group.shape)
    df_cross = pd.concat([df_cross.reset_index(drop=True),
                          lagged_features(group, cols_to_lag).reset_index(drop=True)],
                         axis=0)
    print('Shape of df_cross', df_cross.shape)
    
# Remove rows with NAs
df_cross = df_cross.dropna()
df_cross.head()
# Create Time-Series sampling function to draw train-test splits
def ts_sample(df_input, train_rows, test_rows):
    """
    Function to draw specified train_rows and test_rows in time-series rolling sampling format
    :param df_input: Input DataFrame
    :param train_rows: Number of rows to use as training set
    :param test_rows: Number of rows to use as test set
    :return: List of tuples. Each tuple contains 2 lists of indexes corresponding to train and test index
    """
    if df_input.shape[0] <= train_rows:
        return [(df_input.index, pd.Index([]))]

    i = 0
    train_lower, train_upper = 0, train_rows + test_rows*i
    test_lower, test_upper = train_upper, min(train_upper + test_rows, df_input.shape[0])

    result_list = []
    while train_upper < df_input.shape[0]:
        # Get indexes into result_list
        # result_list += [([df_input.index[train_lower], df_input.index[train_upper]],
        #                  [df_input.index[test_lower], df_input.index[test_upper]])]
        result_list += [(df_input.index[train_lower:train_upper],
                         df_input.index[test_lower:test_upper])]

        # Update counter and calculate new indexes
        i += 1
        train_upper = train_rows + test_rows*i
        test_lower, test_upper = train_upper, min(train_upper + test_rows, df_input.shape[0])

    return result_list

# For each group, apply function ts_sample
# Depending on size of group, the output size of ts_sample (which is a list of (train_index, test_index))
# tuples will vary. However, we want the size of each of these lists to be equal.
# To do that, we will augment the smaller lists by appending the last seen train_index and test_index
# For example:
# group 1 => [(Int64Index([1, 2, 3], dtype='int64'), (Int64Index[4, 5], dtype='int64)),
#             (Int64Index([1, 2, 3, 4, 5], dtype='int64'), (Int64Index([6], dtype='int64'))]
# group 2 => [(Int64Index([10, 11, 12], dtype='int64'), (Int64Index[13, 14], dtype='int64')),
#             (Int64Index([10, 11, 12, 13, 14), Int64Index([15, 16])),
#             (Int64Index([10, 11, 12, 13, 14, 15, 16]), Int64Index([17, 18]))]
# Above, group 2 has 3 folds whereas group 1 has 2. We will augment group 2 to also have 3 folds:
# group 1 => [(Int64Index([1, 2, 3], dtype='int64'), (Int64Index[4, 5], dtype='int64)),
#             (Int64Index([1, 2, 3, 4, 5], dtype='int64'), (Int64Index([6], dtype='int64')),
#             (Int64Index([1, 2, 3, 4, 5, 6]), Int64Index([]))]
grouped_company_cross = df_cross.groupby('Company')
acc = []
max_size = 0
for name, group in grouped_company_cross:
    # For each group, calculate ts_sample and also store largest ts_sample output size
    group_res = ts_sample(group, 4, 4)
    acc += [group_res]
    # print('Working on name:' + str(name))
    # print(acc)

    if len(group_res) > max_size:
        # Update the max_size that we have observed so far
        max_size = len(group_res)

        # All existing lists (apart from the one added latest)in acc need to be augmented
        # to match the new max_size by appending the last value in those list (combining train and test)
        for idx, list_i in enumerate(acc):
            if len(list_i) < max_size:
                last_train, last_test = list_i[-1][0], list_i[-1][1]
                list_i[len(list_i):max_size] = [(last_train.union(last_test),
                                                 pd.Index([]))] * (max_size - len(list_i))

                acc[idx] = list_i

    elif len(group_res) < max_size:
        # Only the last appended list (group_res) needs to be augmented
        last_train, last_test = acc[-1][-1][0], acc[-1][-1][1]
        acc[-1] = acc[-1] + [(last_train.union(last_test), pd.Index([]))] * (max_size - len(acc[-1]))


print(acc[0:2])
# acc now contains a list of lists, where each internal list contains tuples of train_index, test_index
# [[(group_1_train_index1, group_1_test_index1), (group_1_train_index2, group_1_test_index2)],
#  [(group_2_train_index1, group_2_test_index1), (group_2_train_index2, group_2_test_index2)],
#  [(group_3_train_index1, group_3_test_index1), (group_3_train_index2, group_3_test_index2)]]
#
# Our goal is to drill-down by removing group-divisions:
# [(train_index1, test_index1), (train_index2, test_index2)]
flat_acc = []
for idx, list_i in enumerate(acc):
    if len(flat_acc) == 0:
        flat_acc += list_i
        continue

    for inner_idx, tuple_i in enumerate(list_i):
        flat_acc[inner_idx] = (flat_acc[inner_idx][0].union(tuple_i[0]),
                               flat_acc[inner_idx][1].union(tuple_i[1]))


print(flat_acc[0:2])
# Convert Financial Distress column into 0 or 1
df_model = df_cross.copy()
df_model['Financial Distress'] = ['0' if x > -0.50 else '1' for x in df_model['Financial Distress'].values]

df_model.head()
# For each entry in flat_acc, perform train and test and plot metrics
dependent_cols = [col for col in df_model.columns if col != 'Financial Distress']
independent_col = ['Financial Distress']
for idx, tuple_i in enumerate(flat_acc):
    print('---------------------------------------')
    X_train, X_test = df_model.loc[tuple_i[0]][dependent_cols], df_model.loc[tuple_i[1]][dependent_cols]
    y_train, y_test = df_model.loc[tuple_i[0]][independent_col], df_model.loc[tuple_i[1]][independent_col]
    
    # Fit logistic regression model to train data and test on test data
    lr_mod = LogisticRegression(C=0.01, penalty='l2')  # These should be determined by nested cv
    lr_mod.fit(X_train, y_train)
    
    y_pred_proba = lr_mod.predict_proba(X_test)
    y_pred = lr_mod.predict(X_test)
    
    # Print Confusion Matrix and ROC AUC score
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    print('ROC AUC score:')
    print(roc_auc_score(y_test['Financial Distress'].astype(int), y_pred_proba[:, 1]))