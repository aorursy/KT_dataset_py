import pandas as pd

import numpy as np

import math

import string

import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV, ParameterGrid

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import category_encoders as ce



from sys import getsizeof

from datetime import datetime

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test_data = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
print('Train data size : {} MB'.format(round(getsizeof(train_data) / 1024 / 1024)))

print('Test data size : {} MB'.format(round(getsizeof(test_data) / 1024 / 1024)))
print(train_data.shape)

print(test_data.shape)
train_data.head()
test_data.head()
# List down the ord_5 variable values with different cases.



tmp_df = pd.DataFrame(train_data['ord_5'].unique(), columns = ['val'])

tmp_df['trans_val'] = tmp_df['val'].str.upper()



list_1 = []



for trans_val in tmp_df['trans_val'].unique():

    val_cnt = sum(tmp_df['trans_val'] == trans_val)

    

    for val in tmp_df.loc[(tmp_df['trans_val'] == trans_val) & (tmp_df['val'] != trans_val), 'val']:

        if val_cnt == 1:

            break

    list_1.append(val)
list_1[:20]
tmp_df.loc[tmp_df['trans_val'] == 'AP', ]
# Variables to exclude from change case operation.

excl_cols = ['ord_5']



filter_cond = (train_data.dtypes == 'object') & (~train_data.dtypes.index.isin(excl_cols))

cols_for_change_case = train_data.dtypes[filter_cond].index.tolist()

cols_for_change_case
# Create a data frame to hold unique value count for the selected variables of train data set before changing values to lower case.



bef_tr_varunq_count = pd.DataFrame(train_data.loc[:, cols_for_change_case].nunique().sort_values(ascending = False))

bef_tr_varunq_count.reset_index(inplace = True)

bef_tr_varunq_count.rename(columns = {'index':'variable', 0:'unique_val_count'}, inplace = True)
# Create a data frame to hold unique value count for the selected variables of test data set before changing values to lower case.



bef_ts_varunq_count = pd.DataFrame(test_data.loc[:, cols_for_change_case].nunique().sort_values(ascending = False))

bef_ts_varunq_count.reset_index(inplace = True)

bef_ts_varunq_count.rename(columns = {'index':'variable', 0:'unique_val_count'}, inplace = True)
# Change values of the train data set variables to lower case.



train_data.loc[:, cols_for_change_case] = train_data.loc[:, cols_for_change_case].apply(lambda x:x.astype(str).str.lower())
train_data.head()
# Change values of the test data set variables to lower case.



test_data.loc[:, cols_for_change_case] = test_data.loc[:, cols_for_change_case].apply(lambda x:x.astype(str).str.lower())
test_data.head()
# Create a data frame to hold unique value count for the selected variables of train data set after changing values to lower case.



aft_tr_varunq_count = pd.DataFrame(train_data.loc[:, cols_for_change_case].nunique().sort_values(ascending = False))

aft_tr_varunq_count.reset_index(inplace = True)

aft_tr_varunq_count.rename(columns = {'index':'variable', 0:'unique_val_count'}, inplace = True)
# Create a data frame to hold unique value count for the selected variables of test data set after changing values to lower case.



aft_ts_varunq_count = pd.DataFrame(test_data.loc[:, cols_for_change_case].nunique().sort_values(ascending = False))

aft_ts_varunq_count.reset_index(inplace = True)

aft_ts_varunq_count.rename(columns = {'index':'variable', 0:'unique_val_count'}, inplace = True)
aft_tr_varunq_count.shape
bef_tr_varunq_count.shape
# Joining unique value counts of the train data set variables before and after changing case

# to find out how many variables do not match on the basis of unique value counts.



aft_tr_varunq_count.merge(bef_tr_varunq_count, 

                          right_on = ['variable', 'unique_val_count'], 

                          left_on = ['variable', 'unique_val_count'])['variable'].count()
# Joining unique value counts of the test data set variables before and after changing case

# to find out how many variables do not match on the basis of unique value counts.



aft_ts_varunq_count.merge(bef_ts_varunq_count, 

                          right_on = ['variable', 'unique_val_count'], 

                          left_on = ['variable', 'unique_val_count'])['variable'].count()
# Bring unique value count of both train and test data sets together after changing values of the variables to lower case.

# We are going this to figure out if there are any variables in test data set with fewer unique value counts than train data set.



tmp_df = aft_tr_varunq_count.merge(aft_ts_varunq_count,

                                   how = 'inner',

                                   left_on = 'variable',

                                   right_on = 'variable',

                                   suffixes = ['_tr', '_ts'])
print('No. of variables with unique value count of train data set less than test data set : {}'.\

      format(sum(tmp_df['unique_val_count_tr'] < tmp_df['unique_val_count_ts'])))
print('Train data size : {} MB'.format(round(getsizeof(train_data) / 1024 / 1024)))

print('Test data size : {} MB'.format(round(getsizeof(test_data) / 1024 / 1024)))



# Before changing case of the data, sizes of train and test data set were 332 MB and 220 MB.
# Deleting variables which are no longer required.



del tmp_df, bef_tr_varunq_count, aft_tr_varunq_count, bef_ts_varunq_count, aft_ts_varunq_count, cols_for_change_case
# Spliting two character value into two single character values and converting them into numbers according

# to sequence of letters defined in string.ascii_letters did not help me much in improving the score.



# train_data['ord_5_1'] = train_data['ord_5'].apply(lambda x : string.ascii_letters.index(x[0]) + 1)

# train_data['ord_5_2'] = train_data['ord_5'].apply(lambda x : string.ascii_letters.index(x[1]) + 1)



# test_data['ord_5_1'] = test_data['ord_5'].apply(lambda x : string.ascii_letters.index(x[0]) + 1)

# test_data['ord_5_2'] = test_data['ord_5'].apply(lambda x : string.ascii_letters.index(x[1]) + 1)





# Split two character values of ord_5 variable into two single character values.



train_data['ord_5_1'] = train_data['ord_5'].apply(lambda x : x[0])

train_data['ord_5_2'] = train_data['ord_5'].apply(lambda x : x[1])



train_data.drop(columns = 'ord_5', inplace = True)



test_data['ord_5_1'] = test_data['ord_5'].apply(lambda x : x[0])

test_data['ord_5_2'] = test_data['ord_5'].apply(lambda x : x[1])



test_data.drop(columns = 'ord_5', inplace = True)
# Transforming ord_5 two character values into concatenated numeric values of the corresponding 

# string.ascii_letters value did not improve the score of the model. Hence commenting the code.



# train_data['ord_5_1_2'] = train_data['ord_5'].apply(lambda x : str(string.ascii_letters.index(x[0]) + 1) + str(string.ascii_letters.index(x[1]) + 1))

# test_data['ord_5_1_2'] = test_data['ord_5'].apply(lambda x : str(string.ascii_letters.index(x[0]) + 1) + str(string.ascii_letters.index(x[1]) + 1))



# train_data['ord_5_1_2'] = train_data['ord_5_1_2'].astype('int64')

# test_data['ord_5_1_2'] = test_data['ord_5_1_2'].astype('int64')
# Transforming ord_5 two character values into sum of numeric values of the corresponding 

# string.ascii_letters value did not improve the score of the model. Hence commenting the code.



# train_data['ord_5_1plus2'] = train_data['ord_5'].apply(lambda x : (string.ascii_letters.index(x[0]) + 1) + (string.ascii_letters.index(x[1]) + 1))

# test_data['ord_5_1plus2'] = test_data['ord_5'].apply(lambda x : (string.ascii_letters.index(x[0]) + 1) + (string.ascii_letters.index(x[1]) + 1))
# Transforming ord_5 two character values into product of numeric values of the corresponding 

# string.ascii_letters value did not improve the score of the model. Hence commenting the code.



# train_data['ord_5_1mult2'] = train_data['ord_5'].apply(lambda x : (string.ascii_letters.index(x[0]) + 1) * (string.ascii_letters.index(x[1]) + 1))

# test_data['ord_5_1mult2'] = test_data['ord_5'].apply(lambda x : (string.ascii_letters.index(x[0]) + 1) * (string.ascii_letters.index(x[1]) + 1))
# train_data[['ord_5', 'ord_5_1_2', 'ord_5_1plus2', 'ord_5_1mult2']].head()
# Transforming day and month variables containing cyclical values into actual numeric values 

# did not improve the score of the model. Hence commenting the code.



# # For train data set



# train_data['day_sin'] = np.sin(train_data['day'] * (2. * np.pi / train_data['day'].max()))

# train_data['day_cos'] = np.cos(train_data['day'] * (2. * np.pi / train_data['day'].max()))



# train_data['month_sin'] = np.sin((train_data['month']) * (2. * np.pi / train_data['month'].max()))

# train_data['month_cos'] = np.cos((train_data['month']) * (2. * np.pi / train_data['month'].max()))



# # For test data set



# test_data['day_sin'] = np.sin(test_data['day'] * (2. * np.pi / test_data['day'].max()))

# test_data['day_cos'] = np.cos(test_data['day'] * (2. * np.pi / test_data['day'].max()))



# test_data['month_sin'] = np.sin((test_data['month']) * (2. * np.pi / test_data['month'].max()))

# test_data['month_cos'] = np.cos((test_data['month']) * (2. * np.pi / test_data['month'].max()))
# Print unique value counts for each of the bin* variables for train data set.



for col in train_data.columns[train_data.columns.str.contains('bin*')]:

    print(train_data[col].value_counts())

    print(train_data[col].dtype)
# Print unique value counts for each of the bin* variables for test data set.



for col in test_data.columns[test_data.columns.str.contains('bin*')]:

    print(test_data[col].value_counts())

    print(test_data[col].dtype)
# Map value 't' and 'f' to 1 and 0 respectively.



train_data['bin_3'] = train_data['bin_3'].apply(lambda x : 1 if x == 't' else 0)

test_data['bin_3'] = test_data['bin_3'].apply(lambda x : 1 if x == 't' else 0)
# Map value 'y' and 'n' to 1 and 0 respectively.



train_data['bin_4'] = train_data['bin_4'].apply(lambda x : 1 if x == 'y' else 0)

test_data['bin_4'] = test_data['bin_4'].apply(lambda x : 1 if x == 'y' else 0)
# Converting ord_3 and ord_4 values into corresponding string.ascii_letters numeric values 

# did not help me improve score of the model. Hence, commenting the code.



# train_data['ord_3'] = train_data['ord_3'].apply(lambda x : string.ascii_letters.index(x[0]) + 1)

# train_data['ord_4'] = train_data['ord_4'].apply(lambda x : string.ascii_letters.index(x[0]) + 1)



# test_data['ord_3'] = test_data['ord_3'].apply(lambda x : string.ascii_letters.index(x[0]) + 1)

# test_data['ord_4'] = test_data['ord_4'].apply(lambda x : string.ascii_letters.index(x[0]) + 1)
train_data['ord_3_4'] = train_data['ord_3'] + train_data['ord_4']

test_data['ord_3_4'] = test_data['ord_3'] + test_data['ord_4']
excl_cols = ['id', 'target']

excl_cols = excl_cols + ['bin_' + str(i) for i in range(5)]
# excl_cols
# Create a data frame listing unique value count and data type of each variable of train data set.



tr_unique_val_cnt_df = pd.DataFrame(train_data.loc[:, ~train_data.columns.isin(excl_cols)].nunique().sort_values(ascending = False))

tr_unique_val_cnt_df.reset_index(inplace =  True)

tr_unique_val_cnt_df.rename(columns = {'index':'variable', 0:'unique_val_count'}, inplace = True)

tr_unique_val_cnt_df = tr_unique_val_cnt_df.merge(pd.DataFrame(train_data.dtypes).reset_index().rename(columns = {'index':'variable', 0:'dtype'}),

                                                 right_on = 'variable', left_on = 'variable')
# Create a data frame listing unique value count and data type of each variable of test data set.



ts_unique_val_cnt_df = pd.DataFrame(test_data.loc[:, ~test_data.columns.isin(excl_cols)].nunique().sort_values(ascending = False))

ts_unique_val_cnt_df.reset_index(inplace = True)

ts_unique_val_cnt_df.rename(columns = {'index':'variable', 0:'unique_val_count'}, inplace = True)

ts_unique_val_cnt_df = ts_unique_val_cnt_df.merge(pd.DataFrame(test_data.dtypes).reset_index().rename(columns = {'index':'variable', 0:'dtype'}),

                                                 right_on = 'variable', left_on = 'variable')
tr_unique_val_cnt_df
ts_unique_val_cnt_df
tmp_df_1 = pd.DataFrame()

tmp_df_1 = train_data['id'].copy()

tmp_df_1_cols = []



for i in range(len(tr_unique_val_cnt_df)):

    if tr_unique_val_cnt_df.iloc[i, 1] <= 7:

        tmp_df_1 = pd.concat([tmp_df_1, pd.get_dummies(data = train_data.loc[:, tr_unique_val_cnt_df.iloc[i, 0]],

                                                       prefix = tr_unique_val_cnt_df.iloc[i, 0],

                                                       drop_first = True)], axis = 1)

        tmp_df_1_cols.append(tr_unique_val_cnt_df.iloc[i, 0])
sorted(tmp_df_1_cols)
tmp_df_1.shape
# tmp_df_1.head()
tmp_df_2 = pd.DataFrame()

tmp_df_2 = train_data['id'].copy()

tmp_df_2_cols = []



for i in range(len(tr_unique_val_cnt_df)):

    if tr_unique_val_cnt_df.iloc[i, 1] > 7 and tr_unique_val_cnt_df.iloc[i, 1] <= 20:

        tmp_df_2 = pd.concat([tmp_df_2, pd.get_dummies(data = train_data.loc[:, tr_unique_val_cnt_df.iloc[i, 0]],

                                                       prefix = tr_unique_val_cnt_df.iloc[i, 0],

                                                       drop_first = True)], axis = 1)

        tmp_df_2_cols.append(tr_unique_val_cnt_df.iloc[i, 0])
sorted(tmp_df_2_cols)
tmp_df_2.shape
# tmp_df_2.head()
filter_cond = (tr_unique_val_cnt_df['unique_val_count'] > 100)

tmp_df_3_cols = tr_unique_val_cnt_df.loc[filter_cond, 'variable'].tolist()



tmp_df_3_enc = ce.LeaveOneOutEncoder(cols = tmp_df_3_cols)



tmp_df_3 = tmp_df_3_enc.fit_transform(X = train_data.loc[:, ['id'] + tmp_df_3_cols], 

                                      y = train_data['target'])
sorted(tmp_df_3_cols)
tmp_df_3.shape
# tmp_df_3.head()
filter_cond = (tr_unique_val_cnt_df['unique_val_count'] > 20) & (tr_unique_val_cnt_df['unique_val_count'] <= 100)



# Create a data frame with id and the columns that satisfy above unique value count condition.

tmp_df_4 = train_data.loc[:, ['id'] + tr_unique_val_cnt_df.loc[filter_cond, 'variable'].tolist()]



# Create a list to store variable names encoded using this encoding technique.

tmp_df_4_cols = []



col_grp_main_df = pd.DataFrame()



# Flag used to create either "odds for" or "odds against" variable.

# Set to True to create odds for variable. False, otherwise.

pos_odd_flag = False



# Flag used to create either "odds ratio for" or "odds ratio against" variable.

# Set to True to create "odds ratio for" variable. False, otherwise.

pos_odd_ratio_flag = False



# This loop executes once for each variable.

for i in tr_unique_val_cnt_df.loc[filter_cond, ].index:

    

    var_name = tr_unique_val_cnt_df.iloc[i, 0]



    # Create variable names dynamically created for original variable.

    tot_count_var = var_name + '_tot_count'

    pos_count_var = var_name + '_pos_count'

    neg_count_var = var_name + '_neg_count'

    pos_prob_var = var_name + '_pos_prob'

    neg_prob_var = var_name + '_neg_prob'

    odds_var = var_name + '_odds'

    log_odds_var = var_name + '_log_odds'

    odds_ratio_var = var_name + '_odds_ratio'

    log_odds_ratio_var = var_name + '_log_odds_ratio'

    mean_var = var_name + '_mean'

    variance_var = var_name + '_variance'

    

    # Compute unique value count for each variable.

    grp_main_df = pd.DataFrame(train_data[var_name].value_counts())

    grp_main_df.reset_index(inplace = True)

    grp_main_df.rename(columns = {var_name:tot_count_var}, inplace = True)

    

    # Compute unique value count for each variable for negative labelled (target == 0) observations.

    grp_neg_df = pd.DataFrame(train_data.loc[train_data['target'] == 0, var_name].value_counts())

    grp_neg_df.reset_index(inplace = True)

    grp_neg_df.rename(columns = {var_name:neg_count_var}, inplace = True)



    # Compute unique value count for each variable for positive labelled (target == 1) observations.

    grp_pos_df = pd.DataFrame(train_data.loc[train_data['target'] == 1, var_name].value_counts())

    grp_pos_df.reset_index(inplace = True)

    grp_pos_df.rename(columns = {var_name:pos_count_var}, inplace = True)



    # Compute variance of the target variable for each unique value of the predictor.

    grp_variance_df = pd.DataFrame(train_data.groupby([var_name])['target'].var())

    grp_variance_df.reset_index(inplace = True)

    grp_variance_df.rename(columns = {var_name:'index', 'target':variance_var}, inplace = True)

    

    # Merge above computed values into a single data frame.

    grp_main_df = grp_main_df.merge(grp_pos_df, on = 'index', how = 'left')     

    grp_main_df = grp_main_df.merge(grp_neg_df, on = 'index', how = 'left')

    grp_main_df = grp_main_df.merge(grp_variance_df, on = 'index', how = 'left')

    grp_main_df.fillna(0, inplace = True)



    # Compute positive probability and negative probability.

    grp_main_df[pos_prob_var] = grp_main_df[pos_count_var] / grp_main_df[tot_count_var]

    grp_main_df[neg_prob_var] = grp_main_df[neg_count_var] / grp_main_df[tot_count_var]



    # Compute odds for or odds against values.

    if pos_odd_flag:

        grp_main_df[odds_var] = grp_main_df[pos_prob_var] / grp_main_df[neg_prob_var]

    else:

        grp_main_df[odds_var] = grp_main_df[neg_prob_var] / grp_main_df[pos_prob_var]



    # Handling zero or infinite (+/-) values resulted from odds value computation.

    grp_main_df.loc[grp_main_df[odds_var] == 0, odds_var] = 1

    grp_main_df.loc[grp_main_df[odds_var] == float('inf'), odds_var] = .1

    grp_main_df.loc[grp_main_df[odds_var] == float('-inf'), odds_var] = .1



    # Compute log-odds value.

    grp_main_df[log_odds_var] = grp_main_df[odds_var].apply(lambda x : np.log(.1) if math.isinf(x) else np.log(x))



    tot_pos_count = grp_main_df[pos_count_var].sum()

    tot_neg_count = grp_main_df[neg_count_var].sum()



    # Compute odds for ratio or odds against ratio values.

    if pos_odd_ratio_flag:

        grp_main_df[odds_ratio_var] = grp_main_df.apply(lambda x : (x[pos_count_var] / (tot_pos_count - x[pos_count_var])) / (x[neg_count_var] / (tot_neg_count - x[neg_count_var])), axis = 1)

    else:

        grp_main_df[odds_ratio_var] = grp_main_df.apply(lambda x : (x[neg_count_var] / (tot_neg_count - x[neg_count_var])) / (x[pos_count_var] / (tot_pos_count - x[pos_count_var])), axis = 1)



    # Handling zero or infinite (+/-) values resulted from odds ratio value computation.

    grp_main_df.loc[grp_main_df[odds_ratio_var] == 0, odds_ratio_var] = 1

    grp_main_df.loc[grp_main_df[odds_ratio_var] == float('inf'), odds_ratio_var] = 1

    grp_main_df.loc[grp_main_df[odds_ratio_var] == float('-inf'), odds_ratio_var] = 1



    # Compute log-odds ratio value.

    grp_main_df[log_odds_ratio_var] = grp_main_df[odds_ratio_var].apply(lambda x : np.log(.1) if math.isinf(x) else np.log(x))



    # Rename pos_prob column of a variable to variance of the same variable.

    grp_main_df.rename(columns = {pos_prob_var:mean_var}, inplace = True)



    # We do not need these variables anymore. Hence, adding these variables to drop list.

    cols_to_drop = [tot_count_var, pos_count_var, neg_count_var, neg_prob_var]



    if len(cols_to_drop) > 0:

        grp_main_df.drop(columns = cols_to_drop, inplace = True)



    tmp_df_4 = tmp_df_4.merge(grp_main_df, right_on = 'index', left_on = var_name)

    tmp_df_4.drop(columns = ['index', var_name], inplace = True)



    grp_main_df.rename(columns = {log_odds_var:'log_odds'}, inplace = True)

    grp_main_df.rename(columns = {log_odds_ratio_var:'log_odds_ratio'}, inplace = True)



    grp_main_df.rename(columns = {odds_var:'odds'}, inplace = True)

    grp_main_df.rename(columns = {odds_ratio_var:'odds_ratio'}, inplace = True)



    grp_main_df.rename(columns = {mean_var:'mean'}, inplace = True)

    grp_main_df.rename(columns = {variance_var:'variance'}, inplace = True)



    col_grp_main_df = pd.concat([col_grp_main_df, 

                                pd.concat([pd.DataFrame([var_name] * grp_main_df.shape[0], columns = ['variable']), grp_main_df], axis = 1)], 

                                axis = 0,

                                sort = False)



    tmp_df_4_cols.append(var_name)
sorted(tmp_df_4_cols)
# Inf values check.

(~np.isfinite(col_grp_main_df.iloc[:, 2:]) & ~col_grp_main_df.iloc[:, 2:].isna()).sum()
del grp_main_df, grp_pos_df, grp_neg_df, var_name, tot_count_var, pos_count_var, neg_count_var, pos_prob_var, neg_prob_var

del odds_var, log_odds_var, odds_ratio_var, log_odds_ratio_var
tmp_df_4.shape
# tmp_df_4.head()
addnl_cols = ['id', 'target']

addnl_cols = addnl_cols + ['bin_' + str(i) for i in range(5)]
addnl_cols
print(tmp_df_1.shape)

print(tmp_df_2.shape)

print(tmp_df_3.shape)

print(tmp_df_4.shape)
tr_enc_data = pd.DataFrame()

tr_enc_data = tmp_df_1.merge(tmp_df_2, on = 'id')

tr_enc_data = tr_enc_data.merge(tmp_df_3, on = 'id')

tr_enc_data = tr_enc_data.merge(tmp_df_4, on = 'id')

tr_enc_data = tr_enc_data.merge(train_data[addnl_cols], on = 'id')

tr_enc_data.shape
train_data['target'].value_counts()
tr_enc_data['target'].value_counts()
print('Train dataset size : {} MB'.format(round(getsizeof(train_data) / 1024 / 1024)))
print('Encoded train dataset size : {} MB'.format(round(getsizeof(tr_enc_data) / 1024 / 1024)))
del tmp_df_1, tmp_df_2, tmp_df_3, tmp_df_4
# Check for columns with duplicate column names.

if tr_enc_data.columns.duplicated().sum() > 0:

    print(tr_enc_data.columns[tr_enc_data.columns.duplicated().sum()])
tmp_df_1_cols
tmp_df_1 = pd.DataFrame()

tmp_df_1 = test_data['id'].copy()



for col in tmp_df_1_cols:

    tmp_df_1 = pd.concat([tmp_df_1, pd.get_dummies(data = test_data.loc[:, col],

                                                   prefix = col,

                                                   drop_first = True)], axis = 1)
tmp_df_1.shape
tmp_df_2_cols
tmp_df_2 = pd.DataFrame()

tmp_df_2 = test_data['id'].copy()



for col in tmp_df_2_cols:

    tmp_df_2 = pd.concat([tmp_df_2, pd.get_dummies(test_data.loc[:, col],

                                                   prefix = col,

                                                   drop_first = True)], axis = 1)
tmp_df_2.shape
# tmp_df_2.head()
tmp_df_3_cols
tmp_df_3 = tmp_df_3_enc.transform(X = test_data.loc[:, ['id'] + tmp_df_3_cols])
# Missing value check.

tmp_df_3.isnull().sum()[tmp_df_3.isnull().sum() > 0]
# Infinite value check.

(~np.isfinite(tmp_df_3) & ~tmp_df_3.isna()).sum()[(~np.isfinite(tmp_df_3) & ~tmp_df_3.isna()).sum() > 0]
tmp_df_4_cols
tmp_df_4 = test_data.loc[:, ['id'] + tmp_df_4_cols]



for col in tmp_df_4_cols:



    tmp_col_grp_main_df = col_grp_main_df.loc[col_grp_main_df['variable'] == col, 

                                              ['index', 'log_odds', 'log_odds_ratio', 'odds', 'odds_ratio', 'mean', 'variance']].copy()



    tmp_col_grp_main_df.rename(columns = {'log_odds' : col + '_log_odds',

                                          'log_odds_ratio' : col + '_log_odds_ratio'}, inplace = True)

    

    tmp_col_grp_main_df.rename(columns = {'odds' : col + '_odds',

                                          'odds_ratio' : col + '_odds_ratio'}, inplace = True)

    

    tmp_col_grp_main_df.rename(columns = {'mean' : col + '_mean',

                                          'variance' : col + '_variance'}, inplace = True)

    

    tmp_df_4 = tmp_df_4.merge(tmp_col_grp_main_df, right_on = 'index', left_on = col, how = 'left')



    tmp_df_4.drop(columns = ['index', col], inplace = True)
# Missing value check.

tmp_df_4.isnull().sum()[tmp_df_4.isnull().sum() > 0]
# Infinite value check.

(~np.isfinite(tmp_df_4) & ~tmp_df_4.isna()).sum()[(~np.isfinite(tmp_df_4) & ~tmp_df_4.isna()).sum() > 0]
tmp_df_4.shape
addnl_cols.remove('target')



ts_enc_data = pd.DataFrame()

ts_enc_data = tmp_df_1.merge(tmp_df_2, on = 'id')

ts_enc_data = ts_enc_data.merge(tmp_df_3, on = 'id')

ts_enc_data = ts_enc_data.merge(tmp_df_4, on = 'id')

ts_enc_data = ts_enc_data.merge(test_data[addnl_cols], on = 'id')
print('Test dataset size : {} MB'.format(round(getsizeof(test_data) / 1024 / 1024)))
print('Encoded test dataset size : {} MB'.format(round(getsizeof(ts_enc_data) / 1024 / 1024)))
print(tr_enc_data.shape)

print(ts_enc_data.shape)
# Duplicate column name check.

pd.Series(tr_enc_data.columns)[pd.Series(tr_enc_data.columns).duplicated()]
# Duplicate column name check.

pd.Series(ts_enc_data.columns)[pd.Series(ts_enc_data.columns).duplicated()]
print('Test data columns not found in Train data : {}'.format(set(ts_enc_data.columns).difference(set(tr_enc_data.columns))))

print('Train data columns not found in Test data : {}'.format(set(tr_enc_data.columns).difference(set(ts_enc_data.columns))))
# Missing value check.

print(tr_enc_data.isna().sum().sum())

print(ts_enc_data.isna().sum().sum())
# train_data.columns
# tr_enc_data.columns[tr_enc_data.columns.str.contains('ord')]
del tmp_df_1, tmp_df_2, tmp_df_3, tmp_df_4, tmp_col_grp_main_df
def create_submn_file (file_prefix, file_data_df):

    

    '''

    This function is used to create date-time stamped submission file.



    Parameters:

        file_prefix : String to be used as file prefix.

        file_data_df : DataFrame containing data to be written to the file.

    '''

    curr_date_time = datetime.now()

    file_name = file_prefix + '_Submn_File_%d%d%d_%d%d%d.csv' % (curr_date_time.year, curr_date_time.month, curr_date_time.day, curr_date_time.hour, curr_date_time.minute, curr_date_time.second)

    file_data_df.to_csv('./submission_files/' + file_name, index = False)
def make_predictions (model_estimator, X_test, prediction_type = 'class'):



    '''

    This function is used to predict either class or class probability based on 

    the estimator and test data passed as input to this function.



    Parameters:

        model_estimator : Model estimator trained on train data set.

        X_test : Test data set on which model estimator is applied to make predictions.

        prediction_type : Possible values are - 'class' and 'proba'.

                            - 'class' : Default value. Used to make class predictions.

                            - 'proba' : Used to make class probability predictions.

    

    Returns : Class or Class probability preductions.

    '''



    if (prediction_type == 'class'):

        test_pred = model_estimator.predict(X_test)

    elif (prediction_type == 'proba'):

        test_pred = model_estimator.predict_proba(X_test)

        if (test_pred.ndim > 1):

            test_pred = test_pred[:, 1]

        

    return test_pred
def prepare_file_data (model_estimator, X_test, primary_column_data, opt_threshold = 0, prediction_type = 'proba'):



    '''

    This function is used to prepare data to be written to the submission file.



    Parameters:

        model_estimator : Model estimator trained on train data set.

        X_test : Test data set on which model estimator is applied to make predictions.

        primary_column_data : Primary key data expected in the first column of the submission file which in this problem is 'id'.

        opt_threshold : Not relevant for this problem.

        prediction_type : Possible values are - 'class' and 'proba'.



    Returns a DataFrame with two variables: id, target (class probability predictions).

    '''

    

    test_pred = make_predictions(model_estimator, X_test, prediction_type)

    

    if (prediction_type == 'class'):

        None

    

    elif (prediction_type == 'proba'):

        None

        

#         test_pred_list = list(range(len(test_pred)))



#         for i in range(len(test_pred_list)):

#             test_pred_list[i] = 1 if test_pred[i] > opt_threshold else 0

            

#         test_pred = test_pred_list.copy()

    

    file_data = pd.DataFrame({'id':primary_column_data, 'target':test_pred})

    

    return file_data
def plot_cv_results(estimator):



    '''

    This function is used to plot validation scores for each fold of K-fold cross validation.



    Parameters:

        estimator : Model estimator trained on train data set.

    '''



    plt.figure(figsize = (6, 5))



    for idx, i in enumerate(estimator.scores_[1]):

        plt.plot(np.log(estimator.Cs_), i, label = 'Fold-' + str(idx + 1))



    plt.legend()

    plt.show()
cols_to_excl = []



col_names_list = ['ord_5_1', 'ord_5_2', 'ord_4']



# Comment the any one or more lines below to include them in modeling phase.



# cols_to_excl = cols_to_excl + ['_'.join(i) for i in zip(col_names_list, ['log_odds'] * len(col_names_list))]

cols_to_excl = cols_to_excl + ['_'.join(i) for i in zip(col_names_list, ['log_odds_ratio'] * len(col_names_list))]

cols_to_excl = cols_to_excl + ['_'.join(i) for i in zip(col_names_list, ['odds_ratio'] * len(col_names_list))]

cols_to_excl = cols_to_excl + ['_'.join(i) for i in zip(col_names_list, ['odds'] * len(col_names_list))]

cols_to_excl = cols_to_excl + ['_'.join(i) for i in zip(col_names_list, ['mean'] * len(col_names_list))]

cols_to_excl = cols_to_excl + ['_'.join(i) for i in zip(col_names_list, ['variance'] * len(col_names_list))]



model_train_data = tr_enc_data.copy()

model_test_data = ts_enc_data.copy()



if (len(cols_to_excl) > 0):

    model_train_data = model_train_data.drop(columns = cols_to_excl, axis = 1)

    model_test_data = model_test_data.drop(columns = cols_to_excl, axis = 1)
# list(model_train_data.columns)
# len(cols_to_excl)
# model_train_data.head()
# model_test_data.head()
print(model_train_data.shape)

print(model_test_data.shape)
X = model_train_data.drop(['id', 'target'], axis = 1)

y = model_train_data['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
log_reg = LogisticRegression(solver = 'lbfgs', random_state = 100, max_iter = 1000)

log_reg.fit(X_train, y_train)
test_pred = make_predictions(log_reg, X_test, 'proba')

print('Model ROC AUC score: {}'.format(roc_auc_score (y_test, test_pred)))



# Model ROC AUC score: 0.8037375992463995
# Prepare submission file based on class predictions.



file_data = prepare_file_data (log_reg, model_test_data.iloc[:, range(1,model_test_data.shape[1])], model_test_data['id'])

# create_submn_file ('logreg', file_data)



file_data.to_csv('submission.csv', index = False)



# 0.80459
lr_l2_cv = LogisticRegressionCV(random_state = 100, 

                                solver = 'liblinear', 

                                scoring = 'roc_auc',

                                penalty = 'l2',

                                cv = 10, 

                                max_iter = 1000,

                                verbose = 3)



lr_l2_cv.fit(X_train, y_train)
plot_cv_results(lr_l2_cv)
test_pred = make_predictions(lr_l2_cv, X_test, 'proba')

print('Model ROC AUC score: {}'.format(roc_auc_score (y_test, test_pred)))



# Model ROC AUC score: 0.8037420416539948

# Model ROC AUC score: 0.8037513913858887
file_data = prepare_file_data (lr_l2_cv, model_test_data.iloc[:, range(1,model_test_data.shape[1])], model_test_data['id'])

# create_submn_file ('lr_l2_cv', file_data)



file_data.to_csv('submission.csv', index = False)



# 0.80481