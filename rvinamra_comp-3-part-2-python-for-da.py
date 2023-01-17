# Continuation of WILL KOEHRSEN'S notebook used to learn analysis 
# using Python

# here we will be using bureau and bureau_balance data apart from the 
# training data from the application zip fle

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# list of panda operations which will be used extensively:
# groupby: group a dataframe by column (here, SK_ID_CURR)
# agg: perform a calculation on the grouped data like mean, max, min etc.
# merge: match the aggregated statistics to the appropriate client

# reading the bureau file
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()
#print('Shape of the bureau file', bureau.shape)
# groupby the client id, counting the previous loans and renaming the column

previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index = False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
# joining with the training dataframe
train = pd.read_csv('../input/application_train.csv')
train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

# filling the missing values with 0
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()
# using the r-value and kde plots to find the usefulness of a variable

def kde_target(var_name, df):
    
    # Calculating the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculating medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plotting the distribution for target = 0 and target = 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # printing out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    
    # Printing out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    
    
# testing this with EXT_SOURCE_3 as it was found one of the most important 
# variable in Random Forest and Gradient Boosting method

kde_target('EXT_SOURCE_3', train)
# testing with the new variable created - previous_loan_counts
kde_target('previous_loan_counts', train)
# from the above plot, we can't be very certain about the impact of
# previous_loan_count on TARGET 

# creating new variables from the bureau file

bureau_agg = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
bureau_agg.head()
# creating new variable names for each column
# list of column names

columns = ['SK_ID_CURR']

for var in bureau_agg.columns.levels[0]:
    # skiping the id name as it is already added
    if var != 'SK_ID_CURR':
            for stat in bureau_agg.columns.levels[1][:-1]:
                # making a new column for the variable and stat
                columns.append('bureau_%s_%s' % (var, stat))
# assigning the list of column names as dataframe column names
bureau_agg.columns = columns
bureau_agg.head()
# merging the new columns with the training data

train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
train.head()
# correaltion of the new variables with the target

# list of new correlations
new_corrs = []

for col in columns:
    # calculation correlation with the target
    corr = train['TARGET'].corr(train[col])
    
    # appending the list as a tuple
    new_corrs.append((col, corr))
# sorting and displaying the correlations

new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
new_corrs[:15]
# none of the new variables have a significant correlation with the 
# TARGET variable
# looking at the kde plot of the highest correlated variable

# bureau_DAYS_CREDIT_mean:- How many days before current application
# did client apply for Credit Bureau credit

kde_target('bureau_DAYS_CREDIT_mean', train)
# encapsulating all the work related to numeric aggregation into 
# a function so as to use them with other dataframes in the future

def agg_numeric(df, group_var, df_name):
    """ Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # removing the id variable other than the grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
        
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    
    # grouping by the specified variable and calculating the stats
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    
    # creating new column names
    columns = [group_var]
    
    for var in agg.columns.levels[0]:
        # skipping the grouping variable
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                # making a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
            
    agg.columns = columns
    return agg
# checking whether the  function returns the same dataframe as before

bureau_agg_new = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg_new.head()
bureau_agg.head()
# the values match perfectly

# another function to calculate correaltion of a variable with the 'TARGET'

def target_corrs(df):
    
    # list of correlations
    corrs = []
    for col in df.columns:
        print(col)
        #skipping the target columns as the r = 1 
        if col != 'TARGET':
            corr = df['TARGET'].corr(df[col])
            
            # appending the list as a tuple
            corrs.append((col, corr))
            
    # sorting by their magnitude of correlations
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    
    return corrs
# checking the categorical columns
# one-hot encoding for categorical columns

categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical.head()
categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum', 'mean'])
categorical_grouped.head()
# sum represents the count for an individual and mean represents normaized count
# renaming columns

group_var = 'SK_ID_CURR'
columns = []

for var in categorical_grouped.columns.levels[0]:
    if var != group_var:
        for stat in ['count', 'count_norm']:
            columns.append('%s_%s' %(var, stat))
            
categorical_grouped.columns = columns
categorical_grouped.head()
# merging this dataframe into the training data
train = train.merge(categorical_grouped, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
train.head()
train.iloc[:10, 123:]
# function to handle categorical variables
# it will calculate the counts and normalized counts

def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    # selecting the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    
    # putting identifying id on the column
    categorical[group_var] = df[group_var]
    
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    column_names = []
    
    for var in categorical.columns.levels[0]:
        for stat in ['count', 'count_norm']:
            column_names.append('%s_%s_%s' % (df_name, var, stat))
            
    categorical.columns = column_names
    return categorical
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
# using the bureau balance dataframe
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau_balance.head()
# counting each type of status for each previous loan

bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_counts.head()
# calculating aggregation stats using a previously defined function -- agg_numeric
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()
# aggregating dataframes for each client

bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# including SK_ID_CURR
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')

bureau_by_loan.head()
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
bureau_balance_by_client.head()
# Recap of all the things doen with bureau_balance dataframe:
# Calculated numeric stats grouping by each loan
# Made value counts of each categorical variable grouping by loan
# Merged the stats and the value counts on the loans
# Calculated numeric stats for the resulting dataframe grouping by the client id

# the final dataframe has one row for each client
# stats has been calculated for all of their loans with monthly information
# before putting all these information in the main dataframe, resetting all the dataframes
# using the function to build all the variables from the grounds up

# freeing up memory by deleting old objects
import gc
gc.enable()
del train, bureau, bureau_balance, bureau_agg, bureau_agg_new, bureau_balance_agg, bureau_balance_counts, bureau_by_loan, bureau_balance_by_client, bureau_counts
gc.collect()
# rereading new copies of all the dataframes

train = pd.read_csv('../input/application_train.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')

# counts in bureau dataframe
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
# aggregated stats of bureau dataframe
bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()
# counting values of bureau_balance df by loan

bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_counts.head()
# aggregated stats of bureau_balance by loan

bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()
# aggregated stats of bureau_balance by client
# grouping dataframe by loan

bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# merging to include the SK_ID_CURR

bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how  ='left')

# aggregating stats for each client

bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')

# inserting computed features in the training data

original_features = list(train.columns)
print('Original Number of Features: ', len(original_features))
# merging with the value counts of bureau

train = train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')

# merging with the stats of bureau

train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# merging merging with the monthly info grouped by client

train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
new_features = list(train.columns)
print('Number of features using previous loans from other insti data: ', len(new_features))
# dealing with the missing values
# function to calculate missing values by column

def missing_values_table(df):
    #total missing values
    mis_val = df.isnull().sum()
    
    # percentage of missing values
    mis_val_percent = 100*df.isnull().sum() / len(df)
    
    # making a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
    
    # renaming the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0: 'Missing Values', 1: '% of Total Values'})
    
    # sorting the table by percentage of missing in descending order
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1]!=0].sort_values('% of Total Values', ascending=False).round(1)
    
    # printing summary info
    print ("Your selected dataframe has "+ str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values")
    
    # returning the dataframe with missing info
    return mis_val_table_ren_columns
missing_train = missing_values_table(train)
missing_train.head(10)
# check to see whether any columns have more than 90% missing values (I don't there ain't any but just for future references and maintaining an order)
missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])
len(missing_train_vars)
# merging test data with the appropriate data

test = pd.read_csv('../input/application_test.csv')

# merging with the value counts of bureau, stats of bureau, and value counts of bureau_balance

test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
print('Shape of Test data: ', test.shape)
# aligning test and train datasets
# matching up their column number for compatibility

train_labels = train['TARGET']

train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels

print('Training data shape:', train.shape)
print('Testing data shape:', test.shape)
# checking the %age of missing values in testing data

missing_test = missing_values_table(test)
missing_test.head(10)
# none of them have missing values > 90%, might have to use another feature selection to reduce the dimensionality
# saving the changed train and test data

# CORRELATIONS

corrs = train.corr()
corrs = corrs.sort_values('TARGET', ascending = False)

# ten most positive correlations
pd.DataFrame(corrs['TARGET'].head(10))
# ten most negative correlations
pd.DataFrame(corrs['TARGET'].dropna().tail(10))
kde_target(var_name='bureau_CREDIT_ACTIVE_Active_count_norm', df = train)
# due to weak correlation the plot looks completely randoma and it won't be wise to draw any conclusions from it 

# to check collinearity in the data, finding correlations of one variable with another which might help in reducing dimensionality

# checking variable pairs with correlation greater 0.8
threshold = 0.8

# empty dictionary to hold correlated variables
above_threshold_vars = {}

for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
# for the pairs of highly correlated variables we need to only one out of them
# tracking columns to remove and columns already examined

cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# iterating through columns and correlated columns
for key, value in above_threshold_vars.items():
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # removing only one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
                
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))
    
# removing these variables from both test and training datasets

train_corrs_removed = train.drop(columns = cols_to_remove)
test_corrs_removed = test.drop(columns = cols_to_remove)

print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)
# saving new datasets created

train_corrs_removed.to_csv('train_bureau_corrs_removed.csv', index = False)
test_corrs_removed.to_csv('test_bureau_corrs_removed.csv', index = False)
# MODELLING
# control: only the data in application files
# test one: data in application files along all the data recorded from bureau and bureau_balance
# test two: data in the application files with all the data recorded from the bureau and bureau_balance files with highly correlated variables removed 

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import gc

import matplotlib.pyplot as plt
# comparing the raw version and the version with high corelations removed
def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics
    
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        
    """
    # sorting feature according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
# establishing a control
# using function described above that used GBM with the single main source application data 
train_control = pd.read_csv('../input/application_train.csv')
test_control = pd.read_csv('../input/application_test.csv')
# the GBM functions returns a submission df which ca be uploaded to the competition, a f1 dataframe of feature importances adn metrics df with validation and test performance

submission, f1, metrics = model(train_control, test_control)
metrics
# extra steps are required for regularization will help to remove this slight overfitting 
# visualizing feature importance 
fi_sorted = plot_feature_importances(f1)
submission.to_csv('control.csv', index = False)
# running model on test one

submission_raw, fi_raw, metrics_raw = model(train, test)
metrics_raw
# engineered df perform better than control ones
fi_raw_sorted = plot_feature_importances(fi_raw)
# some of the engineered features make it to the most important list

top_100 = list(fi_raw_sorted['feature'])[:100]
new_features = [x for x in top_100 if x not in list(f1['feature'])]

print('%% of Top 100 Features created from the bureau data = %d.00' %len(new_features))

submission_raw.to_csv('test_one.csv', index = False)
# test_two, highly correlated variables removed 

submission_corrs, fi_corrs, metrics_corr = model(train_corrs_removed, test_corrs_removed)
