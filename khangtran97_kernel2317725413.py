import numpy as np

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

import os

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns
print(os.listdir("../input/"))
train_df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

print('Training data shape: ', train_df.shape)

train_df.head()
# Testing data features

test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

print('Testing data shape: ', test_df.shape)

test_df.head()
train_df['TARGET'].value_counts()
train_df['TARGET'].astype(int).plot.hist();
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values = missing_values_table(train_df)

missing_values.head(20)
train_df.dtypes.value_counts()
train_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in train_df:

    if train_df[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(train_df[col].unique())) <= 2:

            # Train on the training data

            le.fit(train_df[col])

            # Transform both training and testing data

            train_df[col] = le.transform(train_df[col])

            test_df[col] = le.transform(test_df[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

train_df = pd.get_dummies(train_df)

test_df = pd.get_dummies(test_df)



print('Training Features shape: ', train_df.shape)

print('Testing Features shape: ', test_df.shape)
train_labels = train_df['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)



# Add the target back in

train_df['TARGET'] = train_labels



print('Training Features shape: ', train_df.shape)

print('Testing Features shape: ', test_df.shape)
train_df.head()
missing_values = missing_values_table(train_df)

missing_values.head(20)
(train_df['DAYS_BIRTH'] / -365).describe()
train_df['DAYS_EMPLOYED'].describe()
train_df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
train_df['DAYS_EMPLOYED_ANOM'] = train_df["DAYS_EMPLOYED"] == 365243



# Replace the anomalous values with nan

train_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



train_df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
test_df['DAYS_EMPLOYED_ANOM'] = test_df["DAYS_EMPLOYED"] == 365243

test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)



print('There are %d anomalies in the test data out of %d entries' % (test_df["DAYS_EMPLOYED_ANOM"].sum(), len(test_df)))
# Find correlations with the target and sort

correlations = train_df.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
from sklearn.impute import SimpleImputer



poly_features = train_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]

poly_features_test = test_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = SimpleImputer(strategy = 'median')



poly_target = poly_features['TARGET']



poly_features = poly_features.drop(columns = ['TARGET'])



# Need to impute missing values

poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)



from sklearn.preprocessing import PolynomialFeatures

                                  

# Create the polynomial object with specified degree

poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)



# Transform the features

poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)

print('Polynomial Features shape: ', poly_features.shape)
poly_features = pd.DataFrame(poly_features, 

                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))



# Add in the target

poly_features['TARGET'] = poly_target



# Find the correlations with the target

poly_corrs = poly_features.corr()['TARGET'].sort_values()



# Display most negative and most positive

print(poly_corrs.head(10))

print(poly_corrs.tail(5))
# Put test features into dataframe

poly_features_test = pd.DataFrame(poly_features_test, 

                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))



# Merge polynomial features into training dataframe

poly_features['SK_ID_CURR'] = train_df['SK_ID_CURR']

app_train_poly = train_df.merge(poly_features, on = 'SK_ID_CURR', how = 'left')



# Merge polnomial features into testing dataframe

poly_features_test['SK_ID_CURR'] = test_df['SK_ID_CURR']

app_test_poly = test_df.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')



# Align the dataframes

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)



# Print out the new shapes

print('Training data with polynomial features shape: ', app_train_poly.shape)

print('Testing data with polynomial features shape:  ', app_test_poly.shape)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer as Imputer



# Drop the target from the training data

if 'TARGET' in train_df:

    train = train_df.drop(columns = ['TARGET'])

else:

    train = train_df.copy()

    

# Feature names

features = list(train.columns)



# Copy of the testing data

test = test_df.copy()



# Median imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0, 1))



# Fit on the training data

imputer.fit(train)



# Transform both training and testing data

train = imputer.transform(train)

test = imputer.transform(test)



# Repeat with the scaler

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



print('Training data shape: ', train.shape)

print('Testing data shape: ', test.shape)
# Train on the training data

random_forest.fit(train, train_labels)



# Extract feature importances

feature_importance_values = random_forest.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})



# Make predictions on the test data

predictions = random_forest.predict_proba(test)[:, 1]
predictions
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

    

    # Sort features according to importance

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
# Make a submission dataframe

submit = test_df[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest.csv', index = False)
feature_importances_sorted = plot_feature_importances(feature_importances)
feature_importances
submit.head()
poly_features_names = list(app_train_poly.columns)



# Impute the polynomial features

imputer = Imputer(strategy = 'median')



poly_features = imputer.fit_transform(app_train_poly)

poly_features_test = imputer.transform(app_test_poly)



# Scale the polynomial features

scaler = MinMaxScaler(feature_range = (0, 1))



poly_features = scaler.fit_transform(poly_features)

poly_features_test = scaler.transform(poly_features_test)



random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# Train on the training data

random_forest_poly.fit(poly_features, train_labels)



# Make predictions on the test data

predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]
# Make a submission dataframe

submit = test_df[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_engineered.csv', index = False)
def agg_numeric(df, group_var, df_name):

    """Aggregates the numeric values in a dataframe. This can

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

    # Remove id variables other than grouping variable

    for col in df:

        if col != group_var and 'SK_ID' in col:

            df = df.drop(columns = col)

            

    group_ids = df[group_var]

    numeric_df = df.select_dtypes('number')

    numeric_df[group_var] = group_ids



    # Group by the specified variable and calculate the statistics

    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()



    # Need to create new column names

    columns = [group_var]



    # Iterate through the variables names

    for var in agg.columns.levels[0]:

        # Skip the grouping variable

        if var != group_var:

            # Iterate through the stat names

            for stat in agg.columns.levels[1][:-1]:

                # Make a new column name for the variable and stat

                columns.append('%s_%s_%s' % (df_name, var, stat))



    agg.columns = columns

    return agg
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

    

    # Select the categorical columns

    categorical = pd.get_dummies(df.select_dtypes('object'))



    # Make sure to put the identifying id on the column

    categorical[group_var] = df[group_var]



    # Groupby the group var and calculate the sum and mean

    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    

    column_names = []

    

    # Iterate through the columns in level 0

    for var in categorical.columns.levels[0]:

        # Iterate through the stats in level 1

        for stat in ['count', 'count_norm']:

            # Make a new column name

            column_names.append('%s_%s_%s' % (df_name, var, stat))

    

    categorical.columns = column_names

    

    return categorical
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')

bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')

bureau_counts.head()
bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')

bureau_agg.head()
bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

bureau_balance_counts.head()
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

bureau_balance_agg.head()
# Dataframe grouped by the loan

bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')



# Merge to include the SK_ID_CURR

bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')



# Aggregate the stats for each client

bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
original_features = list(train.columns)

print('Original Number of Features: ', len(original_features))
# Merge with the value counts of bureau

train = train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')



# Merge with the stats of bureau

train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')



# Merge with the monthly information grouped by client

train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
new_features = list(train.columns)

print('Number of features using previous loans from other institutions data: ', len(new_features))
missing_train = missing_values_table(train)

missing_train.head(10)
missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])

len(missing_train_vars)
# Read in the test dataframe

test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')



# Merge with the value counts of bureau

test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')



# Merge with the stats of bureau

test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')



# Merge with the value counts of bureau balance

test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
print('Shape of Testing Data: ', test.shape)
train_labels = train['TARGET']



# Align the dataframes, this will remove the 'TARGET' column

train, test = train.align(test, join = 'inner', axis = 1)



train['TARGET'] = train_labels
print('Training Data Shape: ', train.shape)

print('Testing Data Shape: ', test.shape)
missing_test = missing_values_table(test)

missing_test.head(10)
train.to_csv('train_bureau_raw.csv', index = False)

test.to_csv('test_bureau_raw.csv', index = False)
# Calculate all correlations in dataframe

corrs = train.corr()



corrs = corrs.sort_values('TARGET', ascending = False)



# Ten most positive correlations

pd.DataFrame(corrs['TARGET'].head(10))
# 0.068927

# Ten most negative correlations

pd.DataFrame(corrs['TARGET'].dropna().tail(10))
# Set the threshold

threshold = 0.8



# Empty dictionary to hold correlated variables

above_threshold_vars = {}



# For each column, record the variables that are above the threshold

for col in corrs:

    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
# Track columns to remove and columns already examined

cols_to_remove = []

cols_seen = []

cols_to_remove_pair = []



# Iterate through columns and correlated columns

for key, value in above_threshold_vars.items():

    # Keep track of columns already examined

    cols_seen.append(key)

    for x in value:

        if x == key:

            next

        else:

            # Only want to remove one in a pair

            if x not in cols_seen:

                cols_to_remove.append(x)

                cols_to_remove_pair.append(key)

            

cols_to_remove = list(set(cols_to_remove))

print('Number of columns to remove: ', len(cols_to_remove))
train_corrs_removed = train.drop(columns = cols_to_remove)

test_corrs_removed = test.drop(columns = cols_to_remove)



print('Training Corrs Removed Shape: ', train_corrs_removed.shape)

print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)
train_corrs_removed.to_csv('train_bureau_corrs_removed.csv', index = False)

test_corrs_removed.to_csv('test_bureau_corrs_removed.csv', index = False)

import lightgbm as lgb



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder



import gc



import matplotlib.pyplot as plt
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

submission_raw, fi_raw, metrics_raw = model(train, test)
metrics_raw
fi_raw_sorted = plot_feature_importances(fi_raw)
submission_raw.to_csv('test_one.csv', index = False)