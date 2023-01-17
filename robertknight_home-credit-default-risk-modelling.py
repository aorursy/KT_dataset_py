# All Package Imports
import pandas as pd
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time

# import sklearn packages for modelling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
# List files available
print(os.listdir("../input"))
# load and print head of the a_train dataset
a_train = pd.read_csv(r'../input/application_train.csv')
a_train.head()

# keep original version of a_train for exploratory visualisations
original_a_train = a_train.copy(deep = True)
# Train matrix shape
a_train.shape
# load and print head of the a_train dataset
a_test = pd.read_csv(r'../input/application_test.csv')
a_test.head()
# Test matrix shape
a_test.shape
# function to find missing values in datafame so we can reuse if we look at other data sources
# including proper docstring
def missing_values_table(df):
        
        # sum of missing values
        mis_val = df.isnull().sum()
        
        # % of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # concat table
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% Missing Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0
                                                             ].sort_values('% Missing Values', ascending=False).round(1)
        
        # Print some summary information
        print ("The dataframe has {} columns.\n".format(str(df.shape[1])),      
                "There are {} columns that have missing values.\n".format(str(mis_val_table_ren_columns.shape[0])),
               "There are {} columns that have no missing values".format(int(df.shape[1]) - int(mis_val_table_ren_columns.shape[0])) )
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    

missing_values_table(a_train)
# Columns Data Types
train_dtypes = pd.DataFrame(a_train.dtypes.value_counts()).reset_index()
train_dtypes.columns = ['dtypes', 'column count']

train_dtypes
# create dict object from columns and datatypes
columns = a_train.columns.to_series().groupby(a_train.dtypes).groups
for key in columns.keys():
    print('\nData Type {} Columns:'.format(key))
    pprint(list(columns[key]))
# desribe the categorical data
a_train.loc[:, a_train.dtypes == np.object].describe()
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in a_train:
    if a_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(a_train[col].unique())) <= 2:
            print("{} was encoded".format(col))
            # Train on the training data
            le.fit(a_train[col])
            # Transform both training and testing data
            a_train[col] = le.transform(a_train[col])
            a_test[col] = le.transform(a_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables
a_train = pd.get_dummies(a_train)
a_test = pd.get_dummies(a_test)

print('Training Features shape: ', a_train.shape)
print('Testing Features shape: ', a_test.shape)
train_labels = a_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
a_train, a_test = a_train.align(a_test, join = 'inner', axis = 1)

# Add the target back in
a_train['TARGET'] = train_labels
print('Aligned Training Features shape: ', a_train.shape)
print('Aligned Testing Features shape: ', a_test.shape)
# Plot TARGET distribution
a_train['TARGET'].value_counts()
a_train['TARGET'].value_counts().plot(kind='bar', figsize=(10,5), color = ['grey', 'cornflowerblue'])
plt.xlabel('Target Class')
plt.ylabel('Count') 
plt.show()
# Create function for plotting kde with scale reversing
def plot_kde(df, var, reverse_scale = False):
    
    plt.figure(figsize = (12, 6))
    
    if reverse_scale == True:
        r = -1
    else:
        r = 1
    
    # KDE plot of loans that were repaid on time
    sns.kdeplot(df.loc[df['TARGET'] == 0, var] * r, label = 'target: negative class', color = 'grey', shade = True)

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(df.loc[df['TARGET'] == 1, var] * r, label = 'target: positive class', color = 'cornflowerblue', shade = True)

    # Labeling of plot
    plt.xlabel('{}'.format(var)); plt.ylabel('Density'); plt.title('KDE for {}'.format(var));
    plt.show()
    plt.close()

# plot age kde plot
plot_kde(a_train,'DAYS_BIRTH', True)
# iterate over float variables and plot KDE
for column in original_a_train.loc[:, (original_a_train.dtypes == np.float64)].columns.values:
    # do not plot target 
    if column != 'TARGET':
        # reverse axis if values are negative
        if (original_a_train[column].median() < 0):
            plot_kde(a_train,column, reverse_scale = True)
        else:
            plot_kde(a_train,column)
def analyse_outliers(df, column):
    
    # Print Summary Statistics
    print('Summary Statistics:\n')
    print(df[column].describe())

    # find mean and std
    outlier_df = df[column]
    std = outlier_df.std()
    print('\nStandard Deviation: ', std)
    mean =  outlier_df.mean()
    print('Mean: ', mean)

    # how many std is the max
    max_outlier = int((outlier_df.max() - mean) / std)

    # separate outliers over 2 std from mean
    outliers_l = outlier_df[(outlier_df < mean - 2 * std)]
    outliers_h = outlier_df[(outlier_df > mean + 2 * std)]
    print('\nThere are {} low end outliers in the {} dataset'.format(len(outliers_l), column ))    
    print('There are {} high end outliers in the {} dataset'.format(len(outliers_h), column ))
    print('The max value is {} standard deviations from the mean'.format(max_outlier))
    
    return mean, std

income_mean, income_std = analyse_outliers(a_train, 'AMT_INCOME_TOTAL')   
# define function for plotting categorical bar charts for remaining variables
def categorical_plot(df, variable):
    
    plt.figure(figsize = (11, 5))
    
    df_high = df[df['TARGET'] == 1].groupby(variable)['TARGET'].agg('count')
    df_var = df.groupby(variable)['TARGET'].agg('count')
    categorical = df_high.divide(df_var, fill_value = 0) * 100

    # Convert back to df
    df_categorical = categorical.to_frame().reset_index().sort_values('TARGET', ascending = True)

    # Create plot in Plotly for interactive visualisation (with some Starling colours)
    ax = df_categorical.plot(x = variable, y = 'TARGET', kind = 'barh', figsize=(10,10), color = 'cornflowerblue')
    ax.set_xlabel('Target: Positive %')
    ax.set_ylabel(variable)
    plt.title('% postive plot for {}'.format(variable.lower()));
    plt.show()
    plt.close()

# iterate over np.object columns and plot
for column in original_a_train.loc[:, original_a_train.dtypes == np.object].columns.values:
    categorical_plot(original_a_train, column)
# Find correlations with the target(takes a while due to many features)
correlations = a_train.corr()['TARGET'].sort_values()
# print ordered list of correlations
print('Most Positive Correlations:\n')
print(correlations.sort_values(ascending = False).head(16))
print('\nMost Negative Correlations:\n')
print(correlations.head(15))
# create X_train, y_train
X_train = a_train.drop('TARGET', axis = 1)
y_train = a_train['TARGET']
X_test = a_test

X_train = X_train.drop('SK_ID_CURR', axis = 1)
X_test = X_test.drop('SK_ID_CURR', axis = 1)

# Feature names
features = list(X_train.columns)
# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(X_train)

# Transform both training and testing data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Repeat with the scaler
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)

print('Training data shape: ', X_train.shape)
print('Testing data shape: ', X_test.shape)
# Using Cross Validation to find a good model
num_folds = 5
seed = 1
scoring = 'roc_auc'
models = []

# Typical Classifiers
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))

# iterate over models and print cross val scores
results = []
names = []
print('Please wait while models train..')
for name, model in models:
    
    # start timer
    start = time.time()
    
    # Cross Validation
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    # stop timing
    end = time.time()
    time_run = (end - start)/60
    output = "{}--> auroc: {}   (Training Time: {}mins)".format(name, cv_results.mean(), time_run)
    
    print(output)
# Train LR Model
RF = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
RF.fit(X_train,y_train)

# Extract feature importances
feature_importance_values = RF.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': feature_importance_values})

# Make predictions on the test data
predictions = RF.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = a_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline.csv', index = False)

# Function to plot feature importance
def plot_feature_importance(df):

    # Normalize the feature importances to add up to one
    df['Importance_normalized'] = df['Importance'] / df['Importance'].sum()
    df = df.sort_values('Importance_normalized', ascending = True).tail(20)

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 16))

    ax = df.plot(x = 'Feature' , y = 'Importance_normalized', kind = 'barh', figsize=(10,10), color = 'blue')
    
    # Plot labeling
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()
    
    # return top 20 features
    return(df['Feature'])

top20 = plot_feature_importance(feature_importances)
# import bureau data
bureau = pd.read_csv(r'../input/bureau.csv')
bureau.head()
# create feature dataframe
bureau_agg = bureau['SK_ID_CURR'].unique()
bureau_agg = pd.DataFrame(bureau_agg, columns = ['SK_ID_CURR'])
# previous loan count
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()

bureau_agg = bureau_agg.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
# active loan count
active_loan_counts = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'active_loan_counts'})
active_loan_counts.head()

# join new features
bureau_agg = bureau_agg.merge(active_loan_counts, on = 'SK_ID_CURR', how = 'left')

# fill na
bureau_agg = bureau_agg.fillna(0)
# join additional features onto train and test
a_train_features = a_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
a_test_features = a_test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
a_train_features = a_train_features.fillna(0)
a_test_features = a_test_features.fillna(0)
# plot kde of new features
plot_kde(a_train_features, 'previous_loan_counts')
plot_kde(a_train_features, 'active_loan_counts')
print('Training data shape: ', a_train_features.shape)
print('Testing data shape: ', a_test_features.shape)
# create X_train, y_train
X_train = a_train_features.drop('TARGET', axis = 1)
y_train = a_train_features['TARGET']
X_test = a_test_features

X_train = X_train.drop('SK_ID_CURR', axis = 1)
X_test = X_test.drop('SK_ID_CURR', axis = 1)

# Feature names
features = list(X_train.columns)

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(X_train)

# Transform both training and testing data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Repeat with the scaler
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)

print('Training data shape: ', X_train.shape)
print('Testing data shape: ', X_test.shape)

# Train LR Model
RF = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
RF.fit(X_train,y_train)

# Extract feature importances
feature_importance_values = RF.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': feature_importance_values})

# Make predictions on the test data
predictions = RF.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = a_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_features.csv', index = False)

top20 = plot_feature_importance(feature_importances)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

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
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
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
submission, fi, metrics = model(a_train_features, a_test_features)
print('Baseline metrics')
print(metrics)
def plot_feature_importances(df, num_bars = 15):
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
    ax.barh(list(reversed(list(df.index[:num_bars]))), 
            df['importance_normalized'].head(num_bars), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:num_bars]))))
    ax.set_yticklabels(df['feature'].head(num_bars))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
fi_sorted = plot_feature_importances(fi, 25)
submission.to_csv('baseline_lgb_features.csv', index = False)