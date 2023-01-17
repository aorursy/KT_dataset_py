# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# python 3 ### Input data files are available in the "../input/" directory.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import seaborn as sns
import warnings
import matplotlib
import matplotlib.pyplot as plt # for plotting
%matplotlib inline
color = sns.color_palette()
warnings.filterwarnings('ignore') # Suppress warnings 
from sklearn.metrics import roc_curve ## for roc curve
from sklearn.metrics import roc_auc_score
app_train = pd.read_csv("../input/application_train.csv")
app_test = pd.read_csv("../input/application_test.csv")
data_train = app_train.copy()
data_test = app_test.copy()
data_train.describe()
data_test.describe()
data_train.shape
data_test.shape
# importing the datasets into Pandas dataframes
bureau_balance = pd.read_csv('../input//bureau_balance.csv')
bureau = pd.read_csv('../input//bureau.csv')
# left joining the dataset on='SK_ID_BUREAU'(left=bureau, right=bureau_balance)
df_bureau_joined = bureau.merge(bureau_balance, 
                                on='SK_ID_BUREAU', 
                                how='left')
df_bureau_joined.shape
target = data_train['TARGET']
target.value_counts()
def plot_count_distribution(df,col_name):
    #define order of bars
    order = list(df[col_name].value_counts().index)
    plt.figure(figsize=(5,4))
    ax = sns.countplot(x=col_name,data=df)
    plt.title('Target Variable Distribution (Training Dataset)')
    plt.xlabel('Target')
    plt.ylabel('Counts')

    #include count labels on top of each bar
    for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+10))
    plt.show()

plot_count_distribution(data_train,'TARGET')
# Count Plot (a.k.a. Bar Plot)
plt.figure(figsize=(10,5))
plt.ylabel('Count')
plt.title('Contract Types by Target Value')
bar_plot = sns.countplot(x='NAME_CONTRACT_TYPE', hue='TARGET', data=data_train)
for p in bar_plot.patches:
        bar_plot.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+10))
plt.show()
plt.figure(figsize=(10,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(data_train.loc[data_train['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label = 'Repaid Loan')
sns.kdeplot(data_train.loc[data_train['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label = 'Not Repaid Loan')
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Age of Client (in Years)');
# Age information into a separate dataframe
age_data = data_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data.loc[:,'DAYS_BIRTH'].copy() / -365
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups.drop(['DAYS_BIRTH'], axis=1, inplace=True)
plt.figure(figsize=(10,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
plt.xticks(rotation = 75)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Client\'s Age Range');
plt.figure(figsize=(14,5))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(data_train.loc[data_train['TARGET'] == 0, 'AMT_CREDIT'], 
            label = 'Repaid Loan')
sns.kdeplot(data_train.loc[data_train['TARGET'] == 1, 'AMT_CREDIT'], 
            label = 'Not Repaid Loan')
plt.xlabel('Amount of Credit')
plt.xticks(np.arange(0, 5000000, 500000))
plt.ylabel('Density')
plt.title('Distribution of Amount of Credit by Target Value');
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
# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(25)

plt.figure(figsize=(10,5))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
x = missing_values['% of Total Values']
x.hist(align='left', bins= 50)
plt.xticks(np.arange(0, 75, 5))
plt.xlabel('% of Missing Values')
plt.yticks(np.arange(0, 12, 2))
plt.ylabel('Count (Features)')
plt.title('Distribution of % of Missing Values in Dataset Features');
plt.show();
app_train.dtypes.value_counts()
# Number of unique classes in each object column
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Find correlations with the target and sort
correlations = data_train.corr()['TARGET'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(15))
print('\nMost Negative Correlations: \n', correlations.head(15))
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in data_train:
    if data_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(data_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(data_train[col])
            # Transform both training and testing data
            data_train[col] = le.transform(data_train[col])
            data_test[col] = le.transform(data_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables
df_train = pd.get_dummies(data_train)
df_test = pd.get_dummies(data_test)

print('Training Features shape for df_train: ', df_train.shape)
print('Testing Features shape for df_test: ', df_test.shape)
print('Training Features shape for data_train: ', data_train.shape)
print('Testing Features shape for data_test: ', data_test.shape)
train_labels = df_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

# Add the target back in
df_train['TARGET'] = train_labels

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)
data_train.dtypes.value_counts()

from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in df_train:
    train = df_train.drop(columns = ['TARGET'])
else:
    train = df_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = df_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(df_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

from sklearn.ensemble import RandomForestClassifier
# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, 
                                       random_state = 50, 
                                       verbose = 1, n_jobs = -1)
from sklearn.model_selection import train_test_split
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.3, stratify =train_labels)
# Train on the training data
random_forest.fit(X_train, y_train)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features,
                                    'importance': feature_importance_values})

# Make predictions on the test data
predictions_val = random_forest.predict_proba(X_test)[:, 1]
#predictions_test = random_forest.predict(test)[:, 1]
predictions_val.shape
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, predictions_val))
from sklearn.metrics import auc
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, predictions_val)
auc_rf = auc(fpr_rf, tpr_rf)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Train on the training data
random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 
                                    'importance': feature_importance_values})

# Make predictions on the test data
predictions_test = random_forest.predict_proba(test)[:, 1]
plot_test = df_test[['SK_ID_CURR','EXT_SOURCE_2','EXT_SOURCE_3']]
plot_test['TARGET'] = predictions_test
#plot_predict_interaction(random_forest, plot_test, "rm", "EX")
# Make a submission dataframe
submit = df_test[['SK_ID_CURR']]
submit['TARGET'] = predictions_test

# Save the submission dataframe
submit.to_csv('random_forest_baseline.csv', index = False)
def plot_feature_importances(df):
    """shows a plot of the 15 most importance features"""
    
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
# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):  
    """
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
submission, fi, metrics = model(df_train, df_test)
print('Baseline metrics')
print(metrics)
fi_sorted = plot_feature_importances(fi)
submission.to_csv('baseline_lgb.csv', index = False)
hyperparameter = {'subsample_for_bin': 220000, 'learning_rate': 0.07016445423929361, 'num_leaves': 86, 'metric': 'auc', 'boosting_type': 'gbdt', 'verbose': 1, 'colsample_bytree': 0.6444444444444444, 'subsample': 0.5303030303030303, 'reg_alpha': 0.9591836734693877, 'min_child_samples': 390, 'is_unbalance': True, 'reg_lambda': 0.673469387755102}
test_ids = df_test['SK_ID_CURR']
train_labels = np.array(df_train['TARGET'].astype(np.int32)).reshape((-1, ))

train_random = df_train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test_random = df_test.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train_random.shape)
print('Testing shape: ', test_random.shape)
train_set = lgb.Dataset(train_random, label = train_labels)

# Cross validation with n_folds and early stopping
cv_results = lgb.cv(hyperparameter,
                    train_set,
                    num_boost_round = 10000, 
                    early_stopping_rounds = 100,
                    nfold = 5)

print('The cross validation score on the full dataset  for Random Search= {:.5f} with std: {:.5f}.'.format(
    cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('Number of estimators = {}.'.format(len(cv_results['auc-mean'])))
model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameter)
model.fit(train_random, train_labels)

preds = model.predict_proba(test_random)[:, 1]

submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
submission.to_csv('submission_random_search.csv', index = False)
hyper_b = {'learning_rate': 0.07218374731817535, 
           'reg_lambda': 0.7364934411848395, 
           'verbose': 1, 'subsample': 0.6195545022366721, 
           'subsample_for_bin': 60000, 'boosting_type': 'dart',
           'is_unbalance': True, 
           'num_leaves': 47, 'colsample_bytree': 0.6001712855022151, 
           'reg_alpha': 0.5969339070590824, 'min_child_samples': 485,
           'metric': 'auc'}

# Cross validation with n_folds and early stopping
cv_results = lgb.cv(hyper_b, train_set,
                    num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = 5)

print('The cross validation score on the full dataset for Bayesian optimization = {:.5f} with std: {:.5f}.'.format(
    cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('Number of estimators = {}.'.format(len(cv_results['auc-mean'])))
model = lgb.LGBMClassifier(n_estimators = 107, **hyper_b)
model.fit(train_random, train_labels)

preds = model.predict_proba(test_random)[:, 1]

submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
submission.to_csv('submission_bayesian_optimization.csv', index = False)
