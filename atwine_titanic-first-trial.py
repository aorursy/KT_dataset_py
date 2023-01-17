#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

%matplotlib inline
#let us import the data we have from the competition
test = pd.read_csv('../input/test.csv')
train =  pd.read_csv('../input/train.csv')
gendr = pd.read_csv('../input/gender_submission.csv')
#what does our dataset look like
train.shape
#let us see the kind of dat types we are workng with
train.info()
train.head()
test.info()
#lets first drom the survived column in the train dataset and then merge the two of them
survived = train['Survived']
train.drop(columns=('Survived'), inplace=True)
train.shape
#first we create two differentiating columns in the two data frames
train['Tag']='train'
test['Tag']= 'test'
#we now concetenate the fromes
frames = [train, test]
df = pd.concat(frames)
#now we have the two data frames in one it is easier to apply the innitial exploration functions
df.head()
#there are some values that may not be necessary while we are doing our analysis such as name
drop_columns = ['PassengerId','Name','Ticket']
df.drop(columns=drop_columns, inplace=True)
#how many null values do we have
#4 seem like a few
df.isnull().any().sum()
df.info()
df.head()
#now to separate the two data sets based on the tag columns
test_df = df[df['Tag']== 'test']
train_df = df[df['Tag']== 'train']
train_df.drop(columns=('Tag'), inplace= True)
test_df.drop(columns=('Tag'), inplace= True)
#these are the numeric data frames we have formed and are easy to look at
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
train_df.head()
#one hot encoding adds variables that we did not create so we have to align the data frames
train_df,test_df = train_df.align(test_df, join= 'inner', axis=1)
print('The remaining features', train_df.shape)
print('The remaining test features', test_df.shape)
#now lets return the survived to the training data set
train_df['Survived'] = survived
train_df.shape
#replace the NaN with 0 so that i can use the mean values to impute
train_df['Age']= train_df['Age'].replace(0, np.nan)
train_df.fillna(train_df.mean(), inplace=True)

test_df['Age']= test_df['Age'].replace(0, np.nan)
test_df.fillna(test_df.mean(), inplace=True)
#lets have a loot at the survived variable
train_df['Survived'].value_counts()
#lets display it
train_df['Survived'].astype(int).hist()
#lets see if we have any missing values
# we have none of those
train_df.isnull().any().sum()
#column types
train_df.dtypes.value_counts()
#what is the correlation of the variables to the target variable?
Corr = train_df.corr()['Survived'].sort_values()

#Print them
print('Most positive correlations', Corr.tail(10))
print('-'*20)
print('Most negative correlations', Corr.head(10))
#we need to group by the client id therefore we need to add it back inthe df
train_df['ID'] = train['PassengerId']
train_df.head()
#in order to create features we remove survived first
train_df.drop(columns= ('Survived'), inplace= True)
train_df.head()
#we are going to make new features by the aggregation functions those arefunctions like mean and others
train_agg = train_df.groupby('ID',as_index= False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
train_agg.head()
# List of column names
columns = ['ID']

# Iterate through the variables names
for var in train_agg.columns.levels[0]:
    # Skip the id name
    if var != 'ID':
        
        # Iterate through the stat names
        for stat in train_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('train_%s_%s' % (var, stat))
train_agg.columns = columns
train_agg.head()
#we have now to merge the train_df with the train agg
train_df = train_df.merge(train_agg, on = 'ID', how = 'left')

# List of new correlations
new_corrs = []

# Iterate through the columns 
for col in columns:
    # Calculate correlation with the target
    corr = train_df['ID'].corr(train_df[col])
    
    # Append the list as a tuple

    new_corrs.append((col, corr))

# Sort the correlations by the absolute value
# Make sure to reverse to put the largest values at the front of list
new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
new_corrs[:15]
#we now do the same thing for the test data
#we need to group by the client id
test_df['ID'] = train['PassengerId']
#we are going to make new features by the aggregation functions those arefunctions like mean and others
test_agg = test_df.groupby('ID',as_index= False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
test_agg.head()
# List of column names
columns2 = ['ID']

# Iterate through the variables names
for var2 in test_agg.columns.levels[0]:
    # Skip the id name
    if var2 != 'ID':
        
        # Iterate through the stat names
        for stat in test_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns2.append('test_%s_%s' % (var2, stat))
test_agg.columns = columns2
test_agg.head()
#we have now to merge the train_df with the train agg
test_df = test_df.merge(test_agg, on = 'ID', how = 'left')
train_df.shape , test_df.shape
#because the data was made into two data frames now we combine them back to get the full number of peple on the boat
add = [train_df,test_df]
test_fin = pd.concat(add)
#returning the survived column to train
train_df['Survived'] = survived
train_df.shape , test_df.shape
# train_df.to_csv('train_df.csv')
# test_df.to_csv('test_df.csv')

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
    train_ids = features['ID']
    test_ids = test_features['ID']
    
    # Extract the labels for training
    labels = features['Survived']
    
    # Remove the ids and target
    features = features.drop(columns = ['ID', 'Survived'])
    test_features = test_features.drop(columns = ['ID'])
    
    
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
    submission = pd.DataFrame({'ID': test_ids, 'Survived': test_predictions})
    
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
submission, correlations, metrics = model(train_df,test_fin)
metrics
#lets plot the feature importances
corr = plot_feature_importances(correlations)
submission['Survived'] = submission['Survived'].round(0)
# submission.to_csv('Titanic_Submission2.csv')
