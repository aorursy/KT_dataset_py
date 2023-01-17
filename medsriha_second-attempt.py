# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pandas as pd
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
app_train.head()
def missing_values(df):
    miss = df.isnull().sum()
    a = (miss/len(df)) * 100
    frame = pd.DataFrame({'Features':df.columns, 'Missing_percentage':a,'Total_number': miss}).reset_index()
    frame = frame.drop('index',axis=1)
    d =frame[frame['Total_number'] > 0].sort_values(by='Missing_percentage',ascending=False).round(1)
    print('Total Columns Number: %s' %len(df.columns))
    print('Total Columns having missing values: %s' %len(d))
    return d
    
    
missing_values(app_train).head(10)
app_train['TARGET'].hist()
app_train['TARGET'].value_counts()
print(app_train.columns)
from sklearn import preprocessing
def label_encoding(df):
    enc = preprocessing.LabelEncoder()
    count = 0
    for col in df:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                enc.fit(df[col])
                df[col] = enc.transform(df[col])
                count+=1
                
                
    return df
app_train = label_encoding(app_train)
app_test = label_encoding(app_test)
def hot_encoding(df):
    return pd.get_dummies(df)
app_train = hot_encoding(app_train)
app_test = hot_encoding(app_test)
print(app_train.shape)
print(app_test.shape)
def align_data(df_train, df_test):
    labels = df_train['TARGET']
    a,b = df_train.align(df_test, join='inner', axis=1)
    a['TARGET'] = labels
    return a,b
    
app_train,app_test = align_data(app_train,app_test)
print(app_train.shape)
print(app_test.shape)
(app_train['DAYS_BIRTH'] / -360).describe()
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
print('There are %d anomalies in the train data out of %d entries' % (app_train["DAYS_EMPLOYED_ANOM"].sum(), len(app_train)))
# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
correlations = app_train.corr()['TARGET'].sort_values()
print('Negatively Correlated Variables',correlations.head(20))
print('Positively Correlated Variables',correlations.tail(20))
import matplotlib.pyplot as plt
plt.hist(app_train[app_train['TARGET'] == 1]['DAYS_BIRTH'] / -365,edgecolor='k', bins=30)
plt.title('Age of Client (Target = 1)'); 
plt.xlabel('Age'); 
plt.ylabel('Count');
plt.hist(app_train[app_train['TARGET'] == 0]['DAYS_BIRTH'] / -365,edgecolor='k', bins=30)
plt.title('Age of Client (Target  = 0)'); 
plt.xlabel('Age'); 
plt.ylabel('Count');
seg = app_train[['TARGET','DAYS_BIRTH']]
seg['YEAR_BIRTH'] = seg['DAYS_BIRTH'] / -360

seg['YEAR_BIN'] = pd.cut(seg['YEAR_BIRTH'], bins = np.linspace(20, 70, num = 11))
seg.head()
seg = seg.groupby('YEAR_BIN').mean()
seg.head()
plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(seg.index.astype(str), 100 * seg['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
#Impute missing values
imputer = preprocessing.Imputer(strategy='median')
poly_features_train = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
poly_target = poly_features_train['TARGET']

poly_features_train = poly_features_train.drop(columns = ['TARGET'])

# Need to impute missing values
poly_features_train = imputer.fit_transform(poly_features_train)
poly_features_test = imputer.transform(poly_features_test)
poly_transformer = preprocessing.PolynomialFeatures(degree=3)
poly_transformer.fit(poly_features_train)
poly_features_train = poly_transformer.transform(poly_features_train)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features_train.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
#Merge the new features into the train and test dataframe
poly_train = pd.DataFrame(poly_features_train, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_test = pd.DataFrame(poly_features_test, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_train.head()
poly_train = poly_train.drop('1',axis=1)
poly_test = poly_test.drop('1',axis=1)
poly_train['SK_ID_CURR'] = app_train['SK_ID_CURR']
poly_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
print('Shape poly train',poly_train.shape)
print('Shape poly test',poly_test.shape)
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
app_train_domain_poly = app_train_domain.merge(poly_train,how='left',on='SK_ID_CURR')
app_test_domain_poly = app_test_domain.merge(poly_test,how='left',on='SK_ID_CURR')
print('Shape poly train',app_train_domain_poly.shape)
print('Shape poly test',app_test_domain_poly.shape)
corr = app_train_domain_poly.corr()['TARGET'].sort_values()
imputer_ = preprocessing.Imputer(strategy='median')
scale_ = preprocessing.MinMaxScaler(feature_range=(0,1))

#align the two dataset
train, test = align_data(app_train_domain_poly, app_test_domain_poly)
target = train['TARGET']
train = train.drop('TARGET',axis=1)
print('Shape poly train',train.shape)
print('Shape poly test',test.shape)
imputer_.fit(train)
train = imputer_.transform(train)
test = imputer_.transform(test)
print('Shape poly train',train.shape)
print('Shape poly test',test.shape)
scale_.fit(train)
train = scale_.transform(train)
test = scale_.transform(test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.0001)
lr.fit(train,target)
prediction_two = lr.predict_proba(test)[:,1]
submission_two = pd.DataFrame({'SK_ID_CURR':app_test_domain_poly['SK_ID_CURR'],'TARGET':prediction_two})
def plot_feature_importances(data):
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
    df = data.copy()
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
submission, fi, metrics = model(app_train_domain_poly, app_test_domain_poly)
metrics
plot_feature_importances(fi)
submission.to_csv('submissionfive.csv',index=False)
