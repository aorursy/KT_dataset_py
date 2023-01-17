import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import gc
def model(features, test_features, n_folds=5):
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    labels = features['TARGET'].values
    

    ratio = (labels == 0).sum()/ (labels == 1).sum()
    
    #Remove ids and target
    features.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    test_features.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    #features = pd.get_dummies(features)
    #test_features = pd.get_dummies(test_features)
    #features, test_features = features.align(test_features, join='inner', axis=1)
    #Extract feature names
    feature_names = features.columns.tolist()
    
    cat_indices= []
    for i, col in enumerate(feature_names):
        if features[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            test_features[col] = le.transform(test_features[col].astype(str))
            cat_indices.append(i)
        
    
    print("Shape of training data: {}".format(features.shape))
    print("Shape of test data: {}".format(test_features.shape))
    
    features = features.values
    test_features = test_features.values
    
    #Create a stratified Kfold object
    k_fold = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 1)
    #Empy arrat for test and out of fold predictions
    test_predictions = np.zeros(len(test_features))
    oof_predictions = np.zeros(len(features))
    
    feature_importance_values = np.zeros(len(feature_names))
    
    #List for recording training and validation scores
    train_scores = []
    valid_scores = []
    
  
    
    
    for train_indices, val_indices in k_fold.split(features, labels):
        train_features, train_labels = features[train_indices], labels[train_indices]
        val_features, val_labels = features[val_indices], labels[val_indices]
        
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=64, max_depth=-1, learning_rate=0.01, n_estimators=10000, 
                                 subsample_for_bin=200000, objective='binary', min_split_gain=0.0, min_child_weight=0.001, 
                                 min_child_samples=20, subsample=0.7, subsample_freq=0, colsample_bytree=1.0, 
                                 reg_alpha=0.0, reg_lambda=0.0, n_jobs=-1, silent=True, scale_pos_weight=ratio, random_state = 50)
        
        clf.fit(train_features, train_labels, eval_set=[(train_features, train_labels), (val_features, val_labels)], eval_metric='auc',
                eval_names = ['train', 'valid'], verbose=100, early_stopping_rounds=50,feature_name='auto', categorical_feature= cat_indices)
        best_iteration = clf.best_iteration_
        
        test_predictions += clf.predict_proba(test_features, num_iteration=best_iteration)[:, 1]/k_fold.n_splits
        oof_predictions[val_indices] = clf.predict_proba(val_features, num_iteration=best_iteration)[:, 1]/k_fold.n_splits
        
        valid_scores.append(clf.best_score_['valid']['auc'])
        train_scores.append(clf.best_score_['train']['auc'])
        feature_importance_values+=clf.feature_importances_/k_fold.n_splits
        gc.enable()
        del clf, train_features, val_features, train_labels, val_labels 
        gc.collect()
    
    # Overall validation score
    validation_auc = roc_auc_score(labels, oof_predictions)
    valid_scores.append(validation_auc)
    train_scores.append(np.mean(train_scores))
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    #Make feature importance dataframe
    feature_importances = pd.DataFrame({'features': feature_names, 'importance': feature_importance_values})
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
      
    
    return submission, feature_importances, metrics
#Functio to calculate missing values for all the features
def missing_values(df):
    "'Function to get the column-wise missing values in a dataframe'"
    col_missing_values = 100*df.isnull().sum()/len(df)
    df_missing_values = pd.DataFrame({'feature': col_missing_values.index, 'missing values %': col_missing_values.values})
    df_missing_values = df_missing_values.sort_values('missing values %', ascending= False)
    df_missing_values = df_missing_values[df_missing_values['missing values %'] != 0]
    df_missing_values.reset_index(drop=True, inplace=True)
    print("There are {} features with missing values.".format(df_missing_values.shape[0]))
    
    return df_missing_values
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')

print("Shape of training data: {}".format(app_train.shape))
print("Shape of test data: {}".format(app_test.shape))
#Check missing for values
app_train_missing_values = missing_values(app_train)
#List of features having more than 40% of missing values
drop_columns = app_train_missing_values[app_train_missing_values['missing values %'] >=40]['feature'].tolist()
drop_columns.remove('EXT_SOURCE_1')
'EXT_SOURCE_1' in drop_columns

#Remove all the features with greater than 40% missing values
app_train = app_train.drop(drop_columns, axis=1)
app_test = app_test.drop(drop_columns, axis=1)

print(app_train.shape)
print(app_test.shape)
#Replace the outliar in DAYS_EMPLOYED feature with null values
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
#Remove errornous values from CODE_GENDER
app_train = app_train[app_train['CODE_GENDER'] != 'XNA']
app_train['CODE_GENDER'].value_counts(dropna=False)
print("Shape of training data: {}".format(app_train.shape))
print("Shape of test data: {}".format(app_test.shape))
submission, feature_importances, metrics = model(app_train, app_test, n_folds=5)
submission.to_csv("lightgbm_baseline.csv", index=False)
print(metrics)
top30 = feature_importances.sort_values(by='importance', ascending=False).head(30)
plt.figure(figsize=(10, 8))
sns.barplot(x=top30['importance'], y=top30['features'])
plt.show()