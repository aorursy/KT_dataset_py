## Parts of the code is taken from: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
import os
import numpy as np 
import pandas as pd
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
from sklearn.impute import SimpleImputer as Imputer

random_state = 2020
dataset = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

def get_clean_data(features):
    labels = features['TARGET']    
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    features = pd.get_dummies(features)      
    return features, labels
    
def train_model(features, server_features, valid_size):
    features, labels = get_clean_data(features)    
    features_df, _ = features.align(server_features, join = 'inner', axis = 1)
    features = np.array(features_df)
    train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size=valid_size, random_state=50)
    imputer = Imputer(missing_values=np.nan, strategy = 'median')
    imputer.fit(train_features)    
    train_features = imputer.transform(train_features)
    valid_features = imputer.transform(valid_features)
    print('Training Data Shape: ', train_features.shape)
    print('Validation Data Shape: ', valid_features.shape)
    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                               class_weight = 'balanced', learning_rate = 0.05, 
                               reg_alpha = 0.1, reg_lambda = 0.1, 
                               subsample = 0.8, n_jobs = -1, random_state = random_state)
    model.fit(train_features, train_labels, eval_metric = 'auc',
              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
              eval_names = ['valid', 'train'], categorical_feature = 'auto',
              early_stopping_rounds = 200, verbose = 100)
    best_iteration = model.best_iteration_
    valid_preds = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
    valid_auc = roc_auc_score(valid_labels, valid_preds)    
    return valid_auc, model, best_iteration, features_df
print('Dataset shape: ', dataset.shape)
dataset_Alice_Bob, dataset_Server = train_test_split(dataset, test_size=0.33, random_state=random_state)
print("Alice and Bob shares ", dataset_Alice_Bob.shape)
print("Server's share ", dataset_Server.shape)
dataset_Alice, dataset_Bob = train_test_split(dataset_Alice_Bob, test_size=0.5, random_state=random_state)
print("Alice's share: ", dataset_Alice.shape)
print("Bob's share ", dataset_Bob.shape)
## Non-federated : Alice and Bob
server_features, server_labels = get_clean_data(dataset_Server)
valid_auc, trained_model, best_iteration, features = train_model(dataset_Alice_Bob, server_features, valid_size=.1)
server_features, _ = server_features.align(features, join = 'inner', axis = 1)
imputer = Imputer(missing_values=np.nan, strategy = 'median')
imputer.fit(np.array(server_features))
server_features_imputed = imputer.transform(np.array(server_features))
nf_server_preds = trained_model.predict_proba(server_features_imputed, num_iteration = best_iteration)[:, 1]
nf_server_auc = roc_auc_score(server_labels, nf_server_preds)

## Only Alice
server_features, server_labels = get_clean_data(dataset_Server)
valid_auc, trained_model, best_iteration, features = train_model(dataset_Alice, server_features, valid_size=.1)
server_features, _ = server_features.align(features, join = 'inner', axis = 1)
imputer = Imputer(missing_values=np.nan, strategy = 'median')
imputer.fit(np.array(server_features))
server_features_imputed = imputer.transform(np.array(server_features))
a_server_preds = trained_model.predict_proba(server_features_imputed, num_iteration = best_iteration)[:, 1]
a_server_auc = roc_auc_score(server_labels, a_server_preds)

## Only Bob
server_features, server_labels = get_clean_data(dataset_Server)
valid_auc, trained_model, best_iteration, features = train_model(dataset_Bob, server_features, valid_size=.1)
server_features, _ = server_features.align(features, join = 'inner', axis = 1)
imputer = Imputer(missing_values=np.nan, strategy = 'median')
imputer.fit(np.array(server_features))
server_features_imputed = imputer.transform(np.array(server_features))
b_server_preds = trained_model.predict_proba(server_features_imputed, num_iteration = best_iteration)[:, 1]
b_server_auc = roc_auc_score(server_labels, b_server_preds)

## Combined: (Alice and Bob Only send their models, not their data, to the Server) 
federated_preds = (a_server_preds+b_server_preds)/2
f_server_auc = roc_auc_score(server_labels, federated_preds)

## Results
print("\n** AUC Results **")
print('Raw Data Alice & Bob:   ', np.round(nf_server_auc,5))
print('Model Only Alice:\t', np.round(a_server_auc,5))
print('Model Only Bob:\t\t', np.round(b_server_auc,5))
print('Combined Alice & Bob:  ', np.round(f_server_auc,5))