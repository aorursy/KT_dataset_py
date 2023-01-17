import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
from sklearn.ensemble import RandomForestClassifier

train_app = pd.read_csv('../input/application_train.csv')
test_app = pd.read_csv('../input/application_test.csv')
#label encoding
labelEncoder = LabelEncoder()

for col in train_app:
    if train_app[col].dtype == 'object':
        if len(list(train_app[col].unique())) <= 2:
            labelEncoder.fit(train_app[col])
            train_app[col] = labelEncoder.transform(train_app[col])
            test_app[col] = labelEncoder.transform(test_app[col])
train_app = pd.get_dummies(train_app)
test_app = pd.get_dummies(test_app)
labels_train = train_app['TARGET']
train_app, test_app = train_app.align(test_app, join = 'inner', axis = 1)
train_app['TARGET'] = labels_train
train_app['DAYS_EMPLOYED_ANOM'] = train_app["DAYS_EMPLOYED"] == 365243
train_app['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
test_app['DAYS_EMPLOYED_ANOM'] = test_app["DAYS_EMPLOYED"] == 365243
test_app['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
train_app_domain = train_app.copy()
test_app_domain = test_app.copy()

train_app_domain['CREDIT_INCOME_PERCENT'] = train_app_domain['AMT_CREDIT'] / train_app_domain['AMT_INCOME_TOTAL']
train_app_domain['ANNUITY_INCOME_PERCENT'] = train_app_domain['AMT_ANNUITY'] / train_app_domain['AMT_INCOME_TOTAL']
train_app_domain['CREDIT_TERM'] = train_app_domain['AMT_ANNUITY'] / train_app_domain['AMT_CREDIT']
train_app_domain['DAYS_EMPLOYED_PERCENT'] = train_app_domain['DAYS_EMPLOYED'] / train_app_domain['DAYS_BIRTH']

test_app_domain['CREDIT_INCOME_PERCENT'] = test_app_domain['AMT_CREDIT'] / test_app_domain['AMT_INCOME_TOTAL']
test_app_domain['ANNUITY_INCOME_PERCENT'] = test_app_domain['AMT_ANNUITY'] / test_app_domain['AMT_INCOME_TOTAL']
test_app_domain['CREDIT_TERM'] = test_app_domain['AMT_ANNUITY'] / test_app_domain['AMT_CREDIT']
test_app_domain['DAYS_EMPLOYED_PERCENT'] = test_app_domain['DAYS_EMPLOYED'] / test_app_domain['DAYS_BIRTH']
correlationMatrix = train_app_domain.corr().abs()
correlationMatrix.head()

upperTriangle =  correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k = 1).astype(np.bool))
upperTriangle.head()

columnsToDrop = [column for column in upperTriangle.columns if any(upperTriangle[column] > 0.9)]
len(columnsToDrop)
train_app_domain = train_app_domain.drop(columns = columnsToDrop)
test_app_domain = test_app_domain.drop(columns = columnsToDrop)

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
train_app_domain_backup = train_app_domain.copy()
test_app_domain_backup = test_app_domain.copy()

train_app_domain = train_app_domain.drop(columns  = 'TARGET')
imputer_domain = Imputer(strategy = 'median')

train_app_domain = imputer_domain.fit_transform(train_app_domain)
test_app_domain = imputer_domain.transform(test_app_domain)

scaler = MinMaxScaler(feature_range = (0, 1))

train_app_domain = scaler.fit_transform(train_app_domain)
test_app_domain = scaler.transform(test_app_domain)
 
domain_random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
domain_random_forest.fit(train_app_domain,labels_train)

predict = domain_random_forest.predict_proba(test_app_domain)[:,1]
rf_submission = test_app[['SK_ID_CURR']]
rf_submission['TARGET'] = predict
rf_submission.to_csv('rf_submission_domain.csv', index = False)

train_app_domain = train_app_domain_backup.copy()
test_app_domain = test_app_domain_backup.copy()
def lightgbmmodel(features, test_features, encoding = 'ohe', folds = 5) :
    
    ids_train = features['SK_ID_CURR']
    ids_test = test_features['SK_ID_CURR']
    
    class_labels = features['TARGET']
    
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    #onehotencoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        cat_indices = 'auto'
    names_features = list(features.columns)
    
    features = np.array(features)
    test_features = np.array(test_features)
    
    kfold = KFold (n_splits = folds , shuffle = True, random_state = 50)
    
    fi_values = np.zeros(len(names_features))
    
    predictions_test = np.zeros(test_features.shape[0])
    
    outOfFold = np.zeros(features.shape[0])
    
    scores_valid = []
    scores_train = []
    
    for indices_train, indices_valid in kfold.split(features):
        
        train_features, train_labels = features[indices_train], class_labels[indices_train]
        valid_features, valid_labels = features[indices_valid], class_labels[indices_valid]
        
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        best_iteration = model.best_iteration_
        
        fi_values += model.feature_importances_ / kfold.n_splits
        
        predictions_test += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / kfold.n_splits
        
        outOfFold[indices_valid] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        scores_valid.append(valid_score)
        scores_train.append(train_score)
        
        gc.enable()
        del model, train_features, valid_features
        gc.collect()    
        
    submission = pd.DataFrame({'SK_ID_CURR': ids_test, 'TARGET': predictions_test})
    feature_importances = pd.DataFrame({'feature': names_features, 'importance': fi_values})
    
    valid_auc = roc_auc_score(class_labels, outOfFold)

    scores_valid.append(valid_auc)
    scores_train.append(np.mean(scores_train))

    fold_names = list(range(folds))
    fold_names.append('overall')

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': scores_train,
                            'valid': scores_valid}) 

    return submission, feature_importances, metrics
submission, fi, metrics = lightgbmmodel(train_app, test_app)
print('Baseline metrics')
print(metrics)

train_app_domain['TARGET'] = labels_train

submission_domain, fi_domain, metrics_domain = lightgbmmodel(train_app_domain, test_app_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)
fi_domain.sort_values('importance', ascending = False)
zero_features = list(fi_domain[fi_domain['importance'] == 0.0]['feature'])
train_app_domain = train_app_domain.drop(columns = zero_features)
test_app_domain = test_app_domain.drop(columns = zero_features)

train_app_domain['TARGET'] = labels_train

submission_domain, fi_domain, metrics_domain = lightgbmmodel(train_app_domain, test_app_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)
submission_domain.to_csv('baseline_lgb_domain_features1.csv', index = False)