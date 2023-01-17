import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder # Machine learning
import seaborn as sns # data visualization
import matplotlib.pyplot as plt

%matplotlib inline

import os
print(os.listdir("../input"))

print("Importing dataframes")
dtrain = pd.read_csv('../input/application_train.csv')
dtest = pd.read_csv('../input/application_test.csv')
print("Done")
print ("")
print("Combining Train and Test")
df = pd.concat([dtrain,dtest],axis=0)
print("Done")
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

dtrain.head()

dtest.head()

target = dtrain['TARGET'].astype(int)
plt.figure(figsize=(10,6))
sns.distplot(target, kde=False, bins=8)

le = LabelEncoder()
le_cnt = 0

# Iterate through the columns
for col in dtrain:
    if dtrain[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(dtrain[col].unique())) <= 2:
            # Train on the training data
            le.fit(dtrain[col])
            # Transform both training and testing data
            dtrain[col] = le.transform(dtrain[col])
            dtest[col] = le.transform(dtest[col])
            
            # Keep track of how many columns were label encoded
            le_cnt += 1
            
print('Label encoding has been applied on %d columns' % le_cnt)
dtrain = pd.get_dummies(dtrain)
dtest = pd.get_dummies(dtest)

print('Training Features shape: ', dtrain.shape)
print('Testing Features shape: ', dtest.shape)

train_labels = dtrain['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
dtrain, dtest = dtrain.align(dtest, join = 'inner', axis = 1)

print('Training Features shape: ', dtrain.shape)
print('Testing Features shape: ', dtest.shape)

from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in dtrain:
    train = dtrain.drop(columns = ['TARGET'])
else:
    train = dtrain.copy()
features = list(train.columns)

# Copy of the testing data
test = dtest.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(dtest)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import gc

# Format the training and testing data 
train = np.array(dtrain)
test = np.array(dtest)

train_labels = np.array(train_labels).reshape((-1, ))

# 10 fold cross validation
folds = KFold(n_splits=5, shuffle=True, random_state=50)

# Validation and test predictions
valid_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])

# Iterate through each fold
for n_fold, (train_indices, valid_indices) in enumerate(folds.split(train)):
    # Training data for the fold
    train_fold, train_fold_labels = train[train_indices, :], train_labels[train_indices]
    
    # Validation data for the fold
    valid_fold, valid_fold_labels = train[valid_indices, :], train_labels[valid_indices]
    
    # LightGBM classifier with hyperparameters
    clf = LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        subsample_freq=1,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        random_state=0,
        silent=-1,
        verbose=-1,
    )
    
    # Fit on the training data, evaluate on the validation data
    clf.fit(train_fold, train_fold_labels, 
            eval_set= [(train_fold, train_fold_labels), (valid_fold, valid_fold_labels)], 
            eval_metric='auc', early_stopping_rounds=200, verbose = 100
           )
    
    # Validation preditions
    valid_preds[valid_indices] = clf.predict_proba(valid_fold, num_iteration=clf.best_iteration_)[:, 1]
    
    # Testing predictions
    test_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    # Display the performance for the current fold
    print('Fold %d AUC : %0.6f' % (n_fold + 1, roc_auc_score(valid_fold_labels, valid_preds[valid_indices])))
    
    # Delete variables to free up memory
    del clf, train_fold, train_fold_labels, valid_fold, valid_fold_labels
    gc.collect()

submission = dtest[['SK_ID_CURR']]
submission['TARGET'] = test_preds
submission.head()

submission.to_csv("abimannan_home.csv", index=False)

