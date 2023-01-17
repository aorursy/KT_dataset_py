# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import LGBMClassifier

pd.set_option('display.max_rows', 1000)
debug_on = False
random_state = 42
#This program will preidc on actual data for which labels are not availible.




def timer_start():
    global t0
    t0 = time.time()
    

def timer_end():
    t1 = time.time()   
    total = t1-t0
    print('Time elapsed', total)   
train_df = pd.read_csv('../input/train.csv')
sub_df  = pd.read_csv('../input/test.csv')
print('Train Shape:', train_df.shape, 'test Shape', sub_df.shape)


train_df.head()
#Combine Train and Test data
data_df = train_df.append(sub_df).reset_index()
data_copy_df = data_df.copy()

data_df.drop(['index', 'PassengerId'], axis = 1, inplace = True)

data_df.head()
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent= (data.isnull().sum() * 100 / data.isnull().count() ).sort_values(ascending = False)
    df = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    return df[df['Total'] != 0]
missing_data(data_df)

def label_encode(df):
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for cat_cols in categorical_columns:
        df[cat_cols], uniques = pd.factorize(df[cat_cols])
    return  df, categorical_columns

data_df, categorical_columns = label_encode(data_df)
categorical_columns
#The entries with -1 are null after Label Encoding, set them to null again
for col in categorical_columns:
 data_df[col].replace( -1 , np.NaN, inplace = True)
data_df[data_df['Age'] == -1].shape
from sklearn.model_selection import train_test_split
X_train  = data_df[data_df['Survived'].notnull()].copy() 
y_train =  X_train['Survived'].copy()
X_train.drop(['Survived'], inplace =  True, axis = 1)

X_sub    = data_df[data_df['Survived'].isnull()].copy()
X_sub.drop(['Survived'], inplace =  True, axis = 1)

features = X_train.columns

if debug_on:
   X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.33, random_state=42)
   print('Train Shape:', X_train.shape, 'test Shape', X_test.shape)
else:     
  print('Train Shape:', X_train.shape, 'test Shape', X_sub.shape)

timer_start()
folds = KFold(n_splits = 3, shuffle = True, random_state = random_state)
#folds = StratifiedKFold(n_splits= 3, shuffle=True, random_state = random_state)

# Since there are 5 classes crate a array with 5 cols for prediction prbabilties
oof_prob = np.zeros(shape=(X_train.shape[0],2)) 
sub_prob = np.zeros(shape=(X_sub.shape[0],2))  
test_prob =  np.zeros(shape=(X_test.shape[0],2))  
feature_importance_df = pd.DataFrame()


for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
    train_x, train_y = X_train.iloc[train_idx],  y_train.iloc[train_idx]
    valid_x, valid_y = X_train.iloc[valid_idx],  y_train.iloc[valid_idx]
    
   
    
    clf = LGBMClassifier(
                        n_jobs = 4,
                        n_estimators=10000,
                        learning_rate = 0.1,
                        objective = 'binary',                  
                    #    num_leaves=  25,
                     #   colsample_bytree= 0.672414,
                        # min_child_samples = 38,
                     #     subsample=  0.98233,
                        #  subsample_freq = 50,                      
                      #   max_depth= 9,
                      #   reg_alpha = 0.46512,                     
                      #   reg_lambda=  2.8242,
                     #   min_split_gain = 0.07321,
                     #   min_child_weight= 6.31887,
                        silent=-1,
                        verbose=-1,
                      #  device_type = 'gpu',
                        random_state  = random_state 
                        )

    clf.fit( 
              train_x, 
              train_y, 
              eval_set=[(train_x, train_y), (valid_x, valid_y)], 
              eval_metric= 'binary_error',
              verbose= 100,
              early_stopping_rounds= 200,
              feature_name  = 'auto',
              #Make Sure Pandas Dataframe is used or feature_name should be provided 
              categorical_feature = categorical_columns 
               )
    
   
    oof_prob[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)
    oof_pred = np.asarray([np.argmax(line) for line in oof_prob[valid_idx]])
    
    if debug_on:
      
       test_prob += clf.predict_proba(X_test, num_iteration = clf.best_iteration_) / folds.n_splits
    else:            
      #Average out the test set probablities
     sub_prob += clf.predict_proba(X_sub, num_iteration = clf.best_iteration_) / folds.n_splits
   
    
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df['feature'] = features.copy()
    fold_importance_df['importance'] = clf.feature_importances_
    fold_importance_df['fold'] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)
    print('\nFold %2d acuuracy: %.6f' %(n_fold + 1, accuracy_score(valid_y,  oof_pred)))
    print('Fold %2d error: %.6f' %(n_fold + 1, 1 - accuracy_score(valid_y,  oof_pred)))
    


print('Number of folds:', folds.n_splits )

oof_pred = np.asarray([np.argmax(line) for line in oof_prob])
if debug_on:
    print("\nTraining Data Shape", X_train.shape, "Test Data Shape", X_test.shape) 
    test_pred = np.asarray([np.argmax(line) for line in test_prob])
else:    
  print("\nTraining Data Shape", X_train.shape, "Test Data Shape", X_sub.shape) 
  sub_pred = np.asarray([np.argmax(line) for line in sub_prob])

print('\nFull Cross Validation accuracy %.6f' %accuracy_score(y_train, oof_pred))  
print('Full Cross Validation error %.6f' %(1 -accuracy_score(y_train, oof_pred)))    

if debug_on:
   print('\nTest accuracy %.6f' %accuracy_score(y_test, test_pred)) 
   lgb_prob =  test_prob
else:  
    sub_df['Survived'] = sub_pred
    sub_df[['PassengerId', 'Survived' ]].to_csv('lgb_sub.csv', index = False)


timer_end()