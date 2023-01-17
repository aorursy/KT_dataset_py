import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PATH = '/kaggle/input/lish-moa/'
train_df = pd.read_csv(PATH + 'train_features.csv')
test_df = pd.read_csv(PATH + 'test_features.csv')

target_df = pd.read_csv(PATH + 'train_targets_scored.csv')
sub_df = pd.read_csv(PATH + 'sample_submission.csv')
train_df.head()
train_df.drop(['sig_id'], axis=1, inplace=True)
test_df.drop(['sig_id'], axis=1, inplace=True)
target_df.head()
target_df.drop(['sig_id'], axis=1, inplace=True)
target_df.sum(axis=1).sample(20)
idx = len(train_df)
data_df = pd.concat([train_df, test_df], axis = 0)
del train_df, test_df
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

category_cols = ['cp_dose', 'cp_type']

for cols in category_cols:
    data_df[cols] = enc.fit_transform(data_df[cols])
X_train = data_df.iloc[:idx,:]
X_test = data_df.iloc[idx:,:]
y_train = target_df
from xgboost import XGBClassifier

model = XGBClassifier(
            n_estimators=500,
            seed=42,
            learning_rate=0.1,
            max_depth=5, 
            colsample_bytree=1,
            subsample=1,
            tree_method='gpu_hist')
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

columns = target_df.columns
submission = sub_df.copy()
submission.loc[:,columns] = 0

for c, column in enumerate(columns):
    y = y_train[column]
    loss = 0
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)  
    for ix, (train_idx, val_idx) in enumerate(kf.split(X_train)):
              
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
        model.fit(
            X_train_cv, y_train_cv, 
            eval_set=[(X_val_cv,  y_val_cv)], 
            eval_metric = "logloss", 
            early_stopping_rounds=30, 
            verbose=0)
        
        val_preds = model.predict(X_val_cv)
        
        loss += log_loss(y_val_cv,val_preds, labels=[0,1])
        
        preds = model.predict_proba(X_test)[:,1]
        submission[column] += preds/5
                         
    print("model "+str(c+1)+": loss ="+str(loss/5))
submission.loc[test['cp_type']==1, target_df.columns] = 0
submission.to_csv('submission.csv', index=False)
from sklearn.multioutput import MultiOutputClassifier

mo_model = MultiOutputClassifier(model)
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

test_preds = np.zeros((X_test.shape[0], y_train.shape[1]))

kf = KFold(n_splits=5, random_state=42, shuffle=True)

loss = []

for ix, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    
    X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    
    mo_model.fit(X_train_cv, y_train_cv)
    val_preds = model.predict_proba(X_val_cv) 
    val_preds = np.array(val_preds)[:,:,1].T #(num_labels,num_samples,prob_0/1)
    
    loss.append(log_loss(np.ravel(y_val_cv), np.ravel(val_preds)))
    
    preds = model.predict_proba(X_test)
    preds = np.array(preds)[:,:,1].T #(num_labels,num_samples,prob_0/1)
    test_preds += preds / 5 

print(loss)
print('Mean CV loss across folds', np.mean(loss))
mask = X_test['cp_type']=='ctl_vehicle'
test_preds[mask] = 0
sub_df.iloc[:,1:] = test_preds
sub_df.to_csv('submission.csv', index=False)