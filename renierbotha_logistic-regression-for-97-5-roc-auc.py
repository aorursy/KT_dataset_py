import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Modelling
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter('ignore')

print(os.listdir("../input"))
df = pd.read_csv('../input/pulsar_stars.csv')

print(df.shape)
df.head()
X = df.drop(['target_class'], axis=1)
targets = df['target_class'].values
kf = StratifiedKFold(n_splits=5, random_state=2018)
k = 0

# Keep some things after each loop
val_auc_scores =  []
val_preds = []
val_true = []
for train_idx, val_idx in kf.split(X, targets):
#     Select the dat for this fold
    X_train, y_train = X.loc[train_idx, :], targets[train_idx]
    X_val, y_val = X.loc[val_idx, :], targets[val_idx]
    
#     Create simple model
    model = LogisticRegression()
    
#     Fit the model on the training set
    model.fit(X_train, y_train)
    
#     Get the prediction probabilities 
    val_probs = model.predict_proba(X_val)[:, 1]
    train_probs = model.predict_proba(X_train)[:, 1]
    
#     Calculate ROC AUC
    train_auc = roc_auc_score(y_train, train_probs)
    val_auc = roc_auc_score(y_val, val_probs)
    
#     Keep track of val auc
    val_auc_scores.append(val_auc)
    
    print(f'Fold: {k}')
    print(f'\tTrain AUC: {train_auc}\n\tVal AUC: {val_auc} ')
    
    k += 1
    
    val_preds.extend(model.predict(X_val))
    val_true.extend(y_val)
    
print(f'\nMean Validation ROC AUC: {np.mean(val_auc_scores)}')
from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(val_true, val_preds, labels=[0,1]))
cm.columns = ['pred_not_pulsar','pred_pulsar']
cm.index = ['not_pulsar','pulsar']
cm
