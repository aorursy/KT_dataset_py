import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold, train_test_split
import itertools
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train_x = train.drop(['Survived'], axis = 1)
train_y = train['Survived']

test_x = test.copy()
for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))
train_x = train_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin',], axis = 1)
test_x = test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

pred = model.predict_proba(test_x)[:, 1]
pred_label = np.where(pred > 0.5, 1, 0)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index = False)
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
scores_accuracy = []
scores_logloss = []
kf = KFold(n_splits=4, shuffle = True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    model = XGBClassifier(n_estimators = 20, random_state = 71)
    model.fit(tr_x, tr_y)
    
    va_pred = model.predict_proba(va_x)[:, 1]
    
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred>0.5)
    
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
                                     
                              
import itertools
param_space = {
    'max_depth': [3, 5, 7], 
    'min_child_weight': [1.0, 2.0, 4.0]
}

param_combinations = itertools.product(param_space['max_depth'], param_space["min_child_weight"])

params = []
scores = []

for max_depth, min_child_weight in param_combinations:
    
    score_folds = []
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        
        model = XGBClassifier(n_estimators=20, random_state=71, 
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)
        
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)
        
    score_mean = np.mean(score_folds)
    
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)
    
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')