import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')

print("Shape of training data: {}".format(app_train.shape))
print("Shape of test data: {}".format(app_test.shape))
app_train.head()
def missing_values(df):
    
    total_missing = df.isnull().sum()/df.shape[0]
    percent_missing = total_missing*100
    return percent_missing.sort_values(ascending=False).round(1)
app_train_mis_values = missing_values(app_train)
df_app_train_miss_values= pd.DataFrame({'columns': app_train_mis_values.index, 'missing percent': app_train_mis_values.values})
drop_columns = df_app_train_miss_values[df_app_train_miss_values['missing percent'] >= 40]['columns'].tolist()
drop_columns
app_train = app_train.drop(drop_columns, axis=1)
app_test = app_test.drop(drop_columns, axis=1)

print(app_train.shape)
print(app_test.shape)
app_train['TARGET'].value_counts(dropna=False)
app_train.dtypes.value_counts()
app_train.select_dtypes(include=['object']).apply(pd.Series.nunique, axis=0)
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
train_labels = app_train['TARGET']
train_sk_id_curr = app_train['SK_ID_CURR']
test_sk_id_curr = app_test['SK_ID_CURR']

app_train.drop('SK_ID_CURR', inplace=True, axis=1)
app_test.drop('SK_ID_CURR', inplace=True, axis=1)

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
ratio = (train_labels == 0).sum()/ (train_labels == 1).sum()
ratio
#hist = xgb.cv(params, dtrain, num_rounds, nfold=10, stratified=True, early_stopping_rounds=10, verbose_eval=True)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(app_train, train_labels, test_size=0.2, stratify=train_labels, random_state=1)
print("Postive examples in train set: {}".format(np.sum(y_train==0)))
print("Negative examples in train set: {}".format(np.sum(y_train==1)))

print("Postive examples in test set: {}".format(np.sum(y_test==0)))
print("Negative examples in test set: {}".format(np.sum(y_test==1)))
clf = XGBClassifier(n_estimators=1000, objective='binary:logistic', gamma=0.1, subsample=0.5, scale_pos_weight=ratio )
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=10)
n_estimators = clf.best_ntree_limit
clf = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic', gamma=0.1, subsample=0.5, scale_pos_weight=ratio )
clf.fit(app_train.values, train_labels.values, eval_set=[(app_train.values, train_labels.values)], eval_metric='auc')
predictions = clf.predict_proba(app_test.values)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_sk_id_curr.values, 'TARGET': predictions})
submission.head()
submission.to_csv('baseline_xgboost_1.csv', index = False)
top25 = pd.DataFrame({'features': app_test.columns, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False).head(25)
sns.barplot(x=top25['importance'], y=top25['features'])