# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, cv, Pool
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from itertools import combinations
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read in data
test = pd.read_csv("../input/amazon-employee-access-challenge/test.csv")
train = pd.read_csv("../input/amazon-employee-access-challenge/train.csv")
def performance(model, X_test, y_test):
# Make predictions on test set
    y_pred=model.predict(X_test)
    y_pred=np.round(y_pred)
    
    # Confusion matrix
    print(confusion_matrix(y_test, y_pred))
    
    # AUC score
    y_pred_prob = model.predict_proba(X_test)
    print("AUC score: ", roc_auc_score(y_test, y_pred_prob[:,1]))
    
    # Logloss
    print("Logloss : ", log_loss(y_test, y_pred_prob))

    # Accuracy, Precision, Recall, F1 score
    print(classification_report(y_test, y_pred))
    
    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    plt.plot([0, 1], [0, 1],'k--')
    plt.plot(fpr, tpr, label='Neural Network')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))
print("Train datatypes: {}, Test datatypes: {}".format(train.dtypes, test.dtypes))
train.head()
test.head()
print(train.isnull().any()) 
print(test.isnull().any())

unique_train= pd.DataFrame([(col,train[col].nunique()) for col in train.columns], 
                           columns=['Columns', 'Unique categories'])
unique_test=pd.DataFrame([(col,test[col].nunique()) for col in test.columns],
                columns=['Columns', 'Unique categories'])
unique_train=unique_train[1:]
unique_test=unique_test[1:]

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].bar(unique_train.Columns, unique_train['Unique categories'])
ax[1].bar(unique_test.Columns, unique_test['Unique categories'])
plt.xticks(rotation=90)
sns.countplot(train['ACTION'])
if (sum(train.duplicated()), sum(test.duplicated())) == (0,0):
    print('No duplicated rows')
else: 
    print('train: ',sum(train.duplicated()))
    print('test: ',sum(train.duplicated()))
# Check for duplicated columns                          

for col1,col2 in combinations(train.columns, 2):
    condition1=len(train.groupby([col1,col2]).size())==len(train.groupby([col1]).size())
    condition2=len(train.groupby([col1,col2]).size())==len(train.groupby([col2]).size())
    condition3=(train[col1].nunique()==train[col2].nunique())
    if (condition1 | condition2) & condition3:
        print(col1,col2)
        print('Potential Categorical column duplication')
print(train['ROLE_TITLE'].mean())
print(train['ROLE_CODE'].mean())
np.random.seed(123)
# Drop duplicated column
train.drop('ROLE_CODE', axis=1, inplace=True)
test.drop('ROLE_CODE', axis=1, inplace=True)


# Split into features and target y-target and X-features
y = train['ACTION']
X = train.drop('ACTION', axis=1)

# Split into train & validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
cat_features = [*range(8)]
print(cat_features)
model = CatBoostClassifier(custom_metric=['TotalF1'], early_stopping_rounds=100, eval_metric='AUC')

model.fit(X_train, y_train, cat_features=cat_features,
          eval_set=(X_val, y_val), plot=True, verbose=False, use_best_model=True)
performance(model, X_val, y_val)
feat_imp=model.get_feature_importance(prettified=True)
plt.bar(feat_imp['Feature Id'], feat_imp['Importances'])
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.xticks(rotation=90)
sub=pd.read_csv("../input/amazon-employee-access-challenge/sampleSubmission.csv")
sum(test.id==sub.Id), test.shape
y_pred=model.predict_proba(test.drop('id', axis=1))
sub.Action=y_pred[:,1]
sub.to_csv('amazon1.csv', index=False, header=True)
sub.head()
# Best of the tuned models
model = CatBoostClassifier(border_count=248, depth=4, l2_leaf_reg=4.830204209625978,
                           scale_pos_weight=0.4107081177319144, 
                           eval_metric='AUC',
                           use_best_model=True,
                          early_stopping_rounds=100)
best=model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), use_best_model=True,
          verbose=False, plot=False)
performance(model, X_val, y_val)
model = CatBoostClassifier(border_count=248, depth=4, l2_leaf_reg=4.830204209625978,
                           scale_pos_weight=0.4107081177319144,
                           loss_function='Logloss',
                           eval_metric='AUC',
                           use_best_model=True,
                          early_stopping_rounds=100)
cv_data = cv(Pool(X_train, y_train, cat_features=cat_features), params=model.get_params(),
             verbose=False)


score = np.max(cv_data['test-AUC-mean'])
print('AUC score from cross-validation: ', score)
