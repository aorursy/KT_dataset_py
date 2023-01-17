#First public kernel, Here goes!
import os
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
#Load Data
input_dir = '../input/'
train = pd.read_csv(input_dir + 'train.csv')
test = pd.read_csv(input_dir + 'test.csv')
resources = pd.read_csv(input_dir + 'resources.csv')
submission = pd.read_csv(input_dir + 'sample_submission.csv')
 #Get the total price of each resource
resources['total_price'] = resources.quantity * resources.price
resources.head()
#For every project, get its mean price
mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
mean_total_price.head()
#Add id as column for merging
mean_total_price['id'] = mean_total_price.index
train = pd.merge(train, mean_total_price, on='id')
X_train = train.total_price.values.reshape(-1, 1) #reshape because of one column
y_train = train.project_is_approved.values
#CV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
n_folds = 5 
kf = KFold(n_splits=n_folds)
Average_AUC = 0
for (train_index, test_index) in kf.split(X_train):
    X, X_val = X_train[train_index], X_train[test_index]
    y, y_val = y_train[train_index], y_train[test_index]
    clf = LogisticRegression(random_state=333)
    clf.fit(X, y)
    pred = clf.predict_proba(X_val)[:,1]
    AUC = roc_auc_score(y_val, pred)
    Average_AUC += AUC
    print("AUC: {}".format(AUC))
Average_AUC = Average_AUC / n_folds
print("Average_AUC: {}".format(Average_AUC))
test = pd.merge(test, mean_total_price, on='id')
X_test = test.total_price.values.reshape(-1, 1)
pred_test = clf.predict_proba(X_test)[:,1]
submission.project_is_approved = pred_test
submission.head(5)
submission.to_csv('submission.csv', index=False)