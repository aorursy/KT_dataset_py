import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
header_name = ['Age', 'Workclass','fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country']
train_header = header_name + ['target']
# 是categorical 的欄位名稱
cat_column = ['Workclass', 'Education', 
        'Marital-status', 'Sex', 'Occupation', 'Education-num',
        'Relationship', 'Race', 'Native-country']
training = pd.read_csv('../input/train.csv', header=None,names=train_header)
testing = pd.read_csv('../input/test.csv',header=None,names=header_name)
print(training.head())
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
def preprocess(training, testing):
    y_train = training['target'].values
    training = training.drop(['target'], axis=1)

    for col in training.columns:
        if col in cat_column:
            label = preprocessing.LabelEncoder()
            label.fit(training[col].values)
            training[col] = label.transform(training[col].values)
            testing[col] = label.transform(testing[col].values)
    if 'target' in testing.columns:
        y_test = testing['target'].values
        testing = testing.drop(['target'], axis=1)
        return training, y_train, testing, y_test
    return training, y_train, testing
viz_df,target, _ = preprocess(training, testing)
viz_df['target'] = target
hmap = viz_df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.9,annot=True,cmap="BrBG", square=True);
fig, a = plt.subplots(1,1,figsize=(15,4))
sns.countplot(training['Age'],hue=training['target'], ax=a)
fig, a = plt.subplots(1,1,figsize=(15,4))
sns.countplot(training['Education'],hue=training['target'], ax=a)
fig, a = plt.subplots(1,1,figsize=(15,4))
sns.countplot(training['Sex'],hue=training['target'], ax=a)
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
training = pd.read_csv('../input/train.csv', header=None,names=train_header)
testing = pd.read_csv('../input/test.csv',header=None,names=header_name)
train, target, X_validate = preprocess(training, testing)
kf = KFold(n_splits=8, random_state=42)
f1_scores = []

for train_idx, test_idx in kf.split(target):
    X_train, X_test = train.iloc[train_idx], train.iloc[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    clf = XGBClassifier(colsample_bylevel=0.6, 
            colsample_bytree= 0.9, gamma= 5, max_delta_step= 3, max_depth= 11,
            min_child_weight= 5, n_estimators= 209, subsample= 0.9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
print("F1 平均: %f, 標準差 : %f" % (np.mean(f1_scores), np.std(f1_scores)))
kf = KFold(n_splits=8, random_state=42)
f1_scores = []
train = train.drop(['fnlwgt'], axis=1)
for train_idx, test_idx in kf.split(target):
    X_train, X_test = train.iloc[train_idx], train.iloc[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    clf = XGBClassifier(colsample_bylevel=0.6, 
            colsample_bytree= 0.9, gamma= 5, max_delta_step= 3, max_depth= 11,
            min_child_weight= 5, n_estimators= 209, subsample= 0.9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
print("F1 平均: %f, 標準差 : %f" % (np.mean(f1_scores), np.std(f1_scores)))
clf = XGBClassifier(colsample_bylevel=0.6, 
        colsample_bytree= 0.9, gamma= 5, max_delta_step= 3, max_depth= 11,
        min_child_weight= 5, n_estimators= 209, subsample= 0.9)
clf.fit(train, target)
X_validate = X_validate.drop(['fnlwgt'], axis=1)
y_validate = clf.predict(X_validate)
submission = pd.read_csv('../input/sub.csv')
print(submission.head())
submission['ans'] = y_validate.flatten().astype(int)
submission.to_csv('submission.csv', index=False)