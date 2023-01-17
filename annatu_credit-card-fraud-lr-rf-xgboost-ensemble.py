import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, precision_recall_curve

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

data = pd.read_csv("../input/creditcard.csv")
data.head()
data.shape
data.info()
data.describe().T
# split test data
testportion = int(data.shape[0] * 0.2)
test_data = data[:testportion]
test_data.head()
test_data.shape
trainportion = int(data.shape[0] * 0.8)
train_data = data[:trainportion]
train_data.head()
print(test_data.shape, train_data.shape, train_data.shape[0]+test_data.shape[0])
print(data.shape)
train_data.Class.value_counts()
# print(train.groupby(['Class']).size())
227394/395
# q = pd.crosstab(data.Time, data.Class)
# pd.value_counts(data['Class'], sort=True)
train_data.Class.value_counts().plot(kind='bar')
Fraud_transacation = train_data[train_data["Class"]==1]
Normal_transacation = train_data[train_data["Class"]==0]
print("percentage of fraud transacation is", len(Fraud_transacation) / len(train_data) * 100)
print("percentage of normal transacation is", len(Normal_transacation) / len(train_data) * 100)
# pd.crosstab(data.Class, data.Time).plot(kind='bar', stacked=True)
xFraud = train_data.loc[data.Class==1]
#print(xFraud)
xGood = train_data.loc[data.Class==0]
print(xGood)
sns.distplot(train_data.Amount)
plt.figure(figsize=(10, 6))
plt.subplot(121)
Fraud_transacation.Amount.plot.hist(title="Fraud Transacation")
plt.subplot(122)
Normal_transacation.Amount.plot.hist(title="Normal Transaction")
plt.show()
# Amount 显示是 “长尾”数据，套路是np.log(Amount+1) 变成正态分布

train_data['Log_amount'] = np.log(train_data.Amount + 1)
#(data[data.Class==1].Log_amount).hist()
sns.distplot((train_data.Log_amount), bins= 50)
plt.grid(True)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
ax1.hist(train_data[train_data.Class==1].Amount)
ax1.set_title("Fraud")

ax2.hist(train_data[train_data.Class==0].Amount)
ax2.set_title("Non-Fraud")

plt.xlabel("Amount $")
plt.ylabel("# of xsactions")
plt.yscale('log')
plt.grid(True)
print("Max Fraud Amount = $", max(train_data[train_data.Class==1].Amount))
# 归一化 
from sklearn.preprocessing import StandardScaler

train_data['Amount'] = StandardScaler().fit_transform(train_data['Amount'].values.reshape(-1, 1))
train_data['Time'] = StandardScaler().fit_transform(train_data['Amount'].values.reshape(-1, 1))
print("Longest Time = ", max(train_data[train_data.Class==1].Time))
train_data.Time = train_data.Time.apply(lambda x: x//3600)
#pd.qcut(data.Time, 24).mean() 
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
bins = 40
ax1.hist(train_data[train_data.Class==1].Time, bins = bins)
ax1.set_title("Fraud")

ax2.hist(train_data[train_data.Class==0].Time, bins = bins)
ax2.set_title("Non-Fraud")

plt.xlabel("Time in 24 hour")
plt.ylabel("# of xsactions")
plt.yscale('log')
plt.grid(True)
train_data[train_data.Class==1].Time.value_counts().sort_index()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
ax1.scatter(train_data[train_data.Class==1].Time, train_data[train_data.Class==1].Amount)
ax1.set_title("Fraud")

ax2.scatter(train_data[train_data.Class==0].Time, train_data[train_data.Class==0].Amount)
ax2.set_title("Non-Fraud")
plt.xlabel("Time in hour")
plt.ylabel("Amount")

v_features = train_data.ix[:,2:30].columns

plt.figure(figsize=(20, 6*28))
gs = gridspec.GridSpec(28, 1)
# sns.distplot(data["V1"][data["Class"]==1], bins=50)
# sns.distplot(data["V1"][data["Class"]==0], bins=100)

# hist(data["V1"][data["Class"]==1], bins = bins)
# set_title("Fraud")

# ax2.hist(data["V1"][data["Class"]==0], bins = bins)
# ax2.set_title("Non-Fraud")

for i, vf in enumerate(train_data[v_features]):
    #print(i, vf)
    ax = plt.subplot(gs[i])
    sns.distplot(train_data[vf][train_data["Class"]==1], bins=100)
    sns.distplot(train_data[vf][train_data["Class"]==0], bins=100)
    ax.set_title("Diagram of " + str(vf))
    plt.grid(True)
droplist = ['V10', 'V11', 'V14', 'V16', 'V17', 'V18']
data_new = train_data.drop(droplist, axis=1)
data_new = data_new.dropna()
data_new.shape
data_new.head()
# Feature Selection

def NB_Classify(drop_var):
    clf = BernoulliNB()
    scores = cross_val_score(clf, train_data.drop(drop_var, axis=1), train_data.Class, cv=4, scoring='f1')
    print(drop_var, np.mean(scores))
    return np.mean(scores)

var_list = list(train_data.columns.values)
features_importance = dict()
for var in var_list:
    acc = NB_Classify(['Class', var])
    features_importance[var] = acc
    
#print(features_importance)

#sorted(features_importance.items())
sorted(features_importance.items(), key=lambda x: x[1])
data_drop = data_new.drop(['Time', 'Log_amount', 'V3', 'Class'], axis=1, inplace=True)
#vlist = ['V13', 'V15', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V28']
vlist = ['V13', 'V25', 'V26']
droplist = vlist + ['V3', 'Time']
droplist + ['Class']
X = train_data.iloc[:, data.columns != 'Class']
y = train_data.iloc[:, data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
X_train.head()
y_train.head()
X_train['Log_amount'] = np.log(X_train.Amount + 1)
X_train.head()
X_test['Log_amount'] = np.log(X_test.Amount + 1)
X_test.head()
clf = BernoulliNB()
clf.fit(X_train.drop(droplist, axis=1), y_train)
y_pred = clf.predict(X_train.drop(droplist, axis=1))
print(f1_score(y_train, y_pred))
pred_bn = clf.predict(X_test.drop(droplist, axis=1))
pred_bn = pd.DataFrame(pred_bn, columns=['Class'])
pred_bn.to_csv('./pred_creditcard_test_bn.csv')
# """
# Using Random over-sampling to generate new samples by randomly sampling with 
# replacement the current available samples. RandomOverSampler provides such function
# package needed: 
#     conda install -c glemaitre imbalanced-learn
# """ 
# from sklearn.datasets import make_classification
# from imblearn.over_sampling import RandomOverSampler
# from collections import Counter

# n_samples=1000
# weights=(0.01, 0.01, 0.98)
# n_classes=3
# class_sep=0.8
# n_clusters=1
    
# X, y = make_classification(n_samples=n_samples, n_features=2,
#                            n_informative=2, n_redundant=0, n_repeated=0,
#                            n_classes=n_classes,
#                            n_clusters_per_class=n_clusters,
#                            weights=list(weights),
#                            class_sep=class_sep, random_state=0)

# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_sample(X, y)
# print(sorted(Counter(y_resampled).items()))
# Xtrain, Xval, ytrain, yval = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
# print(Xtrain.shape, Xval.shape, ytrain.shape, yval.shape)
# """
# Solution 2: for the dataset balancing

# """
# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=42)    # 处理过采样的方法
# X2_resampled, y2_resampled = sm.fit_sample(X_train, y_train)
# # print('通过SMOTE方法平衡正负样本后')
# n_sample = y.shape[0]
# n_pos_sample = y[y == 0].shape[0]
# n_neg_sample = y[y == 1].shape[0]
# print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
#                                                    n_pos_sample / n_sample))
                                               
# Xtrain, Xval, ytrain, yval = train_test_split(X2_resampled, y2_resampled, test_size=0.2, random_state=0)
# print(Xtrain.shape, Xval.shape, ytrain.shape, yval.shape)
from sklearn.metrics import recall_score
# Under sampling
# or directly use http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.under_sampling.RandomUnderSampler.html
def undersample(credit_data, ratio):
  number_records_fraud = len(credit_data[credit_data.Class == 1])
  fraud_indices = np.array(credit_data[credit_data.Class == 1].index)


  normal_indices = credit_data[credit_data.Class == 0].index
  random_normal_indices = np.random.choice(normal_indices, number_records_fraud * ratio, replace = False)

  under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
  under_sample_data = credit_data.iloc[under_sample_indices,:]

  X_train_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
  y_train_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
  
  return (X_train_undersample, y_train_undersample.values.ravel())


X_train_undersample, y_train_undersample = undersample(train_data, 250)
train_data.head()
nb_clf_resample = BernoulliNB()
nb_clf_resample.fit(X_train_undersample.drop(droplist,axis=1), y_train_undersample)

pred = nb_clf_resample.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, pred))
# print(precision_score(data_test['Class'], pred))
# print(recall_score(data_test['Class'], pred))
# print(f1_score(data_test['Class'], pred))
X_train_undersample.head()
model_lr = LogisticRegression(C = 0.08)  #  C = 0.01

model_lr.fit(X_train.drop(droplist, axis=1), y_train)
y_pred = model_lr.predict(X_train.drop(droplist, axis=1))
print(f1_score(y_train, y_pred))

pred_lr = model_lr.predict(X_test.drop(droplist, axis=1))
X_train_undersample, y_train_undersample = undersample(train_data, 300)
X_test_undersample, y_test_undersample = undersample(test_data, 300)
print(X_train_undersample.shape, y_train_undersample.shape)
print(X_test_undersample.shape, y_test_undersample.shape)
X_test_undersample['Log_amount'] = np.log(X_test_undersample.Amount + 1)
X_test_undersample.head()
print('Now trying LR undersampling...')

model_lr.fit(X_train_undersample.drop(droplist, axis=1), y_train_undersample)
y_pred_sampled = model_lr.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, y_pred_sampled))

pred_lr_test = model_lr.predict(X_test_undersample.drop(droplist, axis=1))
pred_lr_test
output = pd.DataFrame(pred_lr_test, columns=['Class'])
#output['ID'] = output.reset_index
output.to_csv('pred_creditcard_fraud_lr.csv', index=True)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier (estimators = [('nb', nb_clf_resample), ('lr', model_lr)], voting='soft', weights = [1, 1.5])
voting_clf.fit(X_train_undersample.drop(droplist, axis=1), y_train_undersample)
y_pred_voting_sampled = voting_clf.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, y_pred_voting_sampled))

pred1_voting_test = voting_clf.predict(X_test_undersample.drop(droplist, axis=1))
pred2_voting_test = voting_clf.predict_proba(X_test_undersample.drop(droplist, axis=1))[:,1]
print(pred1_voting_test, pred2_voting_test)
output = pd.DataFrame(pred1_voting_test, columns=['Class'])
output.to_csv('pred_creditcard_fraud_lr.csv', index=True)
output = pd.DataFrame(pred2_voting_test, columns=['Class'])
output.to_csv('pred2_creditcard_fraud_lr.csv', index=True)
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train_undersample.drop(droplist, axis=1), y_train_undersample)
y_pred_rf_sampled = rf_clf.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, y_pred_rf_sampled))

pred_rf_test = rf_clf.predict(X_test_undersample.drop(droplist, axis=1))
output = pd.DataFrame(pred_rf_test, columns=['Class'])
output.to_csv('pred_creditcard_fraud_rf.csv', index=True)
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_jobs = -1)
xgb_clf.fit(X_train_undersample.drop(droplist, axis=1), y_train_undersample)
y_pred_xgb_sampled = xgb_clf.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, y_pred_xgb_sampled))

pred_xgb_test = xgb_clf.predict(X_test_undersample.drop(droplist, axis=1))
output = pd.DataFrame(pred_xgb_test, columns=['Class'])
output.to_csv('pred_creditcard_fraud_xgb.csv', index=True)
voting_clf = VotingClassifier (estimators = [('nb', nb_clf_resample), ('lr', model_lr), ('rf', rf_clf)], voting='soft', weights = [1, 1.5, 2])
voting_clf.fit(X_train_undersample.drop(droplist, axis=1), y_train_undersample)
y_pred_voting_sampled = voting_clf.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, y_pred_voting_sampled))

pred1_voting_test = voting_clf.predict(X_test_undersample.drop(droplist, axis=1))
pred2_voting_test = voting_clf.predict_proba(X_test_undersample.drop(droplist, axis=1))[:,1]
output = pd.DataFrame(pred1_voting_test, columns=['Class'])
output.to_csv('pred_creditcard_fraud_Ensemble.csv', index=True)
voting2_clf = VotingClassifier (estimators = [('xgb', xgb_clf), ('rf', rf_clf)], voting='soft', weights = [1.5, 4])
voting2_clf.fit(X_train_undersample.drop(droplist, axis=1), y_train_undersample)
y_pred_voting2_sampled = voting2_clf.predict(X_train_undersample.drop(droplist, axis=1))
print(f1_score(y_train_undersample, y_pred_voting2_sampled))

pred1_voting2_test = voting2_clf.predict(X_test_undersample.drop(droplist, axis=1))
pred2_voting2_test = voting2_clf.predict_proba(X_test_undersample.drop(droplist, axis=1))[:,1]
output = pd.DataFrame(pred1_voting2_test, columns=['Class'])
output.to_csv('pred_creditcard_fraud_Ensemble2.csv', index=True)

