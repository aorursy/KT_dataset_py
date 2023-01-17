import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sb



from sklearn import metrics

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.info()
df.describe()
fraud = df.Class[df.Class == 1].count() / df.Class.count() * 100

non_fraud = 100 - fraud

print('% of fraud transactions: ', fraud)

print('% of non-fraud transactions: ', non_fraud)



sb.countplot('Class', data=df, palette='RdBu_r')

plt.show()
sb.boxplot(x = 'Class', y ='Amount', data = df, showfliers = False)

plt.show()
plt.figure(figsize = [10, 8])

sb.pairplot(df[['Time','Amount','Class']], hue='Class', palette = 'husl')

plt.show()
df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))
y = df.Class

X = df.drop('Class', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
tr_fraud = y_train[y_train == 1].count() / y_train.count() * 100

tr_non_fraud = 100 - tr_fraud



print('% of train fraud transactions: ', tr_fraud)

print('% of train non-fraud transactions: ', tr_non_fraud)

print('\n')



te_fraud = y_test[y_test == 1].count() / y_test.count() * 100

te_non_fraud = 100 - te_fraud

print('% of test fraud transactions: ', te_fraud)

print('% of test non-fraud transactions: ', te_non_fraud)
y_baseline = y_test.copy()

y_baseline[:] = 0



print('Accuracy:', metrics.accuracy_score(y_test, y_baseline))

print('Recall:', metrics.recall_score(y_test, y_baseline))

print('Precision:', metrics.precision_score(y_test, y_baseline))

print('f1-score:', metrics.f1_score(y_test, y_baseline))
log_reg = LogisticRegression(solver='liblinear')

log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr))

print('Recall:', metrics.recall_score(y_test, y_pred_lr))

print('Precision:', metrics.precision_score(y_test, y_pred_lr))

print('f1-score:', metrics.f1_score(y_test, y_pred_lr))
log_reg_cw = LogisticRegression(solver='liblinear', class_weight='balanced')

log_reg_cw.fit(X_train, y_train)

y_pred_lr_cw = log_reg_cw.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr_cw))

print('Recall:', metrics.recall_score(y_test, y_pred_lr_cw))

print('Precision:', metrics.precision_score(y_test, y_pred_lr_cw))

print('f1-score:', metrics.f1_score(y_test, y_pred_lr_cw))
rand_forst = RandomForestClassifier(n_estimators=50, random_state=0)

rand_forst.fit(X_train, y_train)

y_pred_rf = rand_forst.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf))

print('Recall:', metrics.recall_score(y_test, y_pred_rf))

print('Precision:', metrics.precision_score(y_test, y_pred_rf))

print('f1-score:', metrics.f1_score(y_test, y_pred_rf))
#y_pred = cross_val_predict(rand_forst, X_train, y_train)

sb.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap='Greens')

plt.xlabel('Modeled counts')

plt.ylabel('True counts')

plt.show()
df_train = pd.concat([X_train, y_train], axis=1)



fraid_tr = df_train[df_train.Class == 1]

non_fraid_tr = df_train[df_train.Class == 0]



fraud_tr_os = resample(fraid_tr, n_samples=len(non_fraid_tr), replace=True, random_state=0)



df_tr_os = pd.concat([fraud_tr_os, non_fraid_tr])
fraud = fraud_tr_os.Class.count() / df_tr_os.Class.count() * 100

non_fraud = 100 - fraud

print('% of fraud transactions: ', fraud)

print('% of non-fraud transactions: ', non_fraud)



sb.countplot('Class', data=df_tr_os, palette='hls')

plt.show()
plt.figure(figsize = [10, 8])

sb.heatmap(df_tr_os.corr(), vmin=-1, vmax=1, cmap = 'RdBu_r') #annot=True

plt.show()
plt.figure(figsize = [10, 8])

sb.pairplot(df_tr_os[['V4','V11','V12','V14','Class']], hue='Class', palette = 'husl')

plt.show()
fig, axis = plt.subplots(1, 3,figsize=(15,5))

sb.boxplot(x = 'Class', y ='V4', data = df_tr_os, ax = axis[0], showfliers = False, palette = 'hls')

sb.boxplot(x = 'Class', y ='V11', data = df_tr_os, ax = axis[1], showfliers = False, palette = 'RdBu_r')

sb.boxplot(x = 'Class', y ='V14', data = df_tr_os, ax = axis[2], showfliers = False)

plt.show()
X_tr = df_tr_os.drop('Class', axis=1)

y_tr = df_tr_os.Class



log_reg = LogisticRegression(solver='liblinear')

log_reg.fit(X_tr, y_tr)

y_pred_lr_os = log_reg.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr_os))

print('Recall:', metrics.recall_score(y_test, y_pred_lr_os))

print('Precision:', metrics.precision_score(y_test, y_pred_lr_os))

print('f1-score:', metrics.f1_score(y_test, y_pred_lr_os))
rand_forst = RandomForestClassifier(n_estimators=50, random_state=0)

rand_forst.fit(X_tr, y_tr)

y_pred_rf_os = rand_forst.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf_os))

print('Recall:', metrics.recall_score(y_test, y_pred_rf_os))

print('Precision:', metrics.precision_score(y_test, y_pred_rf_os))

print('f1-score:', metrics.f1_score(y_test, y_pred_rf_os))
sb.heatmap(confusion_matrix(y_test, y_pred_rf_os), annot=True, cmap='RdPu')

plt.xlabel('Modeled counts')

plt.ylabel('True counts')

plt.show()
smote = SMOTE(sampling_strategy=1.0, random_state=0)

X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
fraud = y_train_sm[y_train_sm == 1].count() / y_train_sm.count() * 100

non_fraud = 100 - fraud

print('% of fraud transactions: ', fraud)

print('% of non-fraud transactions: ', non_fraud)
log_reg = LogisticRegression(solver='liblinear')

log_reg.fit(X_train_sm, y_train_sm)

y_pred_lr_sm = log_reg.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr_sm))

print('Recall:', metrics.recall_score(y_test, y_pred_lr_sm))

print('Precision:', metrics.precision_score(y_test, y_pred_lr_sm))

print('f1-score:', metrics.f1_score(y_test, y_pred_lr_sm))
rand_forst = RandomForestClassifier(n_estimators=50, random_state=0)

rand_forst.fit(X_train_sm, y_train_sm)

y_pred_rf_sm = rand_forst.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf_sm))

print('Recall:', metrics.recall_score(y_test, y_pred_rf_sm))

print('Precision:', metrics.precision_score(y_test, y_pred_rf_sm))

print('f1-score:', metrics.f1_score(y_test, y_pred_rf_sm))
sb.heatmap(confusion_matrix(y_test, y_pred_rf_sm), annot=True, cmap='Purples')

plt.xlabel('Modeled counts')

plt.ylabel('True counts')

plt.show()
print('Logistic Regression: ', roc_auc_score(y_test, y_pred_lr))

print('Improved Logistic Regression: ', roc_auc_score(y_test, y_pred_lr_cw))

print('Random Forest: ', roc_auc_score(y_test, y_pred_rf))

print('Logistic Regression after Oversampling: ', roc_auc_score(y_test, y_pred_lr_os))

print('Random Forest after Oversampling: ', roc_auc_score(y_test, y_pred_rf_os))

print('Logistic Regression after SMOTE: ', roc_auc_score(y_test, y_pred_lr_sm))

print('Random Forest after SMOTE: ', roc_auc_score(y_test, y_pred_rf_sm))

fpr_lr, tpr_lr, thr_lr          = roc_curve(y_test, y_pred_lr)

fpr_lr_os, tpr_lr_os, thr_lr_os = roc_curve(y_test, y_pred_lr_os)

fpr_lr_sm, tpr_lr_sm, thr_lr_sm = roc_curve(y_test, y_pred_lr_sm)

fpr_rf, tpr_rf, thr_rf          = roc_curve(y_test, y_pred_rf)

fpr_rf_os, tpr_rf_os, thr_rf_os = roc_curve(y_test, y_pred_rf_os)

fpr_rf_sm, tpr_rf_sm, thr_rf_sm = roc_curve(y_test, y_pred_rf_sm)



plt.figure(figsize=(10,8))

plt.plot(fpr_lr, tpr_lr,       label='Logistic regression')

plt.plot(fpr_lr_os, tpr_lr_os, label='Logistic regression after oversampling')

plt.plot(fpr_lr_sm, tpr_lr_sm, label='Logistic regression after SMOTE')

plt.plot(fpr_rf, tpr_rf,       label='Random forest')

plt.plot(fpr_rf_os, tpr_rf_os, label='Random forest after oversampling')

plt.plot(fpr_rf_sm, tpr_rf_sm, label='Random forest after SMOTE')

plt.plot([0, 1], [0, 1], 'k:')



plt.xlim([-0.05, 1])

plt.ylim([0, 1])

plt.legend(fontsize=12)

plt.xlabel('False Positive Rate', fontsize=12)

plt.ylabel('True Positive Rate', fontsize=12)

plt.show()