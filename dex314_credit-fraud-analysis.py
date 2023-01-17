import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, log_loss, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import statsmodels.api as sma
import lightgbm as lgb
from collections import Counter

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/creditcard.csv", index_col=0)
# print(df.columns)
print(df.head(3))
df.info()
df.describe()
Counter(df.Class)
print("Fraud to NonFraud Ratio of {:.3f}%".format(492/284315*100))
fig, ax=plt.subplots(1,1,figsize=(12,6))
df[df.Class==1]['Amount'].hist(bins=100,ax=ax,color='b');
plt.title('Histogram of Fraud Amounts');
plt.ylabel('Counts'); plt.xlabel('$');
fig, ax=plt.subplots(1,1,figsize=(10,8))
sns.heatmap(df.corr(), vmin=-1, vmax=1, ax=ax, cmap='coolwarm');
plt.title('Heat Map of Variable Correlations');
corrs_amt = df.drop('Class',axis=1).corr()['Amount']
print(corrs_amt[np.abs(corrs_amt) > 0.3])
vars_to_cover = ['Amount','V2','V5','V7','V20']
print(df[vars_to_cover].corr())
lin_mod = sma.OLS(exog=df[['V2','V5','V7','V20']], endog=df[['Amount']])
lin_fit = lin_mod.fit()
print(lin_fit.summary())
lin_pred = lin_fit.predict()
lin_pred_df = pd.DataFrame(lin_pred, index=df.index)
fig, ax=plt.subplots(1,1,figsize=(12,6))
lin_pred_df.iloc[:200].plot(ax=ax, style=['r-'], label='Pred', legend=True);
df.Amount.iloc[:200].plot(ax=ax, style=['b:'], label='Actuals', legend=True);
plt.title('First 200 Instances of Amount');
plt.xlabel('Time'); plt.ylabel('$');
from imblearn.over_sampling import SMOTE
y_full = df['Class']
x_full = df.drop(['Class','Amount'], axis=1)
ism = SMOTE(random_state=42)
x_rs, y_rs = ism.fit_sample(x_full, y_full)
print('Resampled dataset shape {}'.format(Counter(y_rs)))
x_rs = pd.DataFrame(x_rs, columns = x_full.columns)
y_rs = pd.DataFrame(y_rs)
corrs = df.drop('Amount',axis=1).corr()['Class']
print(corrs[np.abs(corrs) > 0.2])
xto, xvo, yto, yvo = train_test_split(x_rs, y_rs, test_size=0.2, random_state=42)
print(xto.shape, xvo.shape, yto.shape, yvo.shape)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold
lr_os_mod = sma.Logit(endog = yto, exog = sma.add_constant(pd.DataFrame(xto, columns=x_rs.columns)[['V10','V12','V14','V17']]))
lr_os_fit = lr_os_mod.fit()
print(lr_os_fit.summary2())
lr_os_pred = lr_os_fit.predict(sma.add_constant(pd.DataFrame(xvo, columns=x_rs.columns)[['V10','V12','V14','V17']]))
print(lr_os_pred.head(), '\n', yvo.head())
lr_os_pred_rnd = lr_os_pred.round(0).astype(int)
lr_os_pred_rnd.head()
confusion_matrix(lr_os_pred_rnd, yvo)/len(yvo)
print("Precision     : {:.4f}".format(precision_score(lr_os_pred_rnd, yvo)))
print("Recall        : {:.4f}".format(recall_score(lr_os_pred_rnd, yvo)))
print("Accuracy      : {:.4f}".format(accuracy_score(lr_os_pred_rnd, yvo)))
print("ROC/AUC Score : {:.4f}".format(roc_auc_score(lr_os_pred_rnd, yvo)))
print("F1 Score      : {:.4f}".format( 2*(precision_score(lr_os_pred_rnd, yvo)*recall_score(lr_os_pred_rnd, yvo)) \
                                        / (precision_score(lr_os_pred_rnd, yvo)+recall_score(lr_os_pred_rnd, yvo)) ))
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
x_rs, y_rs = rus.fit_sample(x_full, y_full)
print('Resampled dataset shape {}'.format(Counter(y_rs)))
x_rs = pd.DataFrame(x_rs, columns = x_full.columns)
y_rs = pd.DataFrame(y_rs)
xto, xvo, yto, yvo = train_test_split(x_rs, y_rs, test_size=0.2, random_state=42)
print(xto.shape, xvo.shape, yto.shape, yvo.shape)
lr_us_mod = sma.Logit(endog = yto, exog = sma.add_constant(pd.DataFrame(xto, columns=x_rs.columns)[['V10','V12','V14','V17']]))
lr_us_fit = lr_us_mod.fit()
print(lr_us_fit.summary2())
lr_us_pred = lr_us_fit.predict(sma.add_constant(pd.DataFrame(xvo, columns=x_rs.columns)[['V10','V12','V14','V17']]))
lr_us_pred_rnd = lr_us_pred.round(0).astype(int)
print("Precision     : {:.4f}".format(precision_score(lr_us_pred_rnd, yvo)))
print("Recall        : {:.4f}".format(recall_score(lr_us_pred_rnd, yvo)))
print("Accuracy      : {:.4f}".format(accuracy_score(lr_us_pred_rnd, yvo)))
print("ROC/AUC Score : {:.4f}".format(roc_auc_score(lr_us_pred_rnd, yvo)))
print("F1 Score      : {:.4f}".format( 2*(precision_score(lr_us_pred_rnd, yvo)*recall_score(lr_us_pred_rnd, yvo)) \
                                        / (precision_score(lr_us_pred_rnd, yvo)+recall_score(lr_us_pred_rnd, yvo)) ))
from imblearn.combine import SMOTEENN
cse = SMOTEENN(random_state=42)
x_rs, y_rs = cse.fit_sample(x_full, y_full)
print('Resampled dataset shape {}'.format(Counter(y_rs)))
x_rs = pd.DataFrame(x_rs, columns = x_full.columns)
y_rs = pd.DataFrame(y_rs)

# xto, xvo, yto, yvo = train_test_split(x_rs, y_rs, test_size=0.2, random_state=42)
# print(xto.shape, xvo.shape, yto.shape, yvo.shape)
params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.05, 
    'max_depth': 5,
    'num_leaves': 92, 
    'min_data_in_leaf': 46, 
    'lambda_l1': 1.0,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}

kfolds = 3
kd = 0
preds = 0
for i in range(kfolds):
    print('In kfold:',str(i+1))
    xt,xv,yt,yv = train_test_split(x_rs, y_rs, test_size=0.2, random_state=(i*42))
    
    trn = lgb.Dataset(xt,yt.values.flatten())
    val = lgb.Dataset(xv,yv.values.flatten())
    model = lgb.train(params, train_set=trn, num_boost_round=100,
                     valid_sets=[val], valid_names=['val'],
                     verbose_eval=20,
                     early_stopping_rounds=40)
    
    pred = model.predict(xv, num_iteration=model.best_iteration+50)
    preds += pred
    kd += 1
    print('=========================')
    print("    Precision : {:.4f}".format(precision_score(np.round(pred,0).astype(int), yv)))
    print("    Recall    : {:.4f}".format(recall_score(np.round(pred,0).astype(int), yv)))
    print("    Accuracy  : {:.4f}".format(accuracy_score(np.round(pred,0).astype(int), yv)))
    print("ROC/AUC Score : {:.4f}".format(roc_auc_score(np.round(pred,0).astype(int), yv)))
    print("    F1 Score  : {:.4f}".format( 2*(precision_score(np.round(pred,0).astype(int), yv)*recall_score(np.round(pred,0).astype(int), yv)) \
                                        / (precision_score(np.round(pred,0).astype(int), yv)+recall_score(np.round(pred,0).astype(int), yv)) ))
    print('=========================')
preds /= kd
lgb.plot_importance(model, figsize=(12,8));
X = x_rs[['V4','V10','V12','V14']]
y = y_rs
logmod = sma.Logit(endog=y, exog=sma.add_constant(X))
logfit = logmod.fit()
print(logfit.summary2())
logPred = logfit.predict(sma.add_constant(x_rs[['V4','V10','V12','V14']]))
print(logPred.head(3))
print("Precision     : {:.4f}".format(precision_score(np.round(logPred,0).astype(int), y_rs)))
print("Recall        : {:.4f}".format(recall_score(np.round(logPred,0).astype(int), y_rs)))
print("Accuracy      : {:.4f}".format(accuracy_score(np.round(logPred,0).astype(int), y_rs)))
print("ROC/AUC Score : {:.4f}".format(roc_auc_score(np.round(logPred,0).astype(int), y_rs)))
print("F1 Score      : {:.4f}".format( 2*(precision_score(np.round(logPred,0).astype(int), y_rs)*recall_score(np.round(logPred,0).astype(int), y_rs)) \
                                    / (precision_score(np.round(logPred,0).astype(int), y_rs)+recall_score(np.round(logPred,0).astype(int), y_rs)) ))
print(lr_os_pred.shape, lr_us_pred.shape, logPred.shape, y_rs.shape)
pred_methods = [lr_os_fit.predict(sma.add_constant(pd.DataFrame(x_rs, columns=x_rs.columns)[['V10','V12','V14','V17']])),
                lr_us_fit.predict(sma.add_constant(pd.DataFrame(x_rs, columns=x_rs.columns)[['V10','V12','V14','V17']])), 
                logfit.predict(sma.add_constant(x_rs[['V4','V10','V12','V14']]))]
cols = ['b','r','g','m','c']
pred_fits = ['OverSampling - Logit', 'UnderSampling - Logit', 'Combined Sampling - Light GBM']
fig, ax=plt.subplots(1,1,figsize=(16,10))
plt.title('Variations on Sampling & Models ROC/AUC Curves');
for i in range(len(pred_methods)):
    fpr, tpr, thresholds = roc_curve(y_rs, pred_methods[i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=1, alpha=0.9, color=cols[i],
                 label='Model: %s   (AUC = %0.4f)' % (pred_fits[i], roc_auc))
ax.plot(np.linspace(0,1,100),np.linspace(0,1,100), 'k:');
ax.legend();
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
# cv = StratifiedKFold(n_splits=5)
# mod = LogisticRegression(C=5, fit_intercept=True, penalty='l2',
#                            n_jobs=1, verbose=20, random_state=42)
# X_lr = df.drop(['Class','Amount'],axis=1).values
# y_lr = df['Class'].values
# tprs = []
# aucs = []
# i=0
# cols = ['b','r','g','m','c']
# fig, ax=plt.subplots(1,1,figsize=(12,6))
# plt.title('LogReg ROC AUC Curve');
# for trn, tst in cv.split(X_lr,y_lr):
#     probs = mod.fit(X_lr[trn],y_lr[trn]).predict_proba(X_lr[tst])
#     fpr, tpr, thresholds = roc_curve(y_lr[tst], probs[:, 1])
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     ax.plot(fpr, tpr, lw=1, alpha=0.9, color=cols[i],
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#     i+=1
# ax.plot(np.linspace(0,1,100),np.linspace(0,1,100), 'k:');
# ax.legend();
# plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
