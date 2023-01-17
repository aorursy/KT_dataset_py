import pandas as pd
%matplotlib inline
import numpy as np
from sklearn.preprocessing import Imputer
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import operator
from xgboost import XGBClassifier
df = pd.read_csv('../input/loan.csv')
df.shape
df.head()
df.loan_status.value_counts()
df = df[(df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off')]
print(np.count_nonzero(df['id'].unique()))
print(np.count_nonzero(df['member_id'].unique()))
print(np.count_nonzero(df['pymnt_plan'].unique()))
df = df[df['pymnt_plan'] == 'n']
print(np.count_nonzero(df['application_type'].unique()))
df = df[df['application_type'] == 'INDIVIDUAL']
df.iloc[:, 20:30]
df[['out_prncp','out_prncp_inv', 'loan_status']][(df['out_prncp']>0) | (df['out_prncp_inv']>0)]
df[['next_pymnt_d', 'loan_status']][df.next_pymnt_d.notnull()]
df[['policy_code', 'annual_inc_joint', 'dti_joint']].describe()
df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp','out_prncp_inv', 'pymnt_plan', 
                       'initial_list_status', 'member_id', 'id', 'url', 'application_type',
                       'grade', 'annual_inc_joint', 'dti_joint'])
df1.shape
terms = []
for row in df1.term:
    terms.append(re.findall('\d+', row)[0])
df1.term = terms
emp_lengths = []
for row in df1.emp_length:
    if pd.isnull(row) == False:
        emp_lengths.append(re.findall('\d+', row)[0])
    else:
        emp_lengths.append(row)
df1.emp_length = emp_lengths
df1.iloc[:, 60:70]
for col in df1.columns:
    nan_count = np.sum(pd.isnull(df1[col]))
    if nan_count != 0:
        print(col, " : ", np.sum(pd.isnull(df1[col])))
df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m',
                       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 
                      'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 
                      'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m' ])
df1.shape
for col in df1.columns:
    nan_count = np.sum(pd.isnull(df1[col]))
    if nan_count != 0:
        print(col, " : ", np.sum(pd.isnull(df1[col])))
df1 = df1.drop(columns = ['mths_since_last_major_derog'])
df1.tot_coll_amt = df1.tot_coll_amt.replace(np.nan, 0)
imp = Imputer(strategy='median')
df1.total_rev_hi_lim = imp.fit_transform(df1.total_rev_hi_lim.reshape(-1, 1))
df1.tot_cur_bal = df1.tot_cur_bal.replace(np.nan, 0)
imp = Imputer(strategy='most_frequent')
df1.collections_12_mths_ex_med = imp.fit_transform(df1.collections_12_mths_ex_med.reshape(-1, 1))
df1['mths_since_last_delinq_nan'] =  np.isnan(df1.mths_since_last_delinq)*1
imp = Imputer(strategy='most_frequent')
msld = imp.fit_transform(df1.mths_since_last_delinq.values.reshape(-1, 1))
df1.mths_since_last_delinq = msld
df1.mths_since_last_record.hist()
df1['mths_since_last_record_nan'] =  np.isnan(df1.mths_since_last_record)*1
imp = Imputer(strategy='median')
mslr = imp.fit_transform(df1.mths_since_last_record.values.reshape(-1, 1))
df1.mths_since_last_record = mslr
df1['revol_util_nan'] =  pd.isnull(df1.revol_util)*1
imp = Imputer(strategy='mean')
df1.revol_util = imp.fit_transform(df1.revol_util.values.reshape(-1, 1))
df1['emp_length_nan'] =  pd.isnull(df1.emp_length)*1
imp = Imputer(strategy='median')
df1.emp_length = imp.fit_transform(df1.emp_length.values.reshape(-1, 1))
for col in df1.columns:
    nan_count = np.sum(pd.isnull(df1[col]))
    if nan_count != 0:
        print(col, " : ", np.sum(pd.isnull(df1[col])))
#все категориальные признаки
col_cat = '''
sub_grade
home_ownership
verification_status
purpose
zip_code
addr_state
'''.split()
lbl_enc = LabelEncoder()
for x in col_cat:
    df1[x+'_old'] = df[x]
    df1[x] = lbl_enc.fit_transform(df1[x])
df1['text'] = df1.emp_title + ' ' + df1.title + ' ' + df1.desc
df1['text'] = df1['text'].fillna('nan')
tfidf = TfidfVectorizer()
df_text = tfidf.fit_transform(df1['text'])
df1.issue_d = pd.to_datetime(df1.issue_d, format='%b-%Y')
df1['issue_d_year'] =df1.issue_d.dt.year
df1['issue_d_month'] =df1.issue_d.dt.month
df1.earliest_cr_line = pd.to_datetime(df1.earliest_cr_line, format='%b-%Y')
df1['earliest_cr_line_year'] =df1.earliest_cr_line.dt.year
df1['earliest_cr_line_month'] =df1.earliest_cr_line.dt.month
#real features
col_int_float2 = '''loan_amnt
term
int_rate
installment
emp_length
annual_inc
dti
delinq_2yrs
inq_last_6mths
mths_since_last_delinq
mths_since_last_record
open_acc
pub_rec
revol_bal
revol_util
total_acc
collections_12_mths_ex_med
acc_now_delinq
tot_coll_amt
tot_cur_bal
total_rev_hi_lim
mths_since_last_delinq_nan
mths_since_last_record_nan
revol_util_nan
emp_length_nan
issue_d_year
issue_d_month
earliest_cr_line_year
earliest_cr_line_month
'''.split()
df1[col_int_float2 + col_cat].head()
df1['term'] = df1.term.astype(str).astype(int)
df1['int_rate'] = df1.int_rate.astype(str).astype(float)
df2 = df1[(df1['loan_status'] == 'Fully Paid') | (df1['loan_status'] =='Charged Off') ]
targets = []
for row in df2.loan_status:
    if row == 'Fully Paid':
        targets.append(1)
    else:
        targets.append(0)
df2['target'] = targets
X_train, X_test, y_train, y_test = train_test_split(df2[col_int_float2 + col_cat], df2['target'], 
                                                    test_size=0.3, random_state=42, stratify=df2['target'])
sum(targets)/len(targets)
def classify(est, x, y):

    est.fit(x, y)

    y2 = est.predict_proba(X_test)
    y1 = est.predict(X_test)

    print("Accuracy: ", metrics.accuracy_score(y_test, y1))
    print("Area under the ROC curve: ", metrics.roc_auc_score(y_test, y2[:, 1]))

    print("F-metric: ", metrics.f1_score(y_test, y1))
    print(" ")
    print("Classification report:")
    print(metrics.classification_report(y_test, y1))
    print(" ")
    print("Evaluation by cross-validation:")
    print(cross_val_score(est, x, y))
    
    return est, y1, y2[:, 1]
xgb0, y_pred_b, y_pred2_b = classify(XGBClassifier(), X_train, y_train)
sum(y_pred_b)/len(y_pred_b)
X_train[y_train==1].shape
X_train[y_train==0].shape
145404/31673
64+32+32
X_train2 = X_train.drop(X_train[y_train==1].iloc[32000:].index)
y_train2 = y_train[X_train2.index]

X_train3 = X_train.drop(X_train[y_train==1].iloc[0:32000].index)
X_train3 = X_train3.drop(X_train3[y_train==1].iloc[32000:].index)
y_train3 = y_train[X_train3.index]

X_train4 = X_train.drop(X_train[y_train==1].iloc[0:64000].index)
X_train4 = X_train4.drop(X_train4[y_train==1].iloc[32000:].index)
y_train4 = y_train[X_train4.index]

X_train5 = X_train.drop(X_train[y_train==1].iloc[0:96000].index)
X_train5 = X_train5.drop(X_train5[y_train==1].iloc[32000:].index)
y_train5 = y_train[X_train5.index]

X_train6 = X_train.drop(X_train[y_train==1].iloc[0:128000].index)
X_train6 = X_train6.drop(X_train6[y_train==1].iloc[32000:].index)
y_train6 = y_train[X_train6.index]
xgb = XGBClassifier(n_estimators=47, learning_rate=0.015)
xgb1, y_pred, y_pred2 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train2, y_train2)
xgb2, y_pred_3, y_pred2_3 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train3, y_train3)
xgb3, y_pred_4, y_pred2_4 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train4, y_train4)
xgb4, y_pred_5, y_pred2_5 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train5, y_train5)
xgb5, y_pred_6, y_pred2_6 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train6, y_train6)
y_avg = (y_pred + y_pred_3 + y_pred_4 + y_pred_5 + y_pred_6)/5
y_avg = (y_avg>0.3)*1
print("Accuracy: ", metrics.accuracy_score(y_test, y_avg))
print("F-metric: ", metrics.f1_score(y_test, y_avg))

print(" ")
print("Classification report:")
print(metrics.classification_report(y_test, y_avg))
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
knc, y_p, y_p2 = classify(KNeighborsClassifier(), X_train2, y_train2)
logit, y_p, y_p2 = classify(LogisticRegression(), X_train2, y_train2)
bnb, y_p, y_p2 = classify(BernoulliNB(), X_train2, y_train2)
dtc, y_p, y_p2 = classify(DecisionTreeClassifier(), X_train2, y_train2)
xg, y_p, y_p2 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train2, y_train2)
def feat_importance(estimator):
    feature_importance = {}
    for index, name in enumerate(df2[col_int_float2 + col_cat].columns):
        feature_importance[name] = estimator.feature_importances_[index]

    feature_importance = {k: v for k, v in feature_importance.items()}
    sorted_x = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse = True)
    
    return sorted_x
feat1 = feat_importance(xgb1)
feat1[:12]
feat2 = feat_importance(xgb2)
feat2[:12]
feat3 = feat_importance(xgb3)
feat3[:12]
feat4 = feat_importance(xgb4)
feat4[:12]
feat5 = feat_importance(xgb5)
feat5[:12]
col_xgb = '''annual_inc
int_rate
term
dti
issue_d_year
'''.split()
X_train, X_test, y_train, y_test = train_test_split(df2[col_xgb], df2['target'], 
                                                    test_size=0.3, random_state=42, stratify=df2['target'])
X_train2 = X_train.drop(X_train[y_train==1].iloc[32000:].index)
y_train2 = y_train[X_train2.index]

X_train3 = X_train.drop(X_train[y_train==1].iloc[0:32000].index)
X_train3 = X_train3.drop(X_train3[y_train==1].iloc[32000:].index)
y_train3 = y_train[X_train3.index]

X_train4 = X_train.drop(X_train[y_train==1].iloc[0:64000].index)
X_train4 = X_train4.drop(X_train4[y_train==1].iloc[32000:].index)
y_train4 = y_train[X_train4.index]

X_train5 = X_train.drop(X_train[y_train==1].iloc[0:96000].index)
X_train5 = X_train5.drop(X_train5[y_train==1].iloc[32000:].index)
y_train5 = y_train[X_train5.index]

X_train6 = X_train.drop(X_train[y_train==1].iloc[0:128000].index)
X_train6 = X_train6.drop(X_train6[y_train==1].iloc[32000:].index)
y_train6 = y_train[X_train6.index]
xgb1, y_pred, y_pred2 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train2, y_train2)
xgb2, y_pred_3, y_pred2_3 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train3, y_train3)
xgb3, y_pred_4, y_pred2_4 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train4, y_train4)
xgb4, y_pred_5, y_pred2_5 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train5, y_train5)
xgb5, y_pred_6, y_pred2_6 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train6, y_train6)
y_avg = (y_pred + y_pred_3 + y_pred_4 + y_pred_5 + y_pred_6)/5
y_avg = (y_avg>0.3)*1
print("Accuracy: ", metrics.accuracy_score(y_test, y_avg))
print("F-metric: ", metrics.f1_score(y_test, y_avg))

print(" ")
print("Classification report:")
print(metrics.classification_report(y_test, y_avg))
col_xgb = '''annual_inc
int_rate
term
dti
issue_d_year
'''.split()
y_avg2 = (y_pred2 + y_pred2_3 + y_pred2_4 + y_pred2_5 + y_pred2_6)/5
print("Area under the ROC curve: ", metrics.roc_auc_score(y_test, y_avg2))
ax = df2[['annual_inc', 'target']][df2.annual_inc < 100000].boxplot(by='target', figsize=(6, 5), vert=False )
ax.set_yticklabels(['Charged Off', 'Fully Paid'])
ax.set_title('Annual income of the borrower')
ax = df2[['int_rate']][df2.target==1].hist(bins=20, normed=True, alpha=0.8, figsize=(6,4), label = u'Fully Paid')
df2[['int_rate']][df2.target==0].hist(ax=ax, bins=20, normed=True, alpha=0.5, label = u'Charged off')
plt.title("Interest rate of the loan")
plt.xlabel('Interest rate')
plt.legend()
ax = df2['dti'][df2.target==1].hist(bins=20, normed=True, alpha=0.8, figsize=(8,4), label = u'Fully Paid')
df2['dti'][df2.target==0].hist(ax=ax, bins=20, normed=True, alpha=0.5, label = u'Charged off')
plt.title("DTI")
plt.legend(loc='best', frameon=False)
ax = df2['term'][df2.target==1].hist(bins=2, normed=True, alpha=0.8, figsize=(4,4), label = u'Fully Paid')
df2['term'][df2.target==0].hist(ax=ax, bins=2, normed=True, alpha=0.5, label = u'Charged off')
ax.set_xticklabels(['', '', '36 month', '', '', '60 month'])
plt.title("Term of the loan")
plt.legend(loc='best', frameon=False)
ax.tick_params(axis=x, pad=10)
ax = df2['issue_d_year'][df2.target==1].hist(bins=20, normed=True, alpha=0.8, figsize=(6,4),  label = u'Fully Paid')
df2['issue_d_year'][df2.target==0].hist(ax=ax, bins=20, normed=True, alpha=0.5, label = u'Charged off')
plt.title("The year which the loan was funded")
plt.legend(loc='best', frameon=False)