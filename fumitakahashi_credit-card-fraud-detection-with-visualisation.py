import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
df = pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
df.head(5)
df.iloc[-5:,:].head()
df.info()
df.describe().T
df2 = df.copy()
classes = pd.value_counts(df2['default.payment.next.month'], sort = True).sort_index()
classes.plot(kind = 'bar')
plt.title("default.payment.next.month histogram")
plt.show()
print('Number of frauds clients: ', len(df2[df2['default.payment.next.month'] == 1]))
print('Fraud percentage:', round(df2['default.payment.next.month'].value_counts()[1]/len(df2) * 100,2), "%")
sns.set()
fig = plt.figure(figsize = (7,5))
ax = plt.subplot()

sns.distplot(df2["LIMIT_BAL"][df2['default.payment.next.month']==0], bins = 40, label = 'non-Fraud',kde = False)
sns.distplot(df2["LIMIT_BAL"][df2['default.payment.next.month']==1], bins = 40, label = 'Fraud',kde = False)

plt.legend(loc = 'upper right')
plt.title("LIMIT_BAL Histogram")
fig.show()
sns.boxplot(x='default.payment.next.month',y='LIMIT_BAL',data=df2,palette='Set2')
df2 = df.copy()
SEX_dict = {1:"male", 2:"female"}
df2['SEX'] = df2['SEX'].map(SEX_dict)

df_sex = df2.groupby(['SEX', 'default.payment.next.month']).size().unstack(1)
df_sex.plot(kind='bar', stacked = True)
plt.legend(loc = 'upper left')
plt.title("SEX Histogram")
fig.show()

df_sex['Fraud_rate'] = (df_sex[df_sex.columns[1]]/(df_sex[df_sex.columns[0]] + df_sex[df_sex.columns[1]]))
print(df_sex)

EDUCATION_dict = {0:"error", 1:"graduate school", 2:"undergraduate", 3:"high school", 4:"others", 5:"unknown", 6:"unknown"}
df2 = df.copy()

EDUCATION_dict = {0:"error", 1:"graduate school", 2:"undergraduate", 3:"high school", 4:"others", 5:"unknown", 6:"unknown"}
df2['EDUCATION'] = df2['EDUCATION'].map(EDUCATION_dict)
df_ed = df2.groupby(['EDUCATION', 'default.payment.next.month']).size().unstack(1)
df_ed.plot(kind='bar', stacked = True)
plt.legend(loc = 'upper left')
plt.title("EDUCATION Histogram")

fig.show()

df_ed['Fraud_rate'] = (df_ed[df_ed.columns[1]]/(df_ed[df_ed.columns[0]] + df_ed[df_ed.columns[1]]))
print(df_ed)
df2 = df.copy()

MARRIAGE_dict  = {0: 'unknown', 1:"married", 2:"single", 3:"others"}
df2['MARRIAGE'] = df2['MARRIAGE'].map(MARRIAGE_dict)

df_mar = df2.groupby(['MARRIAGE', 'default.payment.next.month']).size().unstack(1)
df_mar.plot(kind='bar', stacked = True)
plt.legend(loc = 'upper left')
plt.title("MARRIAGE Histogram")
fig.show()

df_mar['Fraud_rate'] = (df_mar[df_mar.columns[1]]/(df_mar[df_mar.columns[0]] + df_mar[df_mar.columns[1]]))
print(df_mar)
sns.set()
fig = plt.figure(figsize = (7,5))
ax = plt.subplot()

sns.distplot(df2["AGE"][df2['default.payment.next.month']==0], bins = 40, label = 'non-Fraud',kde = False)
sns.distplot(df2["AGE"][df2['default.payment.next.month']==1], bins = 40, label = 'Fraud',kde = False)

plt.legend(loc = 'upper right')
plt.title("AGE Histogram")
fig.show()
AGE_bin = [20, 30, 40, 50, 60, 70, 80]
AGE_labels  = [ "20s", "30s", "40s", "50s", "60s", "70s"]
df2["AGE"] = pd.cut(df["AGE"], AGE_bin,right=False, labels=AGE_labels)

df_age = df2.groupby(['AGE', 'default.payment.next.month']).size().unstack(1)
df_age.plot(kind='bar', stacked = True)
plt.legend(loc = 'upper left')
plt.title("AGE Bin Histogram")
fig.show()

df_age['Fraud_rate'] = (df_age[df_age.columns[1]]/(df_age[df_age.columns[0]] + df_age[df_age.columns[1]]))
print(df_age)
tab = pd.DataFrame(df2["PAY_0"].value_counts().sort_index(ascending=False))

for i in range(2,7):
    tab["PAY_" + str(i)] = df2["PAY_" + str(i)].value_counts().sort_index(ascending=False)
tab = tab.T
 
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
colors = sns.color_palette("hls", 15)

rows, cols = len(tab), len(tab.columns)
x = range(rows)

    
for i, t in enumerate(tab.columns):
    y = tab.iloc[:, i:cols].sum(axis=1)
    ax.bar(x, y, label=t, color = colors[i])
    
ax.set_xticks(range(rows + 2))
ax.set_xticklabels(tab.index)
ax.legend();
PAY_dict  = {-2:"no pay", -1:"dul pay", 0:"rev pay", 1:"1m del", 2:"2m del",
                                3:"3m del", 4:"4m del", 5:"5m del", 6:"6m del", 7:"8m del", 8:"9m del"}
df2 = df.copy()
from itertools import chain
tab_list = ["PAY_" + str(i) for i in chain(range(0,1),range(2, 7))]
k = 1
fig = plt.figure(figsize=(13,10))
for i in tab_list:
    df2[i] = df2[i].map(PAY_dict)
    tab = pd.crosstab(df2['default.payment.next.month'], df2[i], normalize='index')
    
    ax = fig.add_subplot(2,3, k)
    k = k+1

    rows, cols = len(tab), len(tab.columns)
    x = range(rows)

    for j, t in enumerate(tab.columns):
        y = tab.iloc[:, j:cols].sum(axis=1)
        if((i=='PAY_5')|(i=='PAY_6')):
            ax.bar(x, y, label=t, color = colors[j+1])
        else:
            ax.bar(x, y, label=t, color = colors[j])
    ax.set_xticks(range(rows + 1))
    ax.set_xticklabels(tab.index)
    ax.set_title(i)
    ax.legend();
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(13, 9), dpi=100)


cols = ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
i = 1
for c in cols:
    plt.subplot(2,3, i)
    plt.hist(df2[c][(df2['default.payment.next.month'] == 0)], bins=300, label='non-defalut', color = 'steelblue')
    plt.hist(df2[c][(df2['default.payment.next.month'] == 1)], bins=300, label='default', color = 'orange')
    plt.legend(loc = 'upper right')
    i = i+1
    plt.xlim(-100,50000)
    plt.ylim(0, 20000)
 
    plt.title(c, fontsize=12, fontweight=0 )
#li = ["PAY_AMT" + str(i) for i in range(1, 7)]

fig = plt.figure(figsize=(13,5))
ax  = fig.add_subplot(111)
ax.set_title("PAY_AMT")
ax.boxplot(df[cols].values, labels=cols, patch_artist=True)
ax.set_facecolor
ax.set_ylim(0,600000)
fig = plt.figure(figsize=(13,7))
ax  = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.set_title("PAY_AMT")
ax.boxplot(df2[cols][df2['default.payment.next.month'] == 0].values, labels=cols, patch_artist=True)
ax2.boxplot(df2[cols][df2['default.payment.next.month'] == 1].values, labels=cols, patch_artist=True)
ax.set_ylim(0,20000)
ax2.set_ylim(0,20000)
ax.set_title('Non-default')
ax2.set_title('Default')
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.figure(figsize=(13, 9), dpi=100)


cols2 = ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
i = 1
for c in cols2:
    plt.subplot(2,3, i)
    plt.hist(df2[c][(df2['default.payment.next.month'] == 0)], bins=300, label='non-defalut', color = 'steelblue')
    plt.hist(df2[c][(df2['default.payment.next.month'] == 1)], bins=300, label='default', color = 'orange')
    plt.legend(loc = 'upper right')
    i = i+1
    plt.xlim(-100,200000)
    plt.ylim(0, 8000)
    
    plt.title(c, fontsize=12, fontweight=0 )
fig = plt.figure(figsize=(13,5))
ax  = fig.add_subplot(111)
ax.set_title("BILL_AMT")
ax.boxplot(df2[cols2].values, labels=cols2, patch_artist=True)
ax.set_facecolor
ax.set_ylim(0,2000000)
fig = plt.figure(figsize=(13,7))
ax  = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.set_title("PAY_AMT")
ax.boxplot(df2[cols2][df2['default.payment.next.month'] == 0].values, labels=cols2, patch_artist=True)
ax2.boxplot(df2[cols2][df2['default.payment.next.month'] == 1].values, labels=cols2, patch_artist=True)
ax.set_ylim(0,100000)
ax2.set_ylim(0,100000)
ax.set_title('Non-default')
ax2.set_title('Default')
df_bill_amt = df2[['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'LIMIT_BAL']]
df_bill_amt[cols] = df2[cols]

plt.figure(figsize=(14, 7), dpi=100)
cm = np.corrcoef(df_bill_amt.values.T) 
corr = df_bill_amt.corr()

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_title('BILL_AMT Correlation')
plt.show()
df3 = df.copy()
# EDUCATION
# Merge -4:others to 0:unknown
df3["EDUCATION"] = df["EDUCATION"].apply( lambda x: (x+1) if ((x>0) and (x<4)) else 1 )
# MARRIAGE 
# Mergee unknown:0 to others 3
df3["MARRIAGE"] = df3["MARRIAGE"].apply(lambda x: x if x>0 else 3 )
dffin = df3.copy()
dffin.drop("ID", axis = 1,  inplace = True)
X = dffin.drop("default.payment.next.month", axis = 1,  inplace = False)
y = dffin.loc[:,"default.payment.next.month"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
import sklearn.ensemble
rf = sklearn.ensemble.RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_predrf = rf.predict_proba(X_test)[:,1]
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(14, 7), dpi=100)
feat_importances.plot(kind='barh')
cori = []
for col in df3.columns:
    print(col,'     ', df3[col].dtypes)
    cori.append([col, df3[col].corr(df3['default.payment.next.month'])])
df_cor = pd.DataFrame(sorted(cori, key = lambda x: abs(x[1]), reverse = True), columns = ['feature', 'corr'])

plt.figure(figsize=(10, 7))
plt.barh(df_cor['feature'], np.abs(df_cor['corr']))
plt.title("Correlation with default.payment.next.month")
plt.show()
def plot_roc(fpr, tpr, y_target, y_predicted):
    auc = roc_auc_score(y_target, y_predicted)
    gini = 2*auc-1
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (AUC: {:.3f}, GINI: {:.3f})'.format(auc,gini) )
    plt.legend()
    plt.show()
fpr, tpr, thresholds = roc_curve(Y_test, Y_predrf)
plot_roc(fpr, tpr, Y_test, Y_predrf)
df5 = df3.copy()
# Check how many times clients paid late
# Check whether the client's payment status is constant but delay

dft = df5[cols]
df5['delay_count'] = (dft>=1).sum(axis = 1)

df5['constant_payer'] = (df5[cols].std(axis=1)== 0).astype(int)
df5[df5['delay_count']>0]['constant_payer'] = 0
# Ratio of BILL and LIMIT_BAL
df5['UsedRate6'] = df5.BILL_AMT6 / df5.LIMIT_BAL
df5['UsedRate5'] = df5.BILL_AMT5 / df5.LIMIT_BAL
df5['UsedRate4'] = df5.BILL_AMT4 / df5.LIMIT_BAL
df5['UsedRate3'] = df5.BILL_AMT3 / df5.LIMIT_BAL
df5['UsedRate2'] = df5.BILL_AMT2 / df5.LIMIT_BAL
df5['UsedRate1'] = df5.BILL_AMT1 / df5.LIMIT_BAL
# Cut the upper outliers for BILL_AMT and PAY_AMT
p0 = df5[cols].min()
p98 = df5[cols].quantile(0.98)
df5[cols] = df5[cols].clip(p0, p98, axis = 1)

p0 = df5[cols2].quantile(0.01)
p98 = df5[cols2].quantile(0.98)
df5[cols] = df5[cols2].clip(p0, p98, axis = 1)
p98
dffin = df5.copy()
dffin.drop("ID", axis = 1,  inplace = True)
X = dffin.drop("default.payment.next.month", axis = 1,  inplace = False)
y = dffin.loc[:,"default.payment.next.month"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Normalize the data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
rf = sklearn.ensemble.RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_predrf = rf.predict_proba(X_test)[:,1]
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(14, 7), dpi=100)
feat_importances.plot(kind='barh')
cori = []
for col in dffin.columns:
    print(col,'     ', dffin[col].dtypes)
    cori.append([col, dffin[col].corr(dffin['default.payment.next.month'])])
df_cor = pd.DataFrame(sorted(cori, key = lambda x: abs(x[1]), reverse = True), columns = ['feature', 'corr'])

plt.figure(figsize=(13, 9))
plt.barh(df_cor['feature'], np.abs(df_cor['corr']))
plt.title("Correlation with default.payment.next.month")
plt.show()
fpr, tpr, thresholds = roc_curve(Y_test, Y_predrf)
plot_roc(fpr, tpr, Y_test, Y_predrf)
list(df_cor['feature'])
final_features = [
 'PAY_0',
 'PAY_2',
 'PAY_3',
 'PAY_4',
 'PAY_5',
 'PAY_6',
 'delay_count',
 'LIMIT_BAL',
 'UsedRate6',
 'UsedRate5',
 'UsedRate4',
 'UsedRate3',
 'UsedRate2',
 'UsedRate1',
 'constant_payer',
 'PAY_AMT1',
 'EDUCATION',
 'PAY_AMT2',
 'PAY_AMT4',
 'PAY_AMT3',
 'PAY_AMT5',
 'PAY_AMT6',
 'SEX',
 'MARRIAGE',
 'AGE'
]

X = dffin.loc[:,final_features]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
rf = sklearn.ensemble.RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_predrf = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_predrf)
plot_roc(fpr, tpr, Y_test, Y_predrf)
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(14, 7), dpi=100)
feat_importances.plot(kind='barh')
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn import model_selection
import math
from sklearn.model_selection import GridSearchCV
# grid_param = {
#     'gamma':[7, 9],
#     'eta':[0.6, 0.8],
#     'n_estimators':[100, 1000],
#     'max_depth':[4, 7],
#     'learning_rate':[0.1, 0.01],
#     'eval_metric':['auc'],
#     'object':['binary:logistic'],
#     'subsample': [0.7, 0,9],
    
#     }
# Xgb = XGBClassifier()
# cv = GridSearchCV(Xgb, grid_param, cv = 5, n_jobs =-1,verbose=True)
# cv.fit(X_train, Y_train)
# print(cv.best_params_, cv.best_score_)

Xgbcv = XGBClassifier(gamma = 9, eta= 0.8, learning_rate= 0.01, max_depth=4, n_estimators=1000, subsample= 0.7)
Xgbcv.fit(X_train, Y_train)
Y_pred = Xgbcv.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
