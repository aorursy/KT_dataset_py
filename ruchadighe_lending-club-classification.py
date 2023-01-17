# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_loans=pd.read_csv('../input/LoanStats3a.csv',sep=',', low_memory=False, header=1)
print(df_loans.shape)
df_loans.isnull().sum()
df_loans_copy=df_loans.copy()
df_loans.dropna(axis=1, how='all', inplace= True)
df_loans.dropna(axis=0, how='all', inplace= True)
print(df_loans.shape)
df_loans.isnull().sum()
df_loans['settlement_term'].unique()
df_loans['id'].unique()
df_id= df_loans['id']
df_loans.dropna(thresh=0.1*df_loans.shape[0], axis=1, inplace= True)
print(df_loans.shape)
df_loans.head(2)

df_loans['disbursement_method'].value_counts().plot(kind='bar', title='Disbursement Method')
df_loans.drop(['debt_settlement_flag','disbursement_method','hardship_flag','tax_liens','pub_rec_bankruptcies',
               'delinq_amnt','chargeoff_within_12_mths','acc_now_delinq','application_type','policy_code',
               'collections_12_mths_ex_med','out_prncp_inv','out_prncp','initial_list_status','out_prncp',
               'out_prncp_inv','initial_list_status','pymnt_plan','zip_code','addr_state','emp_title',
               'desc','title','total_pymnt','total_rec_late_fee','recoveries','last_pymnt_amnt'],
              axis=1, inplace=True)

df_loans.dropna(axis=0, how='all', inplace= True)
df_loans.head()
df_loans[df_loans['last_credit_pull_d'].isnull()]
df_loans.dropna(subset=['last_credit_pull_d'], inplace=True)
df_loans['pub_rec'].value_counts()
df_loans.drop(['pub_rec'],axis=1, inplace=True)
df_loans[df_loans['total_acc'].isnull()]
print('median:',df_loans['total_acc'].median())
print(df_loans['total_acc'].describe())
df_loans['total_acc'].fillna(df_loans['total_acc'].median(), inplace=True)

df_loans[['loan_amnt','funded_amnt','funded_amnt_inv']].corr()
df_loans.drop(['funded_amnt','funded_amnt_inv'],axis=1, inplace=True)
df_loans['last_pymnt_d'].fillna(method='ffill',inplace=True)
df_loans['earliest_cr_line'].fillna(method='ffill',inplace=True)
df_loans['inq_last_6mths'].fillna(method='ffill',inplace=True)                
df_loans[df_loans['annual_inc'].isnull()]
print('median:',df_loans['annual_inc'].median())
print(df_loans['annual_inc'].describe())
df_loans['annual_inc'].fillna(df_loans['annual_inc'].median(), inplace=True)
df_loans['delinq_2yrs'].value_counts().plot.bar()
df_loans['delinq_2yrs'].value_counts()
df_loans.drop(['delinq_2yrs'],axis=1, inplace=True)
df_loans['int_rate'] = df_loans['int_rate'].str.rstrip('%').astype('float')/100.0
df_loans['revol_util'] = df_loans['revol_util'].str.rstrip('%').astype('float')/100.0
print('median:',df_loans['revol_util'].median())
print(df_loans['revol_util'].describe())
df_loans['revol_util'].fillna(df_loans['revol_util'].median(), inplace=True)
print('median:',df_loans['open_acc'].median())
print(df_loans['open_acc'].describe())
df_loans['open_acc'].fillna(df_loans['open_acc'].median(), inplace=True)
df_loans['emp_length'].value_counts().plot.bar()
df_loans['emp_length'].fillna(method='ffill',inplace=True)
df_loans.shape
corr=df_loans.corr()

cmap = sns.diverging_palette(5, 250, as_cmap=True)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)
df_loans.corr()
df_loans.drop(['total_pymnt_inv','total_rec_prncp','total_rec_int',
               'collection_recovery_fee','total_acc','installment'],axis=1, inplace=True)
df_loans.drop(['mths_since_last_delinq','issue_d','last_pymnt_d','earliest_cr_line',
               'sub_grade','last_credit_pull_d'],axis=1, inplace=True)   
df_loans.shape

df_loans['loan_status'] = df_loans['loan_status'].str.replace(r'(^.*Fully Paid.*$)', '1')
df_loans['loan_status'] = df_loans['loan_status'].str.replace(r'(^.*Charged Off.*$)', '0')
df_loans['inq_last_6mths'].value_counts().plot.bar()
sns.distplot(df_loans['revol_util'], kde=False,fit=stats.gamma)
sns.boxplot(df_loans['loan_amnt'])
sns.distplot(df_loans['loan_amnt'], kde=False,fit=stats.gamma)
sns.distplot(df_loans['dti'], kde=False,fit=stats.gamma)
sns.distplot(df_loans['int_rate'], kde=False,fit=stats.gamma)
sns.distplot(df_loans['inq_last_6mths'], kde=False,fit=stats.gamma)
sns.distplot(df_loans['open_acc'], kde=False,fit=stats.gamma)
sns.distplot(df_loans['revol_bal'], kde=False,fit=stats.gamma)
sns.distplot(df_loans['revol_util'], kde=False,fit=stats.gamma)

df_loans['term'] = df_loans['term'].astype('category')
df_loans['grade'] = df_loans['grade'].astype('category')
df_loans['emp_length'] = df_loans['emp_length'].astype('category')
df_loans['home_ownership'] = df_loans['home_ownership'].astype('category')
df_loans['verification_status'] = df_loans['verification_status'].astype('category')
df_loans['purpose'] = df_loans['purpose'].astype('category')
#df_loans['loan_status'] = df_loans['loan_status'].astype('category')
term= pd.get_dummies(df_loans['term'],drop_first=True)
df_loans= pd.concat([df_loans,term],axis=1)

grade= pd.get_dummies(df_loans['grade'],drop_first=True)
df_loans= pd.concat([df_loans,grade],axis=1)

emp_length= pd.get_dummies(df_loans['emp_length'],drop_first=True)
df_loans= pd.concat([df_loans,emp_length],axis=1)

owner= pd.get_dummies(df_loans['home_ownership'],drop_first=True)
df_loans= pd.concat([df_loans,owner],axis=1)

verification= pd.get_dummies(df_loans['verification_status'],drop_first=True)
df_loans= pd.concat([df_loans,verification],axis=1)

purpose= pd.get_dummies(df_loans['purpose'],drop_first=True)
df_loans= pd.concat([df_loans,purpose],axis=1)

df_loans.drop(['term','grade','emp_length','home_ownership',
               'verification_status','purpose'],axis=1, inplace=True)       


print(df_loans.shape)
df_loans= df_loans[((df_loans['dti'] - df_loans['dti'].mean()) / df_loans['dti'].std()).abs() < 3]
df_loans= df_loans[((df_loans['int_rate'] - df_loans['int_rate'].mean()) / df_loans['int_rate'].std()).abs() < 3]
df_loans= df_loans[((df_loans['open_acc'] - df_loans['open_acc'].mean()) / df_loans['open_acc'].std()).abs() < 3]
df_loans= df_loans[((df_loans['revol_bal'] - df_loans['revol_bal'].mean()) / df_loans['revol_bal'].std()).abs() < 3]
df_loans= df_loans[((df_loans['revol_util'] - df_loans['revol_util'].mean()) / df_loans['revol_util'].std()).abs() < 3]
df_loans= df_loans[((df_loans['loan_amnt'] - df_loans['loan_amnt'].mean()) / df_loans['loan_amnt'].std()).abs() < 3]
df_loans= df_loans[((df_loans['annual_inc'] - df_loans['annual_inc'].mean()) / df_loans['annual_inc'].std()).abs() < 3]
print(df_loans.shape)
print('loan_amnt_mean: ',df_loans['loan_amnt'].median())
print('loan_amnt_std: ',df_loans['loan_amnt'].std())

print('annual_inc_mean: ',df_loans['annual_inc'].mean())
print('annual_inc_std: ',df_loans['annual_inc'].std())

print('dti_mean: ',df_loans['dti'].mean())
print('dti_std: ',df_loans['dti'].std())

print('dti_mean: ',df_loans['dti'].mean())
print('dti_std: ',df_loans['dti'].std())

print('int_rate_mean: ',df_loans['int_rate'].mean())
print('int_rate_std: ',df_loans['int_rate'].std())

print('open_acc_mean: ',df_loans['open_acc'].mean())
print('open_acc_std: ',df_loans['open_acc'].std())

print('revol_bal_mean: ',df_loans['revol_bal'].mean())
print('revol_bal_std: ',df_loans['revol_bal'].std())

print('revol_util_mean: ',df_loans['revol_util'].mean())
print('revol_util_std: ',df_loans['revol_util'].std())


df_loans.columns.values
df_loans['loan_amnt']= (df_loans['loan_amnt']-df_loans['loan_amnt'].mean())/df_loans['loan_amnt'].std()
df_loans['annual_inc']= (df_loans['annual_inc']-df_loans['annual_inc'].mean())/df_loans['annual_inc'].std()
df_loans['dti']= (df_loans['dti']-df_loans['dti'].mean())/df_loans['dti'].std()
df_loans['int_rate']= (df_loans['int_rate']-df_loans['int_rate'].mean())/df_loans['int_rate'].std()
df_loans['open_acc']= (df_loans['open_acc']-df_loans['open_acc'].mean())/df_loans['open_acc'].std()
df_loans['revol_bal']= (df_loans['revol_bal']-df_loans['revol_bal'].mean())/df_loans['revol_bal'].std()
df_loans['revol_util']= (df_loans['revol_util']-df_loans['revol_util'].mean())/df_loans['revol_util'].std()
df_loans.head()
known, unknown = train_test_split(df_loans, test_size=0.05)
df_status = known['loan_status'].astype(int)
df_independent=known[df_loans.columns.difference(['loan_status'])]
x_train, x_test, y_train, y_test = train_test_split(df_independent, df_status, test_size=0.3, random_state=30)
df_loans.isnull().sum()
df_loans.columns.values
#Create a svm Classifier
clf = LinearSVC(random_state=0) # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
print('\nConfusion Matrix\n',confusion_matrix)
# Model Precision: what percentage of positive tuples are labeled as such?
#print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
#print("Recall:",metrics.recall_score(y_test, y_pred))
print('\nclassification_report\n',classification_report(y_test, y_pred))


logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print('Accuracy : {:.2f}'.format(logreg.score(x_test, y_test)))

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, x_train, y_train, cv=kfold, scoring=scoring)
#print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix\n',confusion_matrix)

print('\nclassification_report\n',classification_report(y_test, y_pred))
logreg.get_params()
#logreg.predict_proba(x_test)

#logreg.densify()

#logreg.decision_function(x_test)

#logreg.AIC
logreg.coef_
random_clf=RandomForestClassifier(n_estimators=100)
random_clf.fit(x_train,y_train)

y_pred=random_clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix\n',confusion_matrix)

print('\nclassification_report\n',classification_report(y_test, y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred,pos_label=1)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
#feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
#feature_imp
from sklearn.externals import joblib
joblib.dump(random_clf, 'ques11.pkl')
