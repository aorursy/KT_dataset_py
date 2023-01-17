import sqlite3
import pandas as pd
import numpy as np
conn = sqlite3.connect("../input/database.sqlite")
query = 'select * from loan where loan_status is not null'
df_all = pd.read_sql(query, conn, index_col=['id'])
df = df_all.copy()
print(df.info())
import matplotlib.pyplot as plt
%matplotlib inline
variables_miss_prcnt = (1-(df.count()/len(df)))*100
variables_miss_prcnt = variables_miss_prcnt[variables_miss_prcnt>0].sort_values()
variables_miss_prcnt.plot(kind='barh', figsize=(15,15), grid=True, colormap='Accent')
plt.title('Variables with most missing data', fontsize=25)
plt.xlabel('Missing values share [%]', fontsize=20)
plt.ylabel('Variable name', fontsize=20)
plt.show()
miss_cutoff=20
variables = list(variables_miss_prcnt[(variables_miss_prcnt>miss_cutoff)].index)
df = df.drop(variables, axis=1, errors='ignore')
print('Number of variables with missing values: '+str(len(variables_miss_prcnt[(variables_miss_prcnt<=miss_cutoff)])))
print(df.info())
variables = [
    'url', # text
    'title',# text, no categories
    'member_id', # id
    'emp_title', # text, no categories
    'index', # id
    'earliest_cr_line', # date
    'last_credit_pull_d', # date
]
df = df.drop(variables, axis=1, errors='ignore')
df.info()
variables = [
    'collection_recovery_fee',
    'collections_12_mths_ex_med',
    'funded_amnt',
    'funded_amnt_inv',
    'issue_d',
    'last_pymnt_amnt',
    'last_pymnt_d',
    'out_prncp',
    'out_prncp_inv',
    'recoveries',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_int',
    'total_rec_late_fee',
    'total_rec_prncp',
    'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal'
]
df = df.drop(variables, axis=1, errors='ignore')
print('Number of variables with missing values: '+str(len(df.columns[1-(df.count()/len(df))>0])))
df.info()
print(df.loan_status.value_counts())
df['default'] = np.nan
df.loc[(df.loan_status=='Fully Paid'),'default'] = 0
df.loc[(df.loan_status=='Charged Off') | 
       (df.loan_status=='Default') | 
       (df.loan_status=='Does not meet the credit policy. Status:Charged Off') | 
       (df.loan_status=='Late (31-120 days)'),'default'] = 1
df = df.dropna(subset=['default'])
print(df.default.value_counts())
print(df.default.value_counts()/len(df)*100)
con_vars = ['loan_amnt','int_rate','installment','annual_inc','dti','revol_bal','revol_util','total_acc','delinq_2yrs']
cat_vars=['home_ownership','emp_length','sub_grade','purpose','term','grade','verification_status','pymnt_plan','zip_code','addr_state'
         ,'inq_last_6mths','open_acc','pub_rec','initial_list_status','policy_code','application_type']
# int_rate
df['int_rate'] = df['int_rate'].apply(lambda x: str(x).replace('%',''))
df.loc[df.int_rate=='None','int_rate'] = np.nan
df['int_rate'] = df['int_rate'].astype('float')
# revol_util
df['revol_util'] = df['revol_util'].apply(lambda x: str(x).replace('%',''))
df.loc[df.revol_util=='None','revol_util'] = np.nan
df['revol_util'] = df['revol_util'].astype('float')
for var in con_vars:
    plt.figure(figsize=(11,5))
    df.loc[(df.default>=0),var].hist()
    df.loc[(df.default==1),var].hist()
    
    plt.title(var, fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(('non-default','default'))
#     plt.savefig('plots/'+var+'.png', bbox_inches='tight')
    plt.show()
# chosen continous variables
con_vars=['loan_amnt','int_rate','installment','annual_inc','dti','revol_util','total_acc','revol_bal','delinq_2yrs']
for var in cat_vars:
    plt.figure(figsize=(11,5))
    df.loc[(df.default>=0),var].value_counts().plot(kind='barh',color='lightblue')
    df.loc[(df.default==1),var].value_counts().plot(kind='barh',color='orange')

    plt.title(var, fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(('non-default','default'))
#     plt.savefig('plots/'+var+'.png', bbox_inches='tight')
    plt.show()
df_cat = df.copy()
# chosen categorical variables
cat_vars=['home_ownership','emp_length','sub_grade','purpose','term','verification_status','addr_state','inq_last_6mths',
          'pub_rec','initial_list_status','open_acc'
         ]
dummy_vars=[]
for var in cat_vars:
    df_cat = pd.get_dummies(df_cat, columns=[var])
    dummy_vars = dummy_vars+list(df_cat.filter(regex='^'+var,axis=1).columns)
df_cat.loc[df_cat.revol_util.isnull(),'revol_util'] = df_cat.revol_util.mean()
df_cat.loc[df_cat.total_acc.isnull(),'total_acc'] = df_cat.total_acc.mean()
df_cat.loc[df_cat.annual_inc.isnull(),'annual_inc'] = df_cat.annual_inc.mean()

df_cat.loc[df_cat.revol_bal.isnull(),'revol_bal'] = df_cat.revol_bal.mean()
df_cat.loc[df_cat.delinq_2yrs.isnull(),'delinq_2yrs'] = df_cat.delinq_2yrs.mean()
from sklearn.model_selection import train_test_split
features = con_vars+dummy_vars
dependent_variable=['default']
X_train, X_test, y_train, y_test = train_test_split(
    df_cat[features], df_cat['default'], test_size=0.3, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
# define classifiers to be tested
classifiers = {}

clf = DecisionTreeClassifier(
)
classifiers['descision_tree'] = clf

clf = DecisionTreeClassifier(
    min_samples_leaf = 200
)
classifiers['descision_tree_adjusted'] = clf

clf = RandomForestClassifier()
classifiers['random_forest'] = clf

clf = RandomForestClassifier(
    min_samples_leaf = 100
)
classifiers['random_forest_adjusted'] = clf

clf = GradientBoostingClassifier()
classifiers['gradient_boosting'] = clf

clf = GradientBoostingClassifier(
    n_estimators = 200
)
classifiers['gradient_boosting_adjusted'] = clf
# fit the model
for i,clf in classifiers.items():
    clf.fit(X_train,y_train)
    print(i)
    print('Accuracy: '+str(clf.score(X_test,y_test)))
# prepare and display ROC curves
for i,clf in classifiers.items():
    print(i)
    print('Accuracy: '+str(clf.score(X_test,y_test)))
    
    # display ROC Curve
    y_true = y_test
    y_score = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr,tpr)
    print('AUC: '+str(roc_auc))

    plt.figure(figsize=(6,6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC '+i)
    plt.legend(loc="lower right")
#     plt.savefig('plots/roc/'+str(i)+'.png', bbox_inches='tight')
    plt.show()