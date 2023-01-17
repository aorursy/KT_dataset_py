import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(font_scale=1.6)

from sklearn.preprocessing import StandardScaler
data=pd.read_csv('../input/loan.csv',parse_dates=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
data.shape
data.head()
pd.value_counts(data.loan_status).to_frame().reset_index()
data = data[data.loan_status != 'Fully Paid']
data = data[data.loan_status != 'Does not meet the credit policy. Status:Fully Paid']
data['rating'] = np.where((data.loan_status != 'Current'), 1, 0)
pd.value_counts(data.rating).to_frame()
print ('Bad Loan Ratio: %.2f%%'  % (data.rating.sum()/len(data)*100))
data.info()
pd.value_counts(data.title).to_frame()
pd.value_counts(data.purpose).to_frame()
pd.value_counts(data.application_type).to_frame()
app_type={'INDIVIDUAL':0,'JOINT':1}
data.application_type.replace(app_type,inplace=True)
pd.value_counts(data.term).to_frame()
term={' 36 months':36,' 60 months':60}
data.term.replace(term,inplace=True)
pd.value_counts(data.grade).to_frame()
grade=data.grade.unique()
grade.sort()
grade
for x,e in enumerate(grade):
    data.grade.replace(to_replace=e,value=x,inplace=True)
data.grade.unique()
pd.value_counts(data.sub_grade).to_frame()
sub_grade=data.sub_grade.unique()
sub_grade.sort()
sub_grade
for x,e in enumerate(sub_grade):
    data.sub_grade.replace(to_replace=e,value=x,inplace=True)

data.sub_grade.unique()
pd.value_counts(data.emp_title).to_frame()
pd.value_counts(data.emp_length).to_frame()
emp_len={'n/a':0,'< 1 year':1,'1 year':2,'2 years':3,'3 years':4,'4 years':5,'5 years':6,'6 years':7,'7 years':8,'8 years':9,'9 years':10,'10+ years':11}
data.emp_length.replace(emp_len,inplace=True)
data.emp_length=data.emp_length.replace(np.nan,0)
data.emp_length.unique()
pd.value_counts(data.home_ownership).to_frame()
pd.value_counts(data.verification_status).to_frame()
pd.value_counts(data.pymnt_plan).to_frame()
pd.value_counts(data.zip_code).to_frame()
pd.value_counts(data.addr_state).to_frame()
pd.value_counts(data.initial_list_status).to_frame()
int_status={'w':0,'f':1}
data.initial_list_status.replace(int_status,inplace=True)
pd.value_counts(data.policy_code).to_frame()
pd.value_counts(data.recoveries).to_frame()
data['recovery'] = np.where((data.recoveries != 0.00), 1, 0)
pd.value_counts(data.collection_recovery_fee).to_frame()
data.issue_d=pd.to_datetime(data.issue_d)
earliest_cr_line=pd.to_datetime(data.earliest_cr_line)
data.earliest_cr_line=earliest_cr_line.dt.year
data.last_pymnt_d=pd.to_datetime(data.last_pymnt_d)
data.next_pymnt_d=pd.to_datetime(data.next_pymnt_d)
data.last_credit_pull_d=pd.to_datetime(data.last_credit_pull_d)
data.drop(['id','member_id','desc','loan_status','url', 'title','collection_recovery_fee','recoveries','policy_code','zip_code','emp_title','pymnt_plan'],axis=1,inplace=True)
data.head(10)
def meta (dataframe):
    metadata = []
    for f in data.columns:
    
        # Counting null values
        null = data[f].isnull().sum()
    
        # Defining the data type 
        dtype = data[f].dtype
    
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'nulls':null,
            'dtype': dtype
        }
        metadata.append(f_dict)

    meta = pd.DataFrame(metadata, columns=['varname','nulls', 'dtype'])
    meta.set_index('varname', inplace=True)
    meta=meta.sort_values(by=['nulls'],ascending=False)
    return meta
meta(data)
data.dti_joint=data.dti_joint.replace(np.nan,0)
data.annual_inc_joint=data.annual_inc_joint.replace(np.nan,0)
data.verification_status_joint=data.verification_status_joint.replace(np.nan,'None')
data.loc[(data.open_acc_6m.isnull())].info()
variables1=['open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m','collections_12_mths_ex_med']

for e in variables1:
    data[e]=data[e].replace(np.nan,0)
    
meta(data)
pd.value_counts(data.mths_since_last_record).unique()
pd.value_counts(data.mths_since_last_major_derog).unique()
pd.value_counts(data.mths_since_last_delinq).unique()
data.loc[(data.mths_since_last_delinq.notnull()),'delinq']=1
data.loc[(data.mths_since_last_delinq.isnull()),'delinq']=0

data.loc[(data.mths_since_last_major_derog.notnull()),'derog']=1
data.loc[(data.mths_since_last_major_derog.isnull()),'derog']=0

data.loc[(data.mths_since_last_record.notnull()),'public_record']=1
data.loc[(data.mths_since_last_record.isnull()),'public_record']=0

data.drop(['mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record'],axis=1,inplace=True)

meta(data)
data.loc[(data.tot_coll_amt.isnull())].info()
variables2=['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

for e in variables2:
    data[e]=data[e].replace(np.nan,0)
    
meta(data)
data.loc[(data.revol_util.isnull())].head(10)
pd.value_counts(data.revol_util).to_frame()
data.revol_util=data.revol_util.replace(np.nan,0)
    
meta(data)
pd.value_counts(data.last_pymnt_d).to_frame()
late=data.loc[(data.last_pymnt_d=='2015-08-01')|(data.last_pymnt_d=='2015-09-01')|(data.last_pymnt_d=='2015-05-01')|(data.last_pymnt_d=='2015-06-01')]
pd.value_counts(late.rating).to_frame()
data.loc[(data.last_pymnt_d.notnull()),'pymnt_received']=1
data.loc[(data.last_pymnt_d.isnull()),'pymnt_received']=0
data.drop(['last_pymnt_d','issue_d','last_credit_pull_d','next_pymnt_d'],axis=1,inplace=True)

meta(data)
variables3=['acc_now_delinq', 'open_acc', 'total_acc','pub_rec','delinq_2yrs','inq_last_6mths','earliest_cr_line']

for e in variables3:
    data[e]=data[e].replace(np.nan,data[e].mode()[0])
    
meta(data)
data.head()
data.describe()
X=data.drop(['rating'],axis=1,inplace=False)
y=data.rating
num_cols = X.columns[X.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
num_cols
scaler=StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X.head()
X=pd.get_dummies(X,drop_first=True)
X.head()
X.shape
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report,accuracy_score 
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from copy import deepcopy

def cross_validate_repeated_undersampling_full(X, Y, model, n_estimators=3, cv=StratifiedKFold(5,random_state=1)):
    
    preds = []
    true_labels = []
        
    for train_index, test_index in cv.split(X,Y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
        scores = np.zeros((n_estimators,len(y_test)))
        for i in range(n_estimators):
            num1 = len(y_train[y_train==1])
            ind0 = np.random.choice(y_train.index[y_train==0], num1) 
            ind1 = y_train.index[y_train==1] 
            ind_final = np.r_[ind0, ind1]
            X_train_subsample = X_train.loc[ind_final]
            y_train_subsample = y_train.loc[ind_final]

            clf = deepcopy(model)
            clf.fit(X_train_subsample,y_train_subsample)  
            
            probs = clf.predict_proba(X_test)[:,1]
            scores[i,:] = probs

        preds_final = scores.mean(0) 
        preds.extend(preds_final)
        preds_labels=[round(x) for x in preds]
        
        true_labels.extend(y_test)
        
    cnf_matrix = confusion_matrix(true_labels,preds_labels)
    np.set_printoptions(precision=2)

    print("Accuracy score in the testing dataset: ", accuracy_score(true_labels,preds_labels))
    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
        
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                    , classes=class_names
                    , title='Confusion matrix')
    plt.show()
        
    print("ROC AUC score in the testing dataset: ", roc_auc_score(true_labels,preds))
        
    fpr, tpr, _ = roc_curve(true_labels,preds)
    roc_auc = auc(fpr, tpr)
        
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return
models=[['LogisticRegression',LogisticRegression()],['RandomForest',RandomForestClassifier()],['NaiveBayes',GaussianNB()],['LDA',LinearDiscriminantAnalysis()],['QDA',QuadraticDiscriminantAnalysis()]]
for e in models:
    print ("Testing:", e[0])
    cross_validate_repeated_undersampling_full(X, y, e[1])