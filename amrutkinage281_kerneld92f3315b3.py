import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataset=pd.read_csv('../input/loan.csv', low_memory=False)
dataset.head()
var_null_pc=dataset.isnull().sum(axis=0).sort_values(ascending=False)/float(len(dataset))
var_null_pc[var_null_pc>0.75]
dataset.drop( var_null_pc[ var_null_pc > 0.75 ].index, axis = 1, inplace = True ) 
dataset.dropna( axis = 0, thresh = 30, inplace = True )
vars_to_be_removed=['id','member_id','policy_code','zip_code','application_type']

dataset.drop( vars_to_be_removed , axis = 1, inplace = True )
dataset.head()
dataset.select_dtypes(include=[np.object]).isnull().sum()
for value in ('emp_title','emp_length','title','last_pymnt_d','next_pymnt_d','last_credit_pull_d','earliest_cr_line'):
    dataset[value].fillna(dataset[value].mode()[0],inplace=True)

dataset.select_dtypes(include=[np.object]).isnull().sum()
dataset.select_dtypes(include=[np.number]).isnull().sum()
import math
for value in ('annual_inc','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','open_acc','pub_rec','revol_util','total_acc',
             'collections_12_mths_ex_med','acc_now_delinq','tot_coll_amt','tot_cur_bal','total_rev_hi_lim'):
   dataset[value].fillna(math.floor(dataset[value].mean()),inplace=True) 
dataset.select_dtypes(include=[np.number]).isnull().sum()
dataset.drop(['url'],axis=1,inplace=True)
dataset['loan_status'].value_counts()
dataset = dataset[~dataset['loan_status'].isin(['Issued',
                                 'Does not meet the credit policy. Status:Fully Paid',
                                 'Does not meet the credit policy. Status:Charged Off'
                                ])]
def CreateDefault(loan_status):
    if loan_status in ['Current', 'Fully Paid', 'In Grace Period']:
        return 0
    else:
        return 1 
dataset['Default'] = dataset['loan_status'].apply(lambda x: CreateDefault(x))
dataset['Default']
dataset['emp_length'] = dataset['emp_length'].str.replace('+','')
dataset['emp_length'] = dataset['emp_length'].str.replace('<','')
dataset['emp_length'].value_counts()
dataset['last_pymnt_d'] = pd.to_numeric(dataset['last_pymnt_d'].str[-4:], errors='coerce').round(0)
dataset['issue_d'] = pd.to_numeric(dataset['issue_d'].str[-4:], errors='coerce').round(0)
dataset['last_credit_pull_d'] = pd.to_numeric(dataset['last_credit_pull_d'].str[-4:], errors='coerce').round(0)
dataset.drop(['next_pymnt_d'],axis=1,inplace=True)
dataset.head()
objects=dataset['emp_length'].unique()
print(objects)
import numpy as np
y_pos = np.arange(len(objects))

new_df=dataset.groupby("emp_length").sum()
#print(new_df)
total_attacks=new_df["Default"]
plt.bar(y_pos, total_attacks, alpha=1.0)
#plt.barh(y_pos, total_attacks)
plt.xticks(y_pos, objects,rotation=90)
plt.xlabel("emp_length")
plt.ylabel('Default')
plt.title('Defaulters vs emp_length')
plt.show()
objects=dataset['term'].unique()
print(objects)
import numpy as np
y_pos = np.arange(len(objects))

new_df=dataset.groupby("term").sum()
#print(new_df)
total_attacks=new_df["Default"]
plt.bar(y_pos, total_attacks, alpha=1.0)
#plt.barh(y_pos, total_attacks)
plt.xticks(y_pos, objects,rotation=360)
plt.xlabel("term")
plt.ylabel('Default')
plt.title('Defaulters vs term')
plt.show()
dataset['home_ownership'].unique()
dataset['home_ownership'].value_counts()
objects=dataset['home_ownership'].unique()
print(objects)
import numpy as np
y_pos = np.arange(len(objects))

new_df=dataset.groupby("home_ownership").sum()
#print(new_df)
total_attacks=new_df["Default"]
plt.bar(y_pos, total_attacks, alpha=1.0)
#plt.barh(y_pos, total_attacks)
plt.xticks(y_pos, objects,rotation=360)
plt.xlabel("home_ownership")
plt.ylabel('Default')
plt.title('Defaulters vs home_ownership')
plt.show()
dataset.dtypes
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3))

sns.distplot(dataset[dataset['Default'] == 0]['int_rate'], bins=30, ax=ax1, kde=False)
sns.distplot(dataset[dataset['Default'] == 1]['int_rate'], bins=30, ax=ax2, kde=False)

ax1.set_title('No Default')
ax2.set_title('Default')

ax1.set_xbound(lower=0)
ax2.set_xbound(lower=0)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3))

sns.distplot(dataset[dataset['Default'] == 0]['loan_amnt'], bins=40, ax=ax1, kde=False)
sns.distplot(dataset[dataset['Default'] == 1]['loan_amnt'], bins=40, ax=ax2, kde=False)

ax1.set_title('No Default')
ax2.set_title('Default')

ax1.set_xbound(lower=0)
ax2.set_xbound(lower=0)

plt.tight_layout()
plt.show()
colname = ['term', 'grade','sub_grade', 'home_ownership','emp_length','emp_title', 'verification_status','loan_status','pymnt_plan',
           'purpose','addr_state','earliest_cr_line','title','initial_list_status']
colname
from sklearn import preprocessing
le={}
#iterate through columns and assigning it with labels 
for x in colname:
    le[x]=preprocessing.LabelEncoder()
# iterate through colname and comparing it with the labels assigned to them in le and repalcing it with label value    
for x in colname:
        dataset[x]=le[x].fit_transform(dataset[x])
dataset.dtypes
loan_correlation = dataset.corr()
loan_correlation
f, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(loan_correlation, 
            xticklabels=loan_correlation.columns.values,
            yticklabels=loan_correlation.columns.values,annot= True)
plt.show()
vars_to_be_removed=['loan_amnt','funded_amnt','funded_amnt_inv','verification_status','sub_grade','loan_status','total_pymnt','total_pymnt_inv','installment','issue_d','out_prncp_inv','out_prncp',
                    'total_rec_prncp','revol_bal','total_rec_int']
dataset.drop(vars_to_be_removed,axis=1,inplace=True)
# create X and Y
Y=dataset.values[:,-1]
# it will store in the form of array and we pass -1 to subset an aaray , here  only last column
X=dataset.values[:,:-1]
# all rowsand and columns except last column
Y=Y.astype(int)
print(Y)
# splitting the data into train and set => train_test_split()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
#building,training and testing the model

from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
#fitting the data to model
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification report:")

print(classification_report(Y_test, Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model:",acc)
#predicting using the Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree = DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)
#fit the model on the data and predict the values
Y_pred = model_DecisionTree.predict(X_test)
#print(Y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
