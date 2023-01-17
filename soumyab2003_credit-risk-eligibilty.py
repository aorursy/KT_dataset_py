# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
df=pd.read_csv("../input/credit-risk-loan-eliginility/train_split.csv")
df.shape
df.head(5)
#checking for missing values in all columns
df.isnull().sum()
# Removing columns with less significace
#funded_amnt                        
#funded_amnt_inv 
#member_id 
#batch_enrolled
#emp_title
#sub_grade 
#zip_code 


df=df.drop(['funded_amnt','funded_amnt_inv','member_id','batch_enrolled','emp_title','sub_grade','zip_code'],axis=1)
df.head()
#more columns that we can get rid of that will not effect much.It is always better to have less confusing data
df=df.drop(['recoveries','collection_recovery_fee','total_rec_int','total_rec_late_fee','tot_coll_amt',],axis=1)
df.head()
df.isnull().sum()
#drop columns with high missing value
df=df.drop(['verification_status_joint','mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq','desc'],axis=1)
df.head()
df.isnull().sum()
df.describe()
print(list(df.columns))
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

#checking for corelations
corr=df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
#Data Expploration of target values
y=df['loan_status']
y.value_counts()
ax = sns.countplot(x="loan_status", data=df)
count_0 = len(df[df['loan_status']== 0])
count_1 = len(df[df['loan_status']==1])
pct_of_0 = count_0/(count_1+count_0)
print("percentage of 0 are", pct_of_0*100)
pct_of_1 = count_1/(count_1+count_0)
print("percentage of 1 are", pct_of_1*100)
#most of the loan status are 0
df.groupby('loan_status').mean()
df.isnull().sum()
#removing more than 1% missing values and 29 rows from revol_util
df=df.drop(['tot_cur_bal','total_rev_hi_lim'],axis=1)

df=df.dropna()
df.head()
df.dtypes
#categorical variables are- term,grade
#home_ownership,veri_status,addr state,application type,lastweek pay
col_obj=df[['home_ownership','grade','verification_status','addr_state','initial_list_status','last_week_pay',]]
for i in col_obj:
     
        print(col_obj[i].value_counts())
df.title.value_counts()
df.emp_length.value_counts()
df.purpose.value_counts()
#let us get rid of title and addr state
df=df.drop(['title','addr_state'],axis=1)
df.head()
#ordinal value
#emp length is a categorical value which can be converted into numerical value
mapping_dict = {
"emp_length": {
"10+ years": 10,
"9 years": 9,
"8 years": 8,
"7 years": 7,
"6 years": 6,
"5 years": 5,
"4 years": 4,
"3 years": 3,
"2 years": 2,
"1 year": 1,
"< 1 year": 0,
"n/a": 0
},
"grade":{
"A": 1,
"B": 2,
"C": 3,
"D": 4,
"E": 5,
"F": 6,
"G": 7
}
}
df = df.replace(mapping_dict)

df.head()
df.dtypes
#changing rest of nominal values to numeric 
nominal_columns = ["home_ownership", "verification_status", "purpose", "term","initial_list_status","application_type","last_week_pay"]
dummy_df = pd.get_dummies(df[nominal_columns])
df1 = pd.concat([df, dummy_df], axis=1)
df1 = df1.drop(nominal_columns, axis=1)
df.head()
df1=df1.drop(['pymnt_plan'],axis=1)
df1.head()
df1.columns
from sklearn.model_selection import train_test_split

y=df1.loan_status
x=df1.drop('loan_status',axis=1)
x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape
x_test.shape
#Using Random Forest
from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
rfc_cv_score = cross_val_score(clf, x, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
y_test.head()
y_pred[0:5]
import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=x.columns).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(20,20))
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

