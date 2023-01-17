# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import all the required libraries for data analysis and model building

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 500)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
# Import the dataset
df = pd.read_csv('../input/loan.csv')
# Check dimension
df.shape
# Take a random sample of 40% data from the original dataset (due to pc memory problem)
df= df.sample(frac=0.4, random_state= 1)
# Check new dimension after sampling
df.shape
# Have a look at the data type
df.info()
# Check missing values count and percent
total= df.isnull().sum().sort_values(ascending=False)
percent= (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing_data= pd.concat([total, percent],axis=1, keys=["Total", "Percent"])
missing_data.head(25)
# Plot missing values
plt.figure(figsize=(15,5))
missing= df.isnull().sum()
missing= missing[missing > 0]
missing.sort_values(ascending= False, inplace=True)
plt.xlabel("Bar Plot of Missing Values", fontsize=15)
plt.ylabel("Number of missing values", fontsize=15)
plt.title("Number of missing data by feature", fontsize=15)
missing.plot(kind="bar")

plt.show()
# Any variable having missing values more than 50 percent are removed
df.drop(["dti_joint", "verification_status_joint", "annual_inc_joint", "il_util", "mths_since_rcnt_il",
            "all_util", "max_bal_bc", "open_rv_24m", "open_rv_12m", "total_cu_tl", "total_bal_il", "open_il_24m",
            "open_il_12m", "total_cu_tl", "total_bal_il", "open_il_24m", "open_il_12m", "open_il_6m", "open_acc_6m",
            "inq_fi", "inq_last_12m", "desc", "mths_since_last_record", "mths_since_last_major_derog",
            "mths_since_last_delinq", "next_pymnt_d", "total_rev_hi_lim", "tot_cur_bal", "tot_coll_amt"], axis=1,
           inplace=True)

# Delete unwanted columns
df.drop(["id", "url", "member_id"], axis=1, inplace=True)

# Payment plan has all the values "n" and only 3 values "y" so it is not important
df.drop(["pymnt_plan"], axis=1, inplace=True)

# Since we have both address state and zip code let's drop zip code and use address state only
df.drop(["zip_code"], axis=1, inplace=True)

# Title is not important instead we will use "purpose" variable
df.drop(["title"], axis=1, inplace=True)

# The grade is implied by the subgrade, so let's drop the grade column.
df.drop(["grade"], axis=1, inplace=True)
# remove "months" from "36 months" and convert it to int type
df["term"]= df['term'].map(lambda x: x.rstrip('months'))
df["term"]= df["term"].astype("int")
# Again Check missing values count and percent in remaining columns
total= df.isnull().sum().sort_values(ascending=False)
percent= (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing_data= pd.concat([total, percent],axis=1, keys=["Total", "Percent"])
missing_data.head(15)
# Loan Status
plt.figure(figsize = (12,8))
g = sns.countplot(x="loan_status",data=df,
                  palette='hls')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Loan Status", fontsize=20)
g.set_xlabel("Loan Status", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=20)
# Lets fill mode value in place of the missing value
df["emp_title"].value_counts(ascending=False).head()
df["emp_title"]= df["emp_title"].fillna(df["emp_title"].mode()[0])
# Lets fill mode value in place of the missing value
df["emp_length"].value_counts(ascending=False).head()
df["emp_length"]= df["emp_length"].fillna(df["emp_length"].mode()[0])
# Convert to date time
df["last_pymnt_d"]= pd.to_datetime(df["last_pymnt_d"])
df["last_pymnt_d"].value_counts(ascending=False).head()
# Lets fill mode value in place of the missing value
df["last_pymnt_d"]= df["last_pymnt_d"].fillna(df["last_pymnt_d"].mode()[0])
# Let's fill the Median in place of the missing values
df["revol_util"].value_counts(ascending=False).head()
df["revol_util"]= df["revol_util"].fillna(df["revol_util"].median())
# Lets fill mode value in place of the missing value
df["purpose"].value_counts(ascending=False).head()
df["purpose"]= df["purpose"].fillna(df["purpose"].mode()[0])
# Lets fill mode value in place of the missing value
df["collections_12_mths_ex_med"].value_counts()
df["collections_12_mths_ex_med"]= df["collections_12_mths_ex_med"].fillna(df["collections_12_mths_ex_med"].mode()[0])
# Convert to date time
df["last_credit_pull_d"]= pd.to_datetime(df["last_credit_pull_d"])
df["last_credit_pull_d"].value_counts(ascending=False).head()

# Lets fill mode value in place of the missing value
df["last_credit_pull_d"]= df["last_credit_pull_d"].fillna(df["last_credit_pull_d"].mode()[0])
# Lets fill mode value in place of the missing value
df["acc_now_delinq"].value_counts(ascending=False)
df["acc_now_delinq"]= df["acc_now_delinq"].fillna(df["acc_now_delinq"].mode()[0])
# Let's fill the Median in place of the missing values
df["total_acc"].value_counts(ascending=False)
df["total_acc"]= df["total_acc"].fillna(df["total_acc"].median())
# Let's fill the Median in place of the missing values
df["open_acc"].value_counts(ascending=False)
df["open_acc"]= df["open_acc"].fillna(df["open_acc"].median())
# Convert to date time
df["earliest_cr_line"]= pd.to_datetime(df["earliest_cr_line"])
df["earliest_cr_line"].value_counts(ascending=False).head()

# Lets fill mode value in place of the missing value
df["earliest_cr_line"]= df["earliest_cr_line"].fillna(df["earliest_cr_line"].mode()[0])
# Lets fill mode value in place of the missing value
df["inq_last_6mths"].value_counts(ascending=False)
df["inq_last_6mths"]= df["inq_last_6mths"].fillna(df["inq_last_6mths"].mode()[0])
# Lets fill mode value in place of the missing value
df["pub_rec"].value_counts(ascending=False)
df["pub_rec"]= df["pub_rec"].fillna(df["pub_rec"].mode()[0])
# Lets fill mode value in place of the missing value
df["delinq_2yrs"].value_counts(ascending=False)
df["delinq_2yrs"]= df["delinq_2yrs"].fillna(df["delinq_2yrs"].mode()[0])
# Let's fill the Median in place of the missing values
df["annual_inc"].value_counts(ascending=False)
df["annual_inc"]= df["annual_inc"].fillna(df["annual_inc"].median())
# Extract only year from the variable
df["earliest_cr_line"] = pd.DatetimeIndex(df["earliest_cr_line"]).month
df["last_pymnt_d"] = pd.DatetimeIndex(df["last_pymnt_d"]).month
df["last_credit_pull_d"] = pd.DatetimeIndex(df["last_credit_pull_d"]).month
df["loan_status"].value_counts()
# Create a target variable "Loan Status" with two categories Good and Bad Loan
df["Loan_Class"] = np.where((df.loan_status == 'Current') |
                        (df.loan_status == 'Fully Paid') |
                        (df.loan_status== "Issued") |
                        (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 1, 0)
# Bar plot of Term (Loan taken for number of months)
plt.figure(figsize=(10,6))
sns.barplot("term", "loan_amnt", data=df, palette='spring')
plt.title("Term of Loan", fontsize=16)
plt.xlabel("Months", fontsize=14)
plt.ylabel("Loan Amount", fontsize=14)
# Frequency distribution of Loan Amount
plt.figure(figsize=(12,6))
g = sns.distplot(df["loan_amnt"])
g.set_xlabel("Loan Amount", fontsize=12)
g.set_ylabel("Frequency", fontsize=12)
g.set_title("Frequency Distribuition- Loan Amount", fontsize=20)
# Frequency distribution of Interest Rate
plt.figure(figsize=(12,6))
g = sns.distplot(df["int_rate"])
g.set_xlabel("Interest Rate", fontsize=12)
g.set_ylabel("Frequency", fontsize=12)
g.set_title("Int Rate Distribuition", fontsize=20)
# Loan Status vs Loan Amount
plt.figure(figsize = (12,8))
g = sns.countplot(x="loan_status",data=df,
                  palette='hls')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Loan Status", fontsize=20)
g.set_xlabel("Loan Status", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=20)
# Application Type and Loan Amount
plt.figure(figsize = (12,8))
g = sns.countplot(x="purpose",data=df,
                  palette='hls')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Application Type", fontsize=20)
g.set_xlabel("Application Type", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)
# Boxplot of Employment Length and Inter
plt.figure(figsize = (12,8))
ax = sns.boxplot(x="emp_length" ,y= "int_rate", data=df, linewidth=2.5)

plt.show()
# Employment Length and Number of Loans
df['emp_length'].value_counts().sort_values().plot(kind='barh',figsize=(18,8))
plt.title('Number of loans distributed by Employment Years',fontsize=20)
plt.xlabel('Number of loans',fontsize=15)
plt.ylabel('Years worked',fontsize=15);
# No of Defaulted Loans per State
fig = plt.figure(figsize=(18,10))
df[df['Loan_Class']==1].groupby('addr_state')['Loan_Class'].count().sort_values().plot(kind='barh')
plt.ylabel('State',fontsize=15)
plt.xlabel('Number Of Loans',fontsize=15)
plt.title('Number Of Defaulted Loans Per State',fontsize=20);
# Boxplot of Term and Loan Amount
print("Loan Amount Distribution BoxPlot")
plt.figure(figsize=(8,5))
sns.boxplot(x=df.term, y=df.loan_amnt)
# Boxplot of Verification Status and Loan Amount
plt.figure(figsize=(10,8))
sns.boxplot(x=df.verification_status, y=df.loan_amnt)
plt.xlabel("Verification Status")
plt.ylabel("Loan Amount")
# Crosstabulation of Purpose and Loan Status
purp_loan= ['purpose', 'loan_status']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df[purp_loan[0]], df[purp_loan[1]]).style.background_gradient(cmap = cm)
# Convert into One Hot Encoding
# categorical_features.columns

cols= ['sub_grade', 'emp_title', 'emp_length', 'home_ownership',
       'verification_status', 'issue_d', 'loan_status', 'purpose', 'addr_state',
       'initial_list_status', 'application_type']

for i in cols:
    lbl= LabelEncoder()
    lbl.fit(list(df[i].values))
    df[i]= lbl.transform(list(df[i].values))
# Check correlation between Loan_Class and all other independent variables
correlation_m = df.corr()
correlation_m["Loan_Class"].sort_values(ascending=False)
# Countplot of Good Loans and Bad Loans
g= sns.countplot(df["Loan_Class"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Good Loan and Bad Loan", fontsize=20)
g.set_xlabel("Loan Type", fontsize=15)
g.set_ylabel("Frequency", fontsize=15)
# split the data into training and testing
from sklearn.model_selection import train_test_split
X = df.ix[:, df.columns != "Loan_Class"]
y = df["Loan_Class"]

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=44)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# Logistic Regression
log= LogisticRegression()
log.fit(X_train, y_train)

y_pred= log.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))
# Naive Bayes
naive= GaussianNB()
naive.fit(X_train, y_train)

y_pred= naive.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))
# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# Over Sampling
ros = RandomOverSampler(random_state=0)
X_train_res, y_train_res = ros.fit_sample(X_train, y_train.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
# Naive Bayes
naive= GaussianNB()
naive.fit(X_train, y_train)

y_pred= naive.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))
# Logistic Regression
log= LogisticRegression()
log.fit(X_train, y_train)

y_pred= log.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))
tmp = log.fit(X_train_res, y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# Under Sampling
rus = RandomUnderSampler(random_state=0)
X_train_res, y_train_res = rus.fit_sample(X_train, y_train.ravel())

print("After UnderSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After UnderSampling, counts of label '0': {}".format(sum(y_train_res==0)))
# Naive Bayes
naive= GaussianNB()
naive.fit(X_train, y_train)

y_pred= naive.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))
# Logistic Regression
log= LogisticRegression()
log.fit(X_train, y_train)

y_pred= log.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))
tmp = log.fit(X_train_res, y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
