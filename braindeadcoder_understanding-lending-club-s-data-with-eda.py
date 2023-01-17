import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
loans = pd.read_csv("../input/loan.csv", low_memory=False) #Dataset
# Checking the Dimensions of our dataset

loans.shape
loans.columns
description = pd.read_excel('../input/LCDataDictionary.xlsx').dropna()
description.style.set_properties(subset=['Description'], **{'width' :'850px'})
loans.info()
fig = plt.figure(figsize=(15,10))
sns.heatmap(loans.isna(),cmap='inferno')
loans['loan_status'].value_counts()
target = [1 if i=='Default' else 0 for i in loans['loan_status']]
loans['target'] = target
loans['target'].value_counts()
nulls = pd.DataFrame(round(loans.isnull().sum()/len(loans.index)*100,2),columns=['null_percent'])
#sns.barplot(x='index',y='null_percent',data=nulls.reset_index())
nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)
# Drop unneccesary columns
loans = loans.drop(['url', 'desc', 'policy_code', 'last_pymnt_d', 'next_pymnt_d', 'earliest_cr_line', 'emp_title'], axis=1)
loans = loans.drop(['id', 'title', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'zip_code'], axis=1)

loans['member_id'].value_counts().head(5)
loans.drop(['member_id'], axis=1, inplace=True)
i = len(loans)
loans = pd.DataFrame(loans[loans['loan_status'] != "Does not meet the credit policy. Status:Fully Paid"])
loans = pd.DataFrame(loans[loans['loan_status'] != "Does not meet the credit policy. Status:Charged Off"])
loans = pd.DataFrame(loans[loans['loan_status'] != "Issued"])
loans = pd.DataFrame(loans[loans['loan_status'] != "In Grace Period"])
a = len(loans)
print(f"We dropped {i-a} rows, a {((i-a)/((a+i)/2))*100}% reduction in rows")
# Number of each type of column
sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(loans.dtypes,palette='viridis')
plt.title('Number of columns distributed by Data Types',fontsize=20)
plt.ylabel('Number of columns',fontsize=15)
plt.xlabel('Data type',fontsize=15)
# Let us see how many Object type features are actually Categorical
loans.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
sns.set(rc={'figure.figsize':(15,8)})
sns.countplot(loans['emp_length'],palette='inferno')
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Employement Length For Issued Loans")
plt.show()
sns.set(rc={'figure.figsize':(15,10)})
sns.violinplot(x="target",y="loan_amnt",data=loans, hue="pymnt_plan", split=True,palette='inferno')
plt.title("Payment plan - Loan Amount", fontsize=20)
plt.xlabel("TARGET", fontsize=15)
plt.ylabel("Loan Amount", fontsize=15);
sns.set(rc={'figure.figsize':(15,6)})
sns.boxplot(x='loan_amnt', y='loan_status', data=loans)
sns.set(rc={'figure.figsize':(15,6)})
sns.countplot(loans['grade'], palette='inferno')
loan_grades = loans.groupby("grade").mean().reset_index()

sns.set(rc={'figure.figsize':(15,6)})
sns.barplot(x='grade', y='loan_amnt', data=loan_grades, palette='inferno')
plt.title("Average Loan Amount - Grade", fontsize=20)
plt.xlabel("Grade", fontsize=15)
plt.ylabel("Average Loan Amount", fontsize=15);
sns.set(rc={'figure.figsize':(15,10)})
sns.violinplot(x="grade", y="int_rate", data=loans, palette='viridis', order="ABCDEFG",hue='target',split=True)
plt.title("Interest Rate - Grade", fontsize=20)
plt.xlabel("Grade", fontsize=15)
plt.ylabel("Interest Rate", fontsize=15);
sns.set(rc={'figure.figsize':(15,10)})
sns.violinplot(x="target",y="loan_amnt",data=loans, hue="application_type", split=True,palette='viridis')
plt.title("Application Type - Loan Amount", fontsize=20)
plt.xlabel("TARGET", fontsize=15)
plt.ylabel("Loan Amount", fontsize=15);
sns.set(rc={'figure.figsize':(15,5)})
sns.kdeplot(loans.loc[loans['target'] == 1, 'int_rate'], label = 'target = 1',shade=True)
sns.kdeplot(loans.loc[loans['target'] == 0, 'int_rate'], label = 'target = 0',shade=True);
plt.xlabel('Interest Rate (%)',fontsize=15)
plt.ylabel('Density',fontsize=15)
plt.title('Distribution of Interest Rate',fontsize=20);
state_default = loans[loans['target']==1]['addr_state']

sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(state_default, order=state_default.value_counts().index, palette='viridis')
plt.xlabel('State',fontsize=15)
plt.ylabel('Number of loans',fontsize=15)
plt.title('Number of defaulted loans per state',fontsize=20);
state_non_default = loans[loans['target']==0]['addr_state']

sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(state_non_default, order=state_non_default.value_counts().index, palette='viridis')
plt.xlabel('State',fontsize=15)
plt.ylabel('Number of loans',fontsize=15)
plt.title('Number of not-defaulted loans per state',fontsize=20);
loans.columns
loans.shape
loans.head(5)
nulls = pd.DataFrame(round(loans.isnull().sum()/len(loans.index)*100,2),columns=['null_percent'])
drop_cols = nulls[nulls['null_percent']>75.0].index
loans.drop(drop_cols, axis=1, inplace=True)
loans.shape
loans.head(5)
loans.columns
loans['issue_d']= pd.to_datetime(loans['issue_d']).apply(lambda x: int(x.strftime('%Y')))
loans['last_credit_pull_d']= pd.to_datetime(loans['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))
loans.drop(['loan_status'],axis=1,inplace=True)
categorical = []
for column in loans:
    if loans[column].dtype == 'object':
        categorical.append(column)
categorical
loans = pd.get_dummies(loans, columns=categorical)
loans.shape
loans['mths_since_last_delinq'].fillna(loans['mths_since_last_delinq'].median(), inplace=True)
# Finally we are going to drop all the rows that contain null values
loans.dropna(inplace=True)
sns.set(rc={'figure.figsize':(15,8)})
sns.heatmap(loans.isna(),cmap='inferno')