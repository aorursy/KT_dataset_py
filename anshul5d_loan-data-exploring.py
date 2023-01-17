import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loan = pd.read_csv('../input/LoanStats3a.csv')
loan.shape
loan.isnull().all()
loan.isnull().sum()
loan = loan.dropna(axis=1, how='all')
loan = loan.dropna(axis=0, how='all')
loan.shape 
loan.isnull().sum()
loan[loan['id'].notnull()]
loan = loan.drop('id',axis=1)
loan.shape
loan=loan.drop(['sub_grade','pymnt_plan','zip_code','initial_list_status','out_prncp','out_prncp_inv',
                 'collections_12_mths_ex_med','policy_code','application_type','acc_now_delinq',
                 'chargeoff_within_12_mths','pub_rec_bankruptcies','desc','disbursement_method','hardship_flag',
                 'tax_liens','delinq_amnt','settlement_term'],axis=1)

# current shape of loan dataset is 
loan.shape
loan.describe()
loan.head(1)
# This bar graph is show the No. of Borrowers of Loan to the Loan Grades
sns.countplot(x='grade',data =loan,order=['A','B','C','D','E','F'])
plt.xlabel('Loan Grades')
plt.ylabel('No. of Borrowers')
plt.title('No. of Borrowers vs Loan Grades')
# Count the values of Each Grade
loan['grade'].value_counts()
sns.countplot(x='home_ownership', data= loan)
plt.xlabel('Borrower Lives')
plt.ylabel('No. of Loan Borrowers')
plt.title('Home Owners vs No. of Loan Borrowers')
#Term to complete the Loan 
sns.countplot(x='term',data=loan)
plt.xlabel('Terms ')
plt.ylabel('No. of Borrowers')
plt.title('No. of Borrowers vs Terms')
loan['emp_length'].unique()
plt.figure(figsize=(14,5))
order = ['nan','< 1 year', '1 year' ,'2 years','3 years','4 years', '5 years', '6 years',  '7 years',
         '8 years', '9 years','10+ years']
sns.countplot(x='emp_length',data=loan,order=order)
plt.xlabel('Loan Years')
plt.ylabel('No. of Borrowers')
plt.title('No. of Borrowers Vs Loan Years')
plt.figure(figsize=(16,4))
order = ['nan','< 1 year', '1 year' ,'2 years','3 years','4 years', '5 years', '6 years',  '7 years',
         '8 years', '9 years','10+ years']
sns.violinplot(x='emp_length',y='loan_amnt',data =loan,order=order,hue='term', split=True)
plt.xlabel('Loan Year Length')
plt.ylabel('Loan Amount')
plt.title('Laon Amount vs Loan Year Length')
sns.countplot('verification_status', data= loan)
plt.xlabel('Varification Status')
plt.ylabel('No. of Application')
plt.title('Application Vs Verification Status')
month ={'Dec':12,'Nov':11,'Oct':10,'Sep':9,'Aug':8,'Jul':7,'Jun':6,'May':5,'Apr':4,'Mar':3,'Feb':2,'Jan':1}
loan = loan.dropna(axis=0,how='all')
#for x in loan['issue_d']:
 #   print(x.split('-')[1])
x=loan['issue_d'].iloc[0]
x.split('-')[1]
loan['issue_d'].isnull().sum()
loan['Year']=loan['issue_d'].apply(lambda issue_d : issue_d.split('-')[1])
loan['Year'] = '20' + loan['Year'].astype(str)
loan['Month'] = loan['issue_d'].apply(lambda issue_d: issue_d.split('-')[0])
loan['Month']
loan['MonthNUM'] = loan['Month'].map(month) 
loan['MonthNUM'].head()
loan['Year'].head()
bad_loan = ['Charged Off','Does not meet the credit policy. Status:Charged Off']
good_loan=['Fully Paid','Does not meet the credit policy. Status:Charged Off']
loan['Loan_condition'] = np.nan

def loan_condition(status):
    if status in bad_loan:
        return 'Bad_loans'
    else:
        return 'Good_loans'

loan['Loan_condition'] = loan['loan_status'].apply(loan_condition)
sns.countplot(x='Loan_condition', data = loan).grid()
plt.title('Good Loan Vs Bad Loan')
plt.xlabel('Loan_Condition on Value Paid')
plt.ylabel('No. of Values')
f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#3191D7", "#D71616"]
labels ="Good Loans", "Bad Loans"


plt.suptitle('Information on Loan Conditions', fontsize=20)

loan["Loan_condition"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=110)

ax[0].set_ylabel('% of Condition of Loans', fontsize=14)
palette = ["#3791D7", "#E01E1B"]

sns.barplot(x='Year',y='loan_amnt', hue='Loan_condition', data=loan, palette=palette)
plt.xlabel('Years of Good Loans vs Bad Loans')
plt.ylabel('Loan Amount')
loan_mean = loan.groupby(['Year']).mean()
loan_Month_Mean= loan.groupby(['Month']).mean().reset_index().sort_values(by='Month')
loan_Month_Mean
loan_mean['loan_amnt'].plot.bar(x='Year',y='loan_amnt')
f,(ax1,ax2)=plt.subplots(1,2,figsize=(16,5))
sns.barplot(x='Month',y='loan_amnt',data=loan,ax=ax1,order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
sns.barplot(x='Year',y='loan_amnt',data=loan,ax=ax2)
f, ax= plt.subplots(1,3,figsize=(16,4))

loan_mean['loan_amnt'].plot.bar(x='Year',y='loan_amnt',ax=ax[0])


loan_mean['funded_amnt'].plot.bar(x='Year',y='funded_amnt',ax=ax[1])

loan_mean['funded_amnt_inv'].plot.bar(x='Year',y='funded_amnt_inv',ax=ax[2])
f,(ax1,ax2)=plt.subplots(1,2,figsize=(16,4))

sns.barplot(x='grade',y='loan_amnt',data=loan,order=['A','B','C','D','E','F'],ax=ax1)
sns.boxplot(x='grade', y='loan_amnt', data=loan, order = 'ABCDEFG')
plt.xlabel('Grade')
plt.ylabel('Loan Amount')
plt.title('Loan Amount vs Grade')
loan_Month_Mean['loan_amnt'].plot()
loan_Month_Mean['funded_amnt'].plot()
plt.legend()
US_Army =['US Army','U.S Army','U.S. Army','US ARMY','United States Army','us army','Army','Us army']
US_Air_Force= ['USAF','US Air Force','United States Air Force','U.S. Air Force','Air Force']
US_Navy =['US Navy','Navy','U.S. Navy','United States Navy','US NAVY']
US_Postal_Servies=['us postal service','US Postal Service','United States Postal Service','USPS','usps','U.S. Postal Service',
                  'united states postal service','U S Postal Service','US Postal Service (USPS)','UNITED STATES POSTAL SERVICE','U. S. Postal Service']
JP_Morgan_Chase=['JP Morgan Chase','JPMorgan Chase','JPMorgan Chase & Co.'] 
United_Parcel_Service =['United Parcel Service','UPS','united parcel service'] 
Self_employed=['Self','self','Self-employed','Self-Employed','Self Employed','Self employed','self-employed','self employed']
Walmart =['Walmart','walmart','WalMart','Walmart','Wal-Mart','wal mart']
Bank_of_America =['Bank of America','Bank of America Corp.','Bank Of America','bank of america']
AT_T = ['AT&T','at&t','ATT','att','AT and T','AT&T Labs','AT&T Inc.','At&T']
Dell= ['Dell Inc', 'Dell','DELL','dell inc','dell']
Wells_Fargo_Bank=['Wells Fargo Bank','wells fargo','Wells Fargo']
HP=['Hewlett-Packard','hp','HP','Hewlett Packard']
KPMG=['KPMG','KPMG LLP']
loan['emp_title'].value_counts()
loan['Employee_Title'] = np.nan

def employee_title(title):
    if title in US_Army:
        return 'US_Army'
    elif title in US_Air_Force:
        return 'US_Air_Force'
    elif title in US_Navy:
        return 'US_Navy'
    elif title in US_Postal_Servies:
        return 'US_Postal_Servies'
    elif title in JP_Morgan_Chase:
        return 'JP Morgan Chase'  
    elif title in United_Parcel_Service:
        return 'United Parcel Service'
    elif title in Self_employed:
        return 'Self Employed'
    elif title in Walmart:
        return 'Walmart'
    elif title in Bank_of_America:
        return 'Bank of America'
    elif title in AT_T:
        return 'AT&T Labs'
    elif title in Dell:
        return 'Dell'
    elif title in Wells_Fargo_Bank:
        return 'Wells_Fargo_Bank'
    elif title in HP:
        return 'HP'
    elif title in KPMG:
        return 'KPMG'
    else:
        return title
    
loan['Employee_Title'] = loan['emp_title'].apply(employee_title)
loan['Employee_Title'].value_counts().head(10).plot.bar().grid()
fig ,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,10))

sns.violinplot(x='Loan_condition',y='annual_inc',data=loan,hue='term',ax=ax1)
plt.legend(bbox_to_anchor =(1.05,2),loc=2,borderaxespad= 0.)
sns.boxplot(x='Loan_condition',y='annual_inc',data=loan,hue='term',ax=ax2)
plt.legend(bbox_to_anchor =(1.05,1),loc=2,borderaxespad= 0.)
sns.violinplot(x='Loan_condition',y='funded_amnt_inv',hue='term',data=loan,ax=ax3, split=True)
plt.legend(bbox_to_anchor =(1.05,1),loc=2,borderaxespad= 0.)
sns.boxplot(x='Loan_condition',y='funded_amnt_inv',data=loan,hue='term',ax=ax4)
plt.legend(bbox_to_anchor =(1.05,1),loc=2,borderaxespad= 0.)
sns.jointplot(x='installment',y='loan_amnt',data = loan)
loan_GL=loan[['grade','term','Loan_condition','loan_amnt','installment','annual_inc']]
#sns.pairplot(loan,hue='term')
loan_GL.head(1)
plt.figure(figsize=(16,6))
sns.pairplot(loan_GL,hue='term')
sns.violinplot(x='term',y='installment', data=loan)
sns.lmplot(x='installment',y='loan_amnt',data=loan_GL,hue='Loan_condition')


