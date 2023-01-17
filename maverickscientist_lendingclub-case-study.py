import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas_profiling
from pandas_profiling import ProfileReport 

# set the background theme
sns.set_style("darkgrid")

# variable to store loan data file pathname
loanFilePath = '../input/loan-dataset/loan.csv'

# get the loan data frame
loan_df = pd.read_csv(loanFilePath,encoding = "ISO-8859-1", low_memory=False)

loan_df.head()

# check the dataframe's rows & columns
loan_df.shape
ProfileReport(loan_df)
missing = round(100*(loan_df.isnull().sum()/len(loan_df.id)), 2)
missing.loc[missing > 0]
columns_with_missing_values = list(missing[missing >= 50].index)
len(columns_with_missing_values)
loan_df = loan_df.drop(columns_with_missing_values,axis=1)
loan_df.shape
missing = round(100*(loan_df.isnull().sum()/len(loan_df.id)), 2)
missing.loc[missing!= 0]
# let's remove the description column, it is the description provided by the borrowers at the time of application which makes no difference to our analysis.
loan_df = loan_df.drop('desc',axis=1)
loan_df.shape
print("Distinct records for emp_title : %d"  % len(loan_df.emp_title.unique()))
print("Distinct records for emp_length : %d"  % len(loan_df.emp_length.unique()))
print("Distinct records for title : %d"  % len(loan_df.title.unique()))
print("Distinct records for revol_util : %d"  % len(loan_df.revol_util.unique()))
print("Distinct records for title : %d"  % len(loan_df.title.unique()))
print("Distinct records for last_pymnt_d : %d"  % len(loan_df.last_pymnt_d.unique()))
print("Distinct records for last_credit_pull_d : %d"  % len(loan_df.last_credit_pull_d.unique()))
print("Distinct records for collections_12_mths_ex_med : %d"  % len(loan_df.collections_12_mths_ex_med.unique()))
print("Distinct records for chargeoff_within_12_mths : %d"  % len(loan_df.chargeoff_within_12_mths.unique()))
print("Distinct records for pub_rec_bankruptcies : %d"  % len(loan_df.pub_rec_bankruptcies.unique()))
print("Distinct records for tax_liens : %d"  % len(loan_df.tax_liens.unique()))
columns_to_be_dropped = ['collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'tax_liens']
loan_df = loan_df.drop(columns_to_be_dropped,axis=1)
loan_df.shape
loan_df.pub_rec_bankruptcies.value_counts()
loan_df=loan_df[~loan_df.pub_rec_bankruptcies.isnull()]
missing = round(100*(loan_df.isnull().sum()/len(loan_df.id)), 2) 
missing[missing != 0]
loan_df=loan_df[~loan_df.emp_title.isnull()]
loan_df=loan_df[~loan_df.emp_length.isnull()]
missing = round(100*(loan_df.isnull().sum()/len(loan_df.id)), 2) 
missing[missing != 0]
loan_df=loan_df[~loan_df.title.isnull()]
loan_df=loan_df[~loan_df.revol_util.isnull()]
loan_df=loan_df[~loan_df.last_pymnt_d.isnull()]
missing = round(100*(loan_df.isnull().sum()/len(loan_df.id)), 2) 
missing[missing != 0]
loan_df.shape
loan_df.nunique().sort_values()
columns_to_be_removed = ['id','member_id','pymnt_plan','url','zip_code','initial_list_status','policy_code','application_type','acc_now_delinq','delinq_amnt','funded_amnt','funded_amnt_inv']
loan_df = loan_df.drop(columns_to_be_removed,axis=1)
loan_df.shape
date_col_list =['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
loan_df[date_col_list].info()
loan_df.issue_d = pd.to_datetime(loan_df.issue_d, format='%b-%y')
loan_df.earliest_cr_line = pd.to_datetime(loan_df.earliest_cr_line, format='%b-%y')
loan_df.last_pymnt_d = pd.to_datetime(loan_df.last_pymnt_d, format='%b-%y')
loan_df.last_credit_pull_d = pd.to_datetime(loan_df.last_credit_pull_d, format='%b-%y')
loan_df[date_col_list].info()
loan_df.dtypes
loan_df['int_rate'] = loan_df['int_rate'].str.strip('%').astype('float')
loan_df['revol_util'] = loan_df['revol_util'].str.strip('%').astype('float')
loan_df[['int_rate','revol_util']].info()
loan_df.emp_length.value_counts()
employeeLengthNumberMapping = {
    '< 1 year' : 0,
    '1 year' : 1,
    '2 years' : 2,
    '3 years' : 3,
    '4 years' : 4,
    '5 years' : 5,
    '6 years' : 6,
    '7 years' : 7,
    '8 years' : 8,
    '9 years' : 9,
    '10+ years' : 10
}
loan_df = loan_df.replace({"emp_length": employeeLengthNumberMapping })
loan_df.emp_length.value_counts()
loan_df.term.value_counts()
loan_df['term']=loan_df['term'].map(lambda x: str(x).replace('months','')).astype(int)
loan_df.earliest_cr_line.value_counts()
loan_df['earliest_cr_line_month'] = loan_df['earliest_cr_line'].dt.month
loan_df['earliest_cr_line_year'] = loan_df['earliest_cr_line'].dt.year
loan_df.earliest_cr_line_month.value_counts()
loan_df.earliest_cr_line_year.value_counts()
loan_df['issue_d_month'] = loan_df['issue_d'].dt.month
loan_df['issue_d_year'] = loan_df['issue_d'].dt.year
loan_df.issue_d_month.value_counts()
loan_df.issue_d_year.value_counts()
loan_df.shape
loan_df.loan_status.value_counts()
loan_df.shape
loan_df.head(10)
loan_df.isnull().sum()
loan_df.dropna(axis='columns', how='all',inplace=True)
loan_df.shape
#Further analysis, create dataframes for Fully Paid & Charged Off loan_status
loan_fp=loan_df.loc[(loan_df.loan_status=='Fully Paid'),]
loan_co=loan_df.loc[(loan_df.loan_status=='Charged Off'),]
loan_df.loan_amnt.describe()
#Plot graph for loan amount wise distribution
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title('Loan amount wise distribution')
sns.distplot(loan_df['loan_amnt'])
#Plot graph to verify year wise loan disbursal
plt.subplot(132)
plt.title('Year wise loan disbursal')
loan_df.groupby('issue_d_year').loan_amnt.sum().plot()
#plot a graph to know the loan status
plt.subplot(133)
plt.title('Loan status')
sns.countplot(x="loan_status", data=loan_df)
# Plot the graph between intrest rate and fully paid
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title('Interest Rate vs Fully Paid Customer')
sns.distplot(loan_fp.int_rate)
plt.subplot(132)
plt.title('Interest Rate vs Charge off')
sns.distplot(loan_co.int_rate)
plt.subplot(133)
plt.title('Interest Rate - Year on Year')
loan_df.groupby('issue_d_year').int_rate.median().plot()
plt.figure(figsize=(20,10))
plt.subplot(131)
plt.title('Statewise fully paid loans count')
sns.countplot(y='addr_state', data=loan_fp)
plt.subplot(132)
plt.title('Statewise charge off loans count')
sns.countplot(y='addr_state', data=loan_co)
plt.figure(figsize=(20,15))
plt.subplot(131)
plt.title('Salary range for fully paid customer')
sns.distplot(loan_fp.annual_inc, hist=False)
plt.subplot(132)
plt.title('Salary range for charged off customer')
sns.distplot(loan_co.annual_inc, hist=False)
def getDefaultStatus (x):
    if x == 'Fully Paid':
        return 0
    if x == 'Charged Off':
        return 1
    
loan_df['is_default'] = loan_df['loan_status'].apply(lambda x: getDefaultStatus(x))
# get rid of NaN in "is_default"
loan_df=loan_df[~loan_df.is_default.isnull()]
round(np.mean(loan_df['is_default']), 2)
def plot_cat_against_default(category):
    sns.barplot(x=category, y='is_default', data=loan_df)
    plt.show()
# grade of the loan Vs default
plot_cat_against_default('grade')
# sub_grade Vs default
plt.figure(figsize=(20, 6))
plot_cat_against_default('sub_grade')
# term Vs default
plot_cat_against_default('term')
# verification_status Vs default
plot_cat_against_default('verification_status')
print(round(loan_df.purpose.value_counts(normalize=True)*100),2)
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.title('Loan Purpose vs Fully Paid')
sns.countplot(y="purpose", data=loan_fp)
plt.subplot(122)
plt.title('Loan Purpose vs Charged Off')
sns.countplot(y="purpose", data=loan_co)
# purpose Vs default
plt.figure(figsize=(16, 6))
# Rotating the labels on X-axis to avoid overlapping
plt.xticks(rotation=30) 
plot_cat_against_default('purpose')
# loan_issued_year Vs default
plot_cat_against_default('issue_d_year')
# loan_issued_month Vs default
plt.figure(figsize=(12, 6))
plot_cat_against_default('issue_d_month')
# distribution plot of loan amount
plt.figure(figsize=(12, 6))
sns.distplot(loan_df['loan_amnt'])
plt.show()
# loan amount
def get_loan_amount_category(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'

# interest rate
def get_int_rate_category(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'

# debt to income ratio
def get_dti_category(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'

# funded amount
def get_funded_amount_cat(n):
    if n <= 5000:
        return 'low'
    elif n > 5000 and n <=15000:
        return 'medium'
    else:
        return 'high'


# installment
def get_installment_category(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'

# annual income
def get_annual_income_category(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

# employment length
def get_emp_length_category(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=4:
        return 'junior'
    elif n > 5 and n <=8:
        return 'senior'
    elif n > 8 and n <=12:
        return 'expert'
    else:
        return 'guru'
    
# lets store the above categories in new columns created corresponding to the field in context
loan_df['loan_amnt_cat'] = loan_df['loan_amnt'].apply(lambda x: get_loan_amount_category(x))
loan_df['int_rate_cat'] = loan_df['int_rate'].apply(lambda x: get_int_rate_category(x))
loan_df['dti_cat'] = loan_df['dti'].apply(lambda x: get_dti_category(x))
loan_df['installment_cat'] = loan_df['installment'].apply(lambda x: get_installment_category(x))
loan_df['annual_inc_cat'] = loan_df['annual_inc'].apply(lambda x: get_annual_income_category(x))
loan_df['emp_length_cat'] = loan_df['emp_length'].apply(lambda x: get_emp_length_category(x))
plt.title('Loan Amount Category Vs Default')
plot_cat_against_default('loan_amnt_cat')
plt.title('Interest Rate Category Vs Default')
plot_cat_against_default('int_rate_cat')
plt.title('Debt to Income Ratio Vs Default')
plot_cat_against_default('dti_cat')
plt.title('Installment Category Vs Default')
plot_cat_against_default('installment_cat')
plt.title('Annual Income Category Vs Default')
plot_cat_against_default('annual_inc_cat')
plt.title('Employment Length Category Vs Default')
plot_cat_against_default('emp_length_cat')
# lets check the amount of loan for each purpose
plt.figure(figsize=(16, 6))
plt.xticks(rotation=40)
sns.countplot(x='purpose', data=loan_df)
plt.show()
# Lets analyse the top 5 loans by filtering the dataframe
top_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase","small_business"]
loan_df = loan_df[loan_df['purpose'].isin(top_purposes)]
loan_df['purpose'].value_counts()
sns.countplot(x=loan_df['purpose'])
plt.xticks(rotation=30)
plt.show()
# lets write a method to plot segmented analysis by comparing the default rates with other categorical variables, segmented
# by the purpose of loan
def plot_segmented_graph(cat_var):
    plt.figure(figsize=(20, 6))
    sns.barplot(x=cat_var, y='is_default', hue='purpose', data=loan_df)
    plt.show()
# term
plot_segmented_graph("term")
# issued year
plot_segmented_graph('issue_d_year')
# home ownership
plot_segmented_graph('home_ownership')
# grade
plot_segmented_graph('grade')
# length of employment
plot_segmented_graph('emp_length_cat')
#loan amount category
plot_segmented_graph('loan_amnt_cat')
# interest rate category
plot_segmented_graph('int_rate_cat')
# installment category
plot_segmented_graph('installment_cat')
# annualincome category
plot_segmented_graph('annual_inc_cat')
# debt to income ratio category
plot_segmented_graph('dti_cat')
plt.title('loan status vs loan amount')
sns.boxplot(y='loan_amnt', x='loan_status', data=loan_df)
plt.figure(figsize=(10,3))
plt.title('Loan Status vs Delinquency_2yrs')
sns.countplot(y='delinq_2yrs', data=loan_fp)
sns.countplot(y='delinq_2yrs', data=loan_co)
loan_df.head()
print(loan_df.pivot_table( values='term',index = 'loan_status', columns = 'home_ownership', aggfunc='count'))
plt.figure(figsize=(10,3))
sns.boxplot(x='home_ownership',y='term', data=loan_fp)
sns.boxplot(x='home_ownership',y='term' , data=loan_co)
## Now we plot the graphs loan status vs grade with respect to fully paid and charged off
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.title('Loan Amount vs Grade with respect to Fully Paid status')
sns.barplot(y='loan_amnt', x='grade', data=loan_fp)
plt.subplot(132)
plt.title('Loan Amount vs Grade with respect to Charged Off status')
sns.barplot(y='loan_amnt', x='grade', data=loan_co)
print(loan_df.pub_rec_bankruptcies.value_counts(normalize=True))
print(loan_df.pivot_table( values='term',index = 'loan_status', columns = 'pub_rec_bankruptcies', aggfunc='count'))
print(loan_df.pivot_table( values='loan_amnt',index = 'purpose', columns = 'loan_status', aggfunc='count'))
sns.scatterplot(x='annual_inc',y='emp_length',data=loan_df)
plt.xscale('log')
print(loan_df.pivot_table( values='term',index = 'home_ownership', columns = 'loan_status', aggfunc='count'))
plt.figure(figsize=(20,5))
sns.boxplot(x='loan_status',y='revol_util',data=loan_df)
plt.figure(figsize=(20,5))
sns.boxplot(x='grade',y='revol_util',data=loan_df)
corr = loan_df.corr()
plt.figure(figsize=(15,15))
cont_var= ['loan_amnt', 'int_rate', 'installment',
       'annual_inc',
       'dti', 'delinq_2yrs', 'earliest_cr_line',
       'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
       'total_acc', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d',
       'pub_rec_bankruptcies']
corr = loan_df[cont_var].corr()
sns.heatmap(corr, annot=True, center=0.5)
# final rows & columns info of our dataframe after the analysis has been done
loan_df.shape
