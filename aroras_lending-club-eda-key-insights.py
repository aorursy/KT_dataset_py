# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
# load in the dataset into a pandas dataframe, print statistics
loan_df = pd.read_csv('../input/lending-club-loan-dataset-2007-2011/loan.csv',encoding = "ISO-8859-1", low_memory=False)
print(loan_df.shape)
print(loan_df.dtypes)
print(loan_df.head(10))
# percentage of null values in each column
round(100 * loan_df.isnull().sum()/loan_df['id'].count())
# Removing columns that have more that 50% nulls
threshold_number = loan_df['id'].count()/2
loan_df = loan_df.loc[:, loan_df.isnull().sum(axis=0) <= threshold_number]
loan_df.shape
# Checking number of unique values in each column
loan_df.nunique()
# Removing columns that has single value. Those columns will not give us any insights
loan_df = loan_df.loc[:, loan_df.nunique(axis=0) > 1]
loan_df.shape
loan_df.nunique().sort_values(ascending=False)
# Checking data in the columns with low variation
loan_df['term'].value_counts()
# converting term to int datatype , since term_months represents numeric
loan_df['term_months'] = loan_df['term'].str.lstrip().str.slice(stop=2).astype('int')
loan_df['term_months'].value_counts()
# dropping unused term column
loan_df = loan_df.drop('term', axis=1)
# check unique values for pub_rec_bankruptcies
loan_df['pub_rec_bankruptcies'].value_counts()
# looks like few values are missing
# check null value count
loan_df['pub_rec_bankruptcies'].isnull().sum()
# we dont want to be bias to bankruptcies. Removing rows with null values as it's safe to remove, since
# low percentage of null values
loan_df = loan_df[~loan_df['pub_rec_bankruptcies'].isnull()]
# verify null values have been removed
loan_df['pub_rec_bankruptcies'].isnull().sum() == 0
# check for unique values in loan_status column
loan_df['loan_status'].value_counts()
# check null value count
loan_df['loan_status'].isnull().sum()
# percentage of null values in each column
round(100 * loan_df.isnull().sum()/loan_df['id'].count(),2)
# removing description as it's not significant
loan_df = loan_df.drop('desc', axis=1)
# removing rows with null values(as they are low in percentage):
# employee title, employee length, title, revol_util, last_pymnt_d
loan_df = loan_df[~loan_df['emp_title'].isnull()]
loan_df = loan_df[~loan_df['emp_length'].isnull()]
loan_df = loan_df[~loan_df['title'].isnull()]
loan_df = loan_df[~loan_df['revol_util'].isnull()]
loan_df = loan_df[~loan_df['last_pymnt_d'].isnull()]
# percentage of null values in each column
round(100 * loan_df.isnull().sum()/loan_df['id'].count(),2)
# exploring data values in each column
loan_df.head()
loan_df.head()
# int_rate and revol_util are percentage strings
loan_df['int_rate'] = loan_df['int_rate'].str.strip('%').astype('float')
loan_df['revol_util'] = loan_df['revol_util'].str.strip('%').astype('float')
# emp_length can be numeric as well
loan_df['emp_length'].value_counts()
# can give values 0 to 10: 0 for < 1 year, 10, for 10+
# using replace method on dataframe
replace_dict = {
    '10+ years': 10,
    '2 years': 2,
    '< 1 year': 0,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '1 year': 1,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9
}
loan_df = loan_df.replace({"emp_length": replace_dict })
loan_df['emp_length'].value_counts()
# object date columns: last_pymnt_d, last_credit_pull_d, earliest_cr_line, issue_d
# converting them to datetime columns
loan_df['last_pymnt_d'] = pd.to_datetime(loan_df['last_pymnt_d'], format='%b-%y')
loan_df['last_credit_pull_d'] = pd.to_datetime(loan_df['last_credit_pull_d'], format='%b-%y')
loan_df['earliest_cr_line'] = pd.to_datetime(loan_df['earliest_cr_line'], format='%b-%y')
loan_df['issue_d'] = pd.to_datetime(loan_df['issue_d'], format='%b-%y')
# verify columns are converted to datetime
loan_df.info()
# splitting of month and year on issue date
loan_df['issue_d_month'] = loan_df['issue_d'].dt.month
loan_df['issue_d_year'] = loan_df['issue_d'].dt.year
# Listing Consumer behaviour columns
behaviour_columns = ['last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'delinq_2yrs', 
                     'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec',
                    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
                    'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
                    'recoveries', 'collection_recovery_fee']
# Listing usued columns for analysis
# id and member_id are insignificant columns. can remove them.
# title can be ignored too as purpose column drives our analysis better
unused_columns = ['funded_amnt_inv', 'zip_code', 'addr_state', 'url', 'id', 'member_id', 'title']
to_drop_columns = behaviour_columns + unused_columns
# droping usused columns
loan_df = loan_df.drop(to_drop_columns, axis=1)
loan_df.shape
loan_df['loan_status'].value_counts()
# keeping the original column before converting to numeric
loan_df['loan_status_cat'] = loan_df['loan_status']
loan_df['loan_status_cat']
# Filtering only Fully Paid and Charged Off loans, converting them to numeric
loan_df = loan_df.loc[loan_df['loan_status'] != 'Current', :]
loan_df['loan_status'] = loan_df['loan_status'].apply(lambda x: 1 if x=='Charged Off' else 0)
loan_df['loan_status']
loan_df['loan_status'].value_counts()
loan_df.info()
df = loan_df.copy()
# let's check the proportion of loan status who are defaults
loan_df['loan_status'].describe()
# let's plot the countplot for loan status category
base_color = sns.color_palette()[0]
sns.countplot(data = df, x = 'loan_status_cat', color = base_color);
plt.title("Distribution of Loan Status")
plt.xlabel('Loan Status')
plt.show();
# let's check the distribution of interest rate charged to customers
df.int_rate.describe()
# let's also plot box plot for interest rate
df.int_rate.plot(kind='box');
# let's check the distribution of Annual Income
df.annual_inc.describe()
# let's also plot box plot for interest rate
df.annual_inc.plot(kind='box');
# grade
level_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sns.countplot(data = df, x = 'grade', color = base_color, order=level_order);
# term_month
sns.countplot(data = df, x = 'term_months', color = base_color);
# Let's check the how purpose looks like
sns.countplot(y='purpose', data=df, color = base_color);
#proportion of unique purpose in dataset
df.purpose.value_counts() / df.shape[0]
# let's check interest rate descriptive stats
df['int_rate'].describe()
# binning int_rate
df['int_rate_bin'] = pd.cut(df['int_rate'], 
                                [0,5,10,15,20,25,30], 
                                labels=['0-5','5-10','10-15','15-20','20-25','25-30'])
df['int_rate_bin'].value_counts()
# Plot between interest rate and loan status
sns.barplot(x='int_rate_bin', y='loan_status', data=df, color = base_color)
plt.title('Interest Rate Status')
plt.xlabel('Interest Rate')
plt.ylabel('Loan Status')
plt.show()
# Continuous variable: annual_inc
df['annual_inc_raw'] = df['annual_inc']
df['annual_inc'].describe().astype('int')
# binning annual income
def annual_inc(inc):
    if inc <= 50000:
        return 'low'
    elif inc > 50000 and inc <=100000:
        return 'medium'
    elif inc > 100000 and inc <=150000:
        return 'high'
    else:
        return 'very high'

df['annual_inc'] = df['annual_inc'].apply(lambda x: annual_inc(x))
df['annual_inc'].value_counts()
# cross tab between annual_inc and loan_status
pd.crosstab(df.annual_inc, df.loan_status_cat, margins=True, margins_name="Total")
## bar plot on categorical variable : annual_inc
sns.barplot(x='annual_inc', y='loan_status', data=df, color = base_color)
plt.title('Annual Income Default Status')
plt.xlabel('Annual Income')
plt.ylabel('Loan Status')
plt.show()
# crosstab between loan status and grade
pd.crosstab(df.grade, df.loan_status_cat, margins=True, margins_name="Total", normalize="index")
# bar plot on categorical variable : grade
sns.barplot(x='grade', y='loan_status', data=df, color = base_color, order = level_order)
plt.title('Loan status for Grades')
plt.xlabel('Grade')
plt.ylabel('Loan Status')
plt.show()
# crosstab between month term and loan status. Showing percentage of defaults
pd.crosstab(df.term_months, df.loan_status_cat, margins=True, margins_name="Total", normalize="index")
# bar plot on categorical variable : term_months
plt.title('Loan status for Term')
sns.barplot(x='term_months', y='loan_status', data=df, color = base_color)
plt.xlabel('Terms in months')
plt.ylabel('Loan Status')
plt.show()
# crosstab between purpose and loan_status
pd.crosstab(df.purpose, df.loan_status_cat, margins=True, margins_name="Total", normalize="index")
## bar plot on categorical variable : purpose
plt.figure(figsize = [12, 5])
sns.barplot(y='purpose', x='loan_status', data=df, color = base_color)
plt.title('Loan Purpose and Status')
plt.xlabel('Purpose')
plt.ylabel('Loan Status')
plt.show()
# crosstab between loan status and issue year
pd.crosstab(df.loan_status_cat, df.issue_d_year, margins=True, margins_name="Total")
## bar plot on categorical variable : issue_d_year
sns.barplot(x='issue_d_year', y='loan_status', data=df,  color = base_color)
plt.show()
## bar plot on categorical variable : home_ownership
sns.barplot(x='home_ownership', y='loan_status', data=df, color = base_color)
plt.show()
# crosstab between emp_length and loan_status
pd.crosstab(df.loan_status_cat, df.emp_length, margins=True, margins_name="Total", normalize="index")
## bar plot on categorical variable : emp_length
plt.figure(figsize=(10,5))
sns.barplot(x='emp_length', y='loan_status', data=df, color = base_color)
plt.show()
df['loan_amnt'].describe().astype('int')
# binning loan_amnt
def loan_amnt(amt):
    if amt <= 5500:
        return 'low'
    elif amt > 5500 and amt <=10000:
        return 'medium'
    elif amt > 10000 and amt <=15000:
        return 'high'
    else:
        return 'very high'

df['loan_amnt_bin'] = df['loan_amnt'].apply(lambda x: loan_amnt(x))
df['loan_amnt_bin'].value_counts()
## bar plot on categorical variable : loan_amnt_bin
loan_order = ['low', 'medium', 'high', 'very high']
sns.barplot(x='loan_amnt_bin', y='loan_status', data=df, color = base_color, order = loan_order)
plt.title('Loan Amount Default Status')
plt.xlabel('Loan Amount')
plt.ylabel('Status')
plt.show()
df['dti'].describe()
# binning debt to income ratio
df['dti_bin'] = pd.cut(df['dti'], 
                                [0,5,10,15,20,25,30], 
                                labels=['0-5','5-10','10-15','15-20','20-25','25-30'])
df['dti_bin'].value_counts()
## bar plot on categorical variable : dti_bin
sns.barplot(x='dti_bin', y='loan_status', data=df, color = base_color)
plt.show()
# installment
def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
df['installment'] = df['installment'].apply(lambda x: installment(x))
## bar plot
sns.barplot(x='installment', y='loan_status', data=df, color = base_color)
plt.show();
# loan purpose Vs Interest Rate
pd.crosstab(df.purpose, df.int_rate_bin, margins=True, margins_name="Total", normalize="index").apply(lambda r: round(100*(r/r.sum())), axis=1)
# Box plot between the loan purpose and interest rate offered
plt.figure(figsize=(20, 10))
sns.boxplot(x='purpose', y='int_rate', data=df, color = base_color);
plt.show();
## bar plot on categorical variable : loan_amnt_bin
sns.barplot(x='loan_amnt_bin', y='int_rate', data=df, color = base_color)
plt.show()
# let's check the top 5 purpose of loans
df.purpose.value_counts()
# let's take top 4 purpose excluding Other since it's detail is not very clear
main_purposes = ["debt_consolidation", "credit_card","home_improvement","major_purchase"]
df = df[df['purpose'].isin(main_purposes)]
df['purpose'].value_counts()
sns.countplot(data = df, x = 'purpose', color = base_color);
# let's now compare the default rates across two types of categorical variables
# purpose of loan (constant) and another categorical variable (which changes)
plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
sns.barplot(x='term_months', y='loan_status', hue="purpose", data=df)
plt.title('Loan Purpose, Term Default Rate')
plt.xlabel('Loan term (Months)')
plt.ylabel('Loan Status')
plt.show();
# Loan Amount
plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
sns.barplot(x='loan_amnt_bin', y='loan_status', hue="int_rate_bin", data=df, order = loan_order)
plt.title('Loan Amount and Interest Default Rate')
plt.xlabel('Loan Amount')
plt.ylabel('Loan Status')
plt.show();
# Installment
plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
sns.barplot(x='annual_inc', y='loan_status', hue="installment", data=df, order = loan_order)
plt.title('Loan Installment, Annual Income Default Rate')
plt.xlabel('Annual Income')
plt.ylabel('Loan Status')
plt.show();
# grade
plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
sns.barplot(x='grade', y='loan_status', hue="purpose", data=df, order = level_order)
plt.title('Loan Purpose, Grade Default Rate')
plt.xlabel('Grades')
plt.ylabel('Loan Status')
plt.show();