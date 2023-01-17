import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# reading data files
import os
print(os.listdir("../input"))
df =  pd.read_csv('../input/loan.csv',dtype='object')
#Look at the loans head
print(df.head())
# inspect the structure
print(df.info(), "\n")
print(df.shape)
# Look if there are any missing values
df.isnull().sum()
# summing up the missing values (column-wise) in master_frame which do not have 100% missing values
Cols_NotNull = round(100*(df.isnull().sum()/len(df.index)), 2) != 100
# Columns having all missing values
df[Cols_NotNull.index[Cols_NotNull == False]].info()
# summing up the missing values (column-wise) in master_frame which do not have 100% missing values
Cols_NotNull = round(100*(df.isnull().sum()/len(df.index)), 2) != 100
# Excluding those columns have 100% missing data 
df = df[Cols_NotNull.index[Cols_NotNull == True]]
df.shape
# summing up the missing values (column-wise) in dataframe
round(100*(df.isnull().sum()/len(df.index)), 2)
#Lets look at the data 
df.head()
# Unique values in the columns
df.nunique()
# Columns having more than one value
cols_unique = df.nunique() != 1
# See the count of columns having more than a single value
cols_unique.value_counts()
# Include only the cols having more than a single value
df = df[cols_unique.index[cols_unique == True]]
df.shape
#drop columns id, member_id, url, desc, zip_code
df = df.drop(['id','member_id','url','desc','zip_code'],axis=1)
#drop column collection_recovery_fee 
df = df.drop(['collection_recovery_fee'],axis=1)
#drop column total_pymnt_inv
df = df.drop(['total_pymnt_inv'],axis=1)
#drop columns total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries
df = df.drop(['total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries'],axis=1)
df.shape
# sum it up to check how many rows have all missing values
df.isnull().all(axis=1).sum()
# look at the df info for number of rows
df.info()
#rows have more than 3 missing values
# calculate the percentage
100*(len(df[df.isnull().sum(axis=1) > 3].index) / len(df.index))
# retaining the rows having <= 3 NaNs
df = df[df.isnull().sum(axis=1) <= 3]

# look at the summary again
round(100*(df.isnull().sum()/len(df.index)), 2)
df.shape
# Lets see the missing values for last_pymnt_d
df[df['last_pymnt_d'].isnull()]['loan_status']
df.shape
# fraction of rows lost
len(df.index)/39717
# Take a look at the spread of the data to see the min and max values
pd.options.display.float_format = "{:.2f}".format
df['annual_inc'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 6000000]
# Define the lables
bucket = ['0-20000', '20000-40000', '40000-60000', '60000-80000', '80000-100000', '100000-120000', 'Above 120000']
# Using the cut function lets create the bins
df['annual_inc_bins'] = pd.cut(df['annual_inc'], bins=cutpoints, labels=bucket)
# Take a look at the spread of the data to see the min and max values
df['loan_amnt'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 5000, 10000, 15000, 20000 , 35000]
# Define the lables
bucket = ['0-5000', '5000-10000', '10000-15000', '15000-20000', 'Above 20000']
# Using the cut function lets create the bins
df['loan_amnt_bins'] = pd.cut(df['loan_amnt'], bins=cutpoints, labels=bucket)
df['installment'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 200, 400, 600, 800, 1306]
# Define the lables
bucket = ['0-200', '200-400', '400-600', '600-800', 'Above 800']
# Using the cut function lets create the bins
df['installment_bins'] = pd.cut(df['installment'], bins=cutpoints, labels=bucket)
df['issue_d_month'] = df['issue_d'].apply(lambda x: x.split('-')[0])
df['mths_since_last_delinq'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 20, 40, 60, 80, 100, 120]
# Define the lables
bucket = ['0-20', '20-40', '40-60', '60-80','80-100','Above 100']
# Using the cut function lets create the bins
df['mths_since_last_delinq_bins'] = pd.cut(df['mths_since_last_delinq'], bins=cutpoints, labels=bucket)
df['mths_since_last_record'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 20, 40, 60, 80, 100, 129]
# Define the lables
bucket = ['0-20', '20-40', '40-60', '60-80','80-100','Above 100']
# Using the cut function lets create the bins
df['mths_since_last_record_bins'] = pd.cut(df['mths_since_last_record'], bins=cutpoints, labels=bucket)
df['open_acc'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 10, 20, 30, 40, 50]
# Define the lables
bucket = ['0-10', '10-20', '20-30', '30-40','Above 40']
# Using the cut function lets create the bins
df['open_acc_bins'] = pd.cut(df['open_acc'], bins=cutpoints, labels=bucket)
df['revol_bal'].describe()
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 30000, 60000, 90000, 120000 , 150000]
# Define the lables
bucket = ['0-30000', '30000-60000', '60000-90000', '90000-120000', 'Above 120000']
# Using the cut function lets create the bins
df['revol_bal_bins'] = pd.cut(df['revol_bal'], bins=cutpoints, labels=bucket)
df['revol_util'].describe()
df['revol_util']= df['revol_util'].apply(lambda x : str(x).rstrip('%'))
df['revol_util']= df['revol_util'].apply(lambda x : str(x).split('.')[0])
df['revol_util']= df['revol_util'].apply(lambda x : str(x).replace('nan','0'))
df['revol_util'] = df['revol_util'].astype('int')

# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Define the lables
bucket = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', 'Above 90']
# Using the cut function lets create the bins
df['revol_util_bins'] = pd.cut(df['revol_util'], bins=cutpoints, labels=bucket)
df['dti'].describe()
### Create bins for Dti
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 5, 10, 15, 20, 25, 30, 35]
# Define the lables
bucket = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', 'Above 30']
# Using the cut function lets create the bins
df['dti_bins'] = pd.cut(df['dti'], bins=cutpoints, labels=bucket)
# plot style used is ggplot used similar in R
plt.style.use('ggplot')
plt.figure(figsize= (5,4))
# frequency plot
sns.countplot(x="term", data=df)
plt.title("Loan Term Count")
#counts
df['term'].value_counts()
# To save the fig to file
plt.savefig('Plot1.png')
plt.figure(figsize= (5,4))
plt.title("Loan Status Count")
sns.countplot(x="loan_status", data=df)
# to show value counts
df['loan_status'].value_counts()
# To save the fig to file
plt.savefig('Plot2.png')
plt.figure(figsize= (10,7))
sns.countplot(y="purpose", data=df)
plt.title("Loan Purpose Count")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
# to show value counts
df['purpose'].value_counts()
# To save the fig to file
plt.savefig('Plot3.png')
plt.figure(figsize= (7,6))
sns.countplot(y="emp_length", data=df)
plt.title("Emp Length Count")
plt.xticks(rotation=45)
df['emp_length'].value_counts()
# To save the fig to file
plt.savefig('Plot4.png')
plt.figure(figsize= (8,7))
sns.countplot(x="annual_inc_bins", data=df)
plt.yticks(rotation=45)
plt.title("Annual Income Bins Count")
plt.xticks(rotation=45)
# To save the fig to file
plt.savefig('Plot5.png')
plt.figure(figsize= (5,7))
sns.countplot(x="loan_amnt_bins", data=df)
plt.xticks(rotation=45)
plt.title('Loan Amount Count')
plt.savefig('Plot6.png')
(df['loan_amnt_bins'].value_counts()/df['loan_amnt_bins'].count()).map(lambda x: '{:,.2%}'.format(x))
sns.countplot(x="installment_bins", data=df)
plt.title("Installment Count")
(df['installment_bins'].value_counts()/df['installment_bins'].count()).map(lambda x: '{:,.2%}'.format(x))
sns.countplot(x="issue_d_month", data=df)
plt.xticks(rotation=45)
plt.title("Loan Issue Month Count")
(df['issue_d_month'].value_counts()/df['issue_d_month'].count()).map(lambda x: '{:,.2%}'.format(x))
plt.figure(figsize= (11,5))
plt.subplot(121)
g = sns.distplot(df['loan_amnt'],bins=5,kde = False)
plt.title("Loan Amnt distribution plot")

plt.subplot(122)
g1 = sns.boxplot(y=df['loan_amnt'])
plt.subplots_adjust(wspace=.2, hspace = 0.3 , top = 0.9)
plt.title("Loan Amnt Boxplot")
plt.figure(figsize= (18,6))
plt.subplot(121)
g = sns.distplot(df['installment'],bins=5,kde = False)
plt.title("Installment distribution plot")

plt.subplot(122)
g1 = sns.boxplot(y=df['installment'])
plt.title("Installment Boxplot")

plt.subplots_adjust(wspace=.2, hspace = 0.3 , top = 0.9)
plt.figure(figsize= (11,5))
plt.subplot(121)
g = sns.distplot(df['annual_inc'],hist = False)
plt.title("Annual Income kdeplot")
plt.xticks(rotation=45)


plt.subplot(122)
g1 = sns.boxplot(y=df['annual_inc'])
g1 = plt.yscale('log')
plt.title("Annual Income Boxplot")

plt.subplots_adjust(wspace=.2, hspace = 0.6 , top =0.9)
# check the data type of int_rate
type(df['int_rate'].iloc[0])
#Interest Rate

# Remove the % symbol
df['int_rate'] = df['int_rate'].str.extract('(\d+)')

# Change the data type to integer
df['int_rate'] = df['int_rate'].astype('int')

# Look at the count of interest rate
plt.figure(figsize= (10,6))
sns.countplot(df['int_rate'])
plt.title('Interest Rate Count')
sns.distplot(df['dti'], bins =10, kde = False)
plt.title("dti: Distribution Plot")
# summary metric
df['dti'].describe()
plt.figure(figsize= (10,6))
sns.countplot(y= 'purpose',hue='loan_status',  data = df)
plt.title("Loan Status distribution by Purpose")
#plt.xticks(rotation=-90)
df[df['loan_status']=='Charged Off'].groupby('purpose')['loan_status'].count()
plt.figure(figsize= (12,12))
sns.countplot(y="addr_state",hue=(df['loan_status']),data=df)
plt.title("Loan Status distribution by Addr State")
df[df['loan_status']=='Charged Off'].groupby('addr_state')[['loan_status']].count().sort_values('loan_status',ascending=False).head(1)
plt.figure(figsize= (8,8))
sns.countplot(x="home_ownership",hue='loan_status', data=df)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,title='loan_status')
plt.title("Loan Status distribution by Home Ownership")
# Lets look at count of home ownership borrowers 
df[df['loan_status']=='Charged Off'].groupby('home_ownership')[['loan_status']].count().sort_values('loan_status',ascending=False).head(1)
plt.figure(figsize= (12,6))
sns.countplot(x="emp_length",hue='loan_status', data=df)
plt.title("Loan Status distribution by Emp Length")
df[df['loan_status']=='Charged Off'].groupby('emp_length')[['loan_status']].count().sort_values('loan_status',ascending=False).head(1)
# adjust figure size
plt.figure(figsize=(15,12))

# subplot 2: Annual Income
plt.subplot(2, 2, 1)
sns.countplot(x="annual_inc_bins",hue='loan_status', data=df)
plt.title("Number of Annual Inc. Bins by Loan Status")
plt.xticks(rotation=30)

# subplot 3: Installment
plt.subplot(2, 2, 2)
sns.countplot(x="verification_status",hue='loan_status', data=df)
plt.title('Number of Ver. Status by loan Status')
plt.xticks(rotation=30)

plt.subplots_adjust(wspace=.5, hspace = 0.7 , top = 0.9)
df[df['loan_status']=='Charged Off'].groupby('annual_inc_bins')[['loan_status']].count().sort_values('loan_status',ascending=False).head(1)
df[df['loan_status']=='Charged Off'].groupby('verification_status')[['loan_status']].count().sort_values('loan_status',ascending=False).head(1)
plt.figure(figsize= (12,8))
sns.countplot(x="open_acc",hue='loan_status', data=df)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,title='loan_status')
plt.title("Loan status distribution by Number of open accounts (credit lines)")

df[df['loan_status']=='Charged Off'].groupby('open_acc')[['loan_status']].count().sort_values('loan_status',ascending=False).head(4)
corr = df[['loan_amnt','total_pymnt','installment','int_rate','annual_inc','dti']].corr()
plt.figure(figsize= (9,6))
plt.title("Correlation Heat Map")
g = sns.heatmap(corr,annot=True)
# adjust figure size
plt.figure(figsize=(15, 8))

# subplot 1: Annual Income
plt.subplot(2, 2, 1)
sns.boxplot(x='annual_inc_bins', y='loan_amnt',hue='loan_status', data=df)
plt.title("Annual Income groups vs Loan Amnt")
plt.xticks(rotation=30)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.1,title='loan_status')

# subplot 2: Loan Amount
plt.subplot(2, 2, 2)
sns.boxplot(x='loan_amnt_bins', y='int_rate',hue='loan_status', data=df)
plt.title("Loan Amount groups vs Interest rate")
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.1,title='loan_status')

# subplot 3: Installment
plt.subplot(2, 2, 3)
sns.boxplot(x='installment_bins',y='loan_amnt',hue='loan_status' , data=df)
plt.title("Installment groups vs Loan Amnt")
           
# subplot 4: Interest Rate
plt.subplot(2, 2, 4)
ax1 =sns.boxplot(x='installment_bins', y='int_rate',hue='loan_status', data=df)
plt.title("Installment groups vs Interest rate")
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.1,title='loan_status')
#plt.yscale('log')

plt.subplots_adjust(wspace=.5, hspace = 0.7 , top = 0.9)
# adjust figure size
plt.figure(figsize=(15, 10))

# subplot 1: 
plt.subplot(3, 3, 1)
# Calculate Charged off % by home_ownership for top 8
ChgOff_hom= (df[df['loan_status']=='Charged Off'].groupby(by=['home_ownership'])[['loan_status']].count()/\
df.groupby(by=['home_ownership'])[['loan_status']].count()).sort_values('loan_status',ascending=False)
ChgOff_hom=ChgOff_hom.reset_index()
ChgOff_hom.rename(columns={"loan_status": "Charged off %"},inplace=True)
#plt.figure(figsize=(8,8))
g=sns.pointplot(x='home_ownership',y='Charged off %',data=ChgOff_hom,color='c',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("home_ownership",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 2: 
plt.subplot(3, 3, 2)
# Calculate Charged off % by emp_length 
ChgOff_emplength= (df[df['loan_status']=='Charged Off'].groupby(by=['emp_length'])[['loan_status']].count()/\
df.groupby(by=['emp_length'])[['loan_status']].count()) 
ChgOff_emplength=ChgOff_emplength.reset_index()
ChgOff_emplength.rename(columns={"loan_status": "Charged off %"},inplace=True)
#plt.figure(figsize=(5,5))
g=sns.pointplot(x='emp_length',y='Charged off %',data=ChgOff_emplength,color='c',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("emp_length",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 3: 
plt.subplot(3, 3, 3)
# Calculate Charged off % by addr_state for top 8
ChgOff_addr= (df[df['loan_status']=='Charged Off'].groupby(by=['addr_state'])[['loan_status']].count()/\
df.groupby(by=['addr_state'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_addr=ChgOff_addr.reset_index()
ChgOff_addr=ChgOff_addr.reset_index()
ChgOff_addr.rename(columns={"loan_status": "Charged off %"},inplace=True)
#plt.figure(figsize=(8,8))
g=sns.pointplot(x='addr_state',y='Charged off %',data=ChgOff_addr,color='c',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=0)
g.set_xlabel("addr_state",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))

# subplot 4: 
plt.subplot(3, 3, 4)
# Calculate Charged off % by Purpose for top 8
ChgOff_purpose= (df[df['loan_status']=='Charged Off'].groupby(by=['purpose'])[['loan_status']].count()/\
df.groupby(by=['purpose'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_purpose=ChgOff_purpose.reset_index()
ChgOff_purpose=ChgOff_purpose.reset_index()
ChgOff_purpose.rename(columns={"loan_status": "Charged off %"},inplace=True)
g=sns.pointplot(x='purpose',y='Charged off %',data=ChgOff_purpose,color='c',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("purpose",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 5: 
plt.subplot(3, 3, 5)
# Calculate Charged off % by annual income bins 
ChgOff_inc= (df[df['loan_status']=='Charged Off'].groupby(by=['annual_inc_bins'])[['loan_status']].count()/\
df.groupby(by=['annual_inc_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False)
ChgOff_inc=ChgOff_inc.reset_index()
ChgOff_inc.rename(columns={"loan_status": "Charged off %"},inplace=True)
g=sns.pointplot(x='annual_inc_bins',y='Charged off %',data=ChgOff_inc,color='c',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("annual_inc_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 3:#plt.subplot(3, 3, 6)

plt.subplots_adjust(wspace=.5, hspace = 0.8 , top = 1.0)
# adjust figure size
plt.figure(figsize=(15, 10))

# subplot 1: 
plt.subplot(3, 3, 1)
# Calculate Charged off % by no of months since last record
ChgOff_mnthLastRcrd= (df[df['loan_status']=='Charged Off'].groupby(by=['mths_since_last_record_bins'])[['loan_status']].count()/\
df.groupby(by=['mths_since_last_record_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8)
ChgOff_mnthLastRcrd=ChgOff_mnthLastRcrd.reset_index()
ChgOff_mnthLastRcrd.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='mths_since_last_record_bins',y='Charged Off %',data=ChgOff_mnthLastRcrd,color='r',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("mths_since_last_record_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 2: 
plt.subplot(3, 3, 2)
# Calculate Charged off % by no of open accounts
ChgOff_opnAcc= (df[df['loan_status']=='Charged Off'].groupby(by=['open_acc_bins'])[['loan_status']].count()/\
df.groupby(by=['open_acc_bins'])[['loan_status']].count()) 
ChgOff_opnAcc=ChgOff_opnAcc.reset_index()
ChgOff_opnAcc.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='open_acc_bins',y='Charged Off %',data=ChgOff_opnAcc,color='r',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("open_acc_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 3: 
plt.subplot(3, 3, 3)
# Calculate Charged off % by public record bankruptcies
ChgOff_pblcBankrpt= (df[df['loan_status']=='Charged Off'].groupby(by=['pub_rec_bankruptcies'])[['loan_status']].count()/\
df.groupby(by=['pub_rec_bankruptcies'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_pblcBankrpt=ChgOff_pblcBankrpt.reset_index()
ChgOff_pblcBankrpt.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='pub_rec_bankruptcies',y='Charged Off %',data=ChgOff_pblcBankrpt,color='r',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=0)
g.set_xlabel("pub_rec_bankruptcies",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))

# subplot 4: 
plt.subplot(3, 3, 4)
# Calculate Charged off % by public record
ChgOff_dti= (df[df['loan_status']=='Charged Off'].groupby(by=['pub_rec'])[['loan_status']].count()/\
df.groupby(by=['pub_rec'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_dti=ChgOff_dti.reset_index()
ChgOff_dti=ChgOff_dti.reset_index()
ChgOff_dti.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='pub_rec',y='Charged Off %',data=ChgOff_dti,color='r',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("pub_rec",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 5: 
plt.subplot(3, 3, 5)
# Calculate Charged off % by revolving balance
ChgOff_revBalance= (df[df['loan_status']=='Charged Off'].groupby(by=['revol_bal_bins'])[['loan_status']].count()/\
df.groupby(by=['revol_bal_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(19)
ChgOff_revBalance=ChgOff_revBalance.reset_index()
ChgOff_revBalance.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='revol_bal_bins',y='Charged Off %',data=ChgOff_revBalance,color='r',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("revol_bal_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 6:
plt.subplot(3, 3, 6)
# Calculate Charged off % by revolving utilization
ChgOff_revUtlz= (df[df['loan_status']=='Charged Off'].groupby(by=['revol_util_bins'])[['loan_status']].count()/\
df.groupby(by=['revol_util_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(19)
ChgOff_revUtlz=ChgOff_revUtlz.reset_index()
ChgOff_revUtlz.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='revol_util_bins',y='Charged Off %',data=ChgOff_revUtlz,color='r',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("revol_util_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))

plt.subplots_adjust(wspace=.5, hspace = 0.8 , top = 1.0)
# adjust figure size
plt.figure(figsize=(10, 10))

# subplot 1: 
plt.subplot(2, 2, 1)
# Calculate Charged off % by issued_month
ChgOff_issue= (df[df['loan_status']=='Charged Off'].groupby(by=['issue_d_month'])[['loan_status']].count()/\
df.groupby(by=['issue_d_month'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8)
ChgOff_issue=ChgOff_issue.reset_index()
ChgOff_issue.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='issue_d_month',y='Charged Off %',data=ChgOff_issue,color='b',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("issue_d_month",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 2: 
plt.subplot(2, 2, 2)
# Calculate Charged off % by delinq within 2yrs
ChgOff_delinq_2yrs= (df[df['loan_status']=='Charged Off'].groupby(by=['delinq_2yrs'])[['loan_status']].count()/\
df.groupby(by=['delinq_2yrs'])[['loan_status']].count()) 
ChgOff_delinq_2yrs=ChgOff_delinq_2yrs.reset_index()
ChgOff_delinq_2yrs.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='delinq_2yrs',y='Charged Off %',data=ChgOff_delinq_2yrs,color='b',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("delinq_2yrs",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 3: 
plt.subplot(2, 2, 3)
# Calculate Charged off % by inq in last months
ChgOff_inq= (df[df['loan_status']=='Charged Off'].groupby(by=['inq_last_6mths'])[['loan_status']].count()/\
df.groupby(by=['inq_last_6mths'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_inq=ChgOff_inq.reset_index()
ChgOff_inq.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='inq_last_6mths',y='Charged Off %',data=ChgOff_inq,color='b',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("inq_last_6mths",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 4: 
plt.subplot(2, 2, 4)
# Calculate Charged off % by no of months since last delinq
ChgOff_month_last_delinq= (df[df['loan_status']=='Charged Off'].groupby(by=['mths_since_last_delinq_bins'])[['loan_status']].count()/\
df.groupby(by=['mths_since_last_delinq_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(19)
ChgOff_month_last_delinq=ChgOff_month_last_delinq.reset_index()
ChgOff_month_last_delinq.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='mths_since_last_delinq_bins',y='Charged Off %',data=ChgOff_month_last_delinq,color='b',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("mths_since_last_delinq_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


plt.subplots_adjust(wspace=.5, hspace = 0.8 , top = 1.0)
# adjust figure size
plt.figure(figsize=(15, 10))

# subplot 1: 
plt.subplot(3, 3, 1)
# Calculate Charged off % by Verification Status
ChgOff_ver= (df[df['loan_status']=='Charged Off'].groupby(by=['verification_status'])[['loan_status']].count()/\
df.groupby(by=['verification_status'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8)
ChgOff_ver=ChgOff_ver.reset_index()
ChgOff_ver.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='verification_status',y='Charged Off %',data=ChgOff_ver,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("verification_status",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 2: 
plt.subplot(3, 3, 2)
# Calculate Charged off % by term
ChgOff_term= (df[df['loan_status']=='Charged Off'].groupby(by=['term'])[['loan_status']].count()/\
df.groupby(by=['term'])[['loan_status']].count()) 
ChgOff_term=ChgOff_term.reset_index()
ChgOff_term.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='term',y='Charged Off %',data=ChgOff_term,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("term",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 3: 
plt.subplot(3, 3, 3)
# Calculate Charged off % by Grade
ChgOff_grade= (df[df['loan_status']=='Charged Off'].groupby(by=['grade'])[['loan_status']].count()/\
df.groupby(by=['grade'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_grade=ChgOff_grade.reset_index()
ChgOff_grade=ChgOff_grade.reset_index()
ChgOff_grade.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='grade',y='Charged Off %',data=ChgOff_grade,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=0)
g.set_xlabel("grade",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))

# subplot 4: 
plt.subplot(3, 3, 4)
# Calculate Charged off % by dti
ChgOff_dti= (df[df['loan_status']=='Charged Off'].groupby(by=['dti_bins'])[['loan_status']].count()/\
df.groupby(by=['dti_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8) 
ChgOff_dti=ChgOff_dti.reset_index()
ChgOff_dti=ChgOff_dti.reset_index()
ChgOff_dti.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='dti_bins',y='Charged Off %',data=ChgOff_dti,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("dti",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 5: 
plt.subplot(3, 3, 5)
# Calculate Charged off % by annual Interest Rate
ChgOff_int= (df[df['loan_status']=='Charged Off'].groupby(by=['int_rate'])[['loan_status']].count()/\
df.groupby(by=['int_rate'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(19)
ChgOff_int=ChgOff_int.reset_index()
ChgOff_int.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='int_rate',y='Charged Off %',data=ChgOff_int,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("interest_rate",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 6:
plt.subplot(3, 3, 6)
# Calculate Charged off % by annual Loan Amount
ChgOff_amnt= (df[df['loan_status']=='Charged Off'].groupby(by=['loan_amnt_bins'])[['loan_status']].count()/\
df.groupby(by=['loan_amnt_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(19)
ChgOff_amnt=ChgOff_amnt.reset_index()
ChgOff_amnt.rename(columns={"loan_status": "Charged Off %"},inplace=True)
g=sns.pointplot(x='loan_amnt_bins',y='Charged Off %',data=ChgOff_amnt,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("loan_amnt_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))

plt.subplots_adjust(wspace=.5, hspace = 0.8 , top = 1.0)
df['month_inc']=df['annual_inc']/12
df['InHandSalary'] = df['month_inc'] - (df['dti']*df['month_inc']/100)
# Lets define the cutpoints by creating a list based on min and the max values
cutpoints = [0, 1000, 2000, 3000, 4000, 5000, 6000, 500000]
# Define the lables
bucket = ['0-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000','5000-6000','Above 6000']
# Using the cut function lets create the bins
df['InHandmonth_inc_bins'] = pd.cut(df['InHandSalary'], bins=cutpoints, labels=bucket)
# Calculate Charged off % by Monthly In Hand Income after paying all EMIs
ChgOff_hand= (df[df['loan_status']=='Charged Off'].groupby(by=['InHandmonth_inc_bins'])[['loan_status']].count()/\
df.groupby(by=['InHandmonth_inc_bins'])[['loan_status']].count()).sort_values('loan_status',ascending=False).head(8)
ChgOff_hand=ChgOff_hand.reset_index()
ChgOff_hand.rename(columns={"loan_status": "Charged Off %"},inplace=True)
#plt.figure(figsize=(8,8))
g=sns.pointplot(x='InHandmonth_inc_bins',y='Charged Off %',data=ChgOff_hand,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("InHandmonth_inc_bins",fontsize=13)
g.set_ylabel("ChargedOff %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 
# adjust figure size
plt.figure(figsize=(15, 10))

# subplot 1: 
plt.subplot(2, 2, 1)
# Calculate Verification status rate by no of open accounts
v_annInc= (df[df['verification_status']=='Not Verified'].groupby(by=['annual_inc_bins'])[['verification_status']].count()/\
df.groupby(by=['annual_inc_bins'])[['verification_status']].count()).sort_values('verification_status',ascending=False).head(8)
v_annInc=v_annInc.reset_index()
v_annInc.rename(columns={"verification_status": "verification status %"},inplace=True)
#plt.figure(figsize=(8,8))
g=sns.pointplot(x='annual_inc_bins',y='verification status %',data=v_annInc,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("annual_inc_bins",fontsize=13)
g.set_ylabel("Not_verified %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 2: 
plt.subplot(2, 2, 2)
# Calculate Verification status rate by home ownership
v_hmOwn= (df[df['verification_status']=='Not Verified'].groupby(by=['home_ownership'])[['verification_status']].count()/\
df.groupby(by=['home_ownership'])[['verification_status']].count()) 
v_hmOwn=v_hmOwn.reset_index()
v_hmOwn.rename(columns={"verification_status": "verification status %"},inplace=True)
#plt.figure(figsize=(5,5))
g=sns.pointplot(x='home_ownership',y='verification status %',data=v_hmOwn,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("home_ownership",fontsize=13)
g.set_ylabel("Not_verified %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y))) 

# subplot 3: 
plt.subplot(2, 2, 3)
# Calculate Verification status rate by state
v_state= (df[df['verification_status']=='Not Verified'].groupby(by=['addr_state'])[['verification_status']].count()/\
df.groupby(by=['addr_state'])[['verification_status']].count()).sort_values('verification_status',ascending=False).head(8) 
v_state=v_state.reset_index()
v_state.rename(columns={"verification_status": "verification status %"},inplace=True)
g=sns.pointplot(x='addr_state',y='verification status %',data=v_state,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=90)
g.set_xlabel("addr_state",fontsize=13)
g.set_ylabel("Not_verified %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


# subplot 4: 
plt.subplot(2, 2, 4)
# Calculate Verification status rate by Monthly InHand Savings (Available Monthly Income after paying all EMIs)
v_InHndIncm= (df[df['verification_status']=='Not Verified'].groupby(by=['InHandmonth_inc_bins'])[['verification_status']].count()/\
df.groupby(by=['InHandmonth_inc_bins'])[['verification_status']].count()).sort_values('verification_status',ascending=False).head(19)
v_InHndIncm=v_InHndIncm.reset_index()
v_InHndIncm.rename(columns={"verification_status": "verification status %"},inplace=True)
#plt.figure(figsize=(8,8))
g=sns.pointplot(x='InHandmonth_inc_bins',y='verification status %',data=v_InHndIncm,color='g',markers=['o'], 
               scale = 0.4)
plt.xticks(rotation=30)
g.set_xlabel("InHandmonth_inc_bins",fontsize=13)
g.set_ylabel("Not_verified %",fontsize=13)
g.tick_params(labelsize=12)
from matplotlib.ticker import FuncFormatter
g.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))


plt.subplots_adjust(wspace=.5, hspace = 0.8 , top = 1.0)