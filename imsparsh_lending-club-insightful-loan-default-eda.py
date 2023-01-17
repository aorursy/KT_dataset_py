# import required libraries

import numpy as np

print('numpy version:',np.__version__)

import pandas as pd

print('pandas version:',pd.__version__)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None

pd.options.display.float_format = '{:.2f}'.format

pd.set_option('display.max_columns', 200)

pd.set_option('display.width', 400)
# file path variable

case_data = "/kaggle/input/lending-club-loan-dataset-2007-2011/loan.csv"

loan = pd.read_csv(case_data, low_memory=False)

loan.head()
#shape of the dataset

print("Number of columns",len(loan.columns))

print("Number of rows",len(loan.index))
# plotting pie chart for different types of loan_status

chargedOffLoans = loan.loc[(loan["loan_status"] == "Charged Off")]

currentLoans = loan.loc[(loan["loan_status"] == "Current")]

fullyPaidLoans = loan.loc[(loan["loan_status"]== "Fully Paid")]

data  = [{"Charged Off": chargedOffLoans["funded_amnt_inv"].sum(),"Fully Paid":fullyPaidLoans["funded_amnt_inv"].sum(),"Current":currentLoans["funded_amnt_inv"].sum()}]

investment_sum = pd.DataFrame(data) 

chargedOffTotalSum = float(investment_sum["Charged Off"])

fullyPaidTotalSum = float(investment_sum["Fully Paid"])

currentTotalSum = float(investment_sum["Current"])

loan_status = [chargedOffTotalSum,fullyPaidTotalSum,currentTotalSum]

loan_status_labels = 'Charged Off','Fully Paid','Current'

plt.pie(loan_status,labels=loan_status_labels,autopct='%1.1f%%')

plt.title('Loan Status Aggregate Information')

plt.axis('equal')

plt.legend(loan_status,title="Loan Amount",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()
# plotting pie chart for different types of purpose

loans_purpose = loan.groupby(['purpose'])['funded_amnt_inv'].sum().reset_index()

plt.figure(figsize=(14, 10))

plt.pie(loans_purpose["funded_amnt_inv"],labels=loans_purpose["purpose"],autopct='%1.1f%%')

plt.title('Loan purpose Aggregate Information')

plt.axis('equal')

plt.legend(loan_status,title="Loan purpose",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()
if(len(loan) == len(loan.member_id.unique())):

    print("No duplicate data found!")

else:

    print("Some duplicates occur.")
print("Completely Null values:")

print(list(loan.columns[round(loan.isnull().sum()/len(loan.index), 2)*100 == 100]))
# we can see around half of the columns are completely null

# remove all columns having no values

loan = loan.dropna(axis=1, how="all")

print("Looking into remaining columns info:")

print(loan.info(max_cols=200))
# remove non-required columns

# id - not required

# member_id - not required

# acc_now_delinq - empty

# funded_amnt - not useful, funded_amnt_inv is useful which is funded to person

# emp_title - brand names not useful

# pymnt_plan - fixed value as n for all

# url - not useful

# desc - can be applied some NLP but not for EDA

# title - too many distinct values not useful

# zip_code - complete zip is not available

# delinq_2yrs - post approval feature

# mths_since_last_delinq - only half values are there, not much information

# mths_since_last_record - only 10% values are there

# revol_bal - post/behavioural feature

# initial_list_status - fixed value as f for all

# out_prncp - post approval feature

# out_prncp_inv - not useful as its for investors

# total_pymnt - post approval feature

# total_pymnt_inv - not useful as it is for investors

# total_rec_prncp - post approval feature

# total_rec_int - post approval feature

# total_rec_late_fee - post approval feature

# recoveries - post approval feature

# collection_recovery_fee - post approval feature

# last_pymnt_d - post approval feature

# last_credit_pull_d - irrelevant for approval

# last_pymnt_amnt - post feature

# next_pymnt_d - post feature

# collections_12_mths_ex_med - only 1 value 

# policy_code - only 1 value

# acc_now_delinq - single valued

# chargeoff_within_12_mths - post feature

# delinq_amnt - single valued

# tax_liens - single valued

# application_type - single

# pub_rec_bankruptcies - single valued for more than 99%

# addr_state - may not depend on location as its in financial domain



colsToDrop = ["id", "member_id", "funded_amnt", "emp_title", "pymnt_plan", "url", "desc", "title", "zip_code", "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record", "revol_bal", "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "collections_12_mths_ex_med", "policy_code", "acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt", "tax_liens", "application_type", "pub_rec_bankruptcies", "addr_state"]

loan.drop(colsToDrop, axis=1, inplace=True)

print("Features we are left with",list(loan.columns))

# loan.info(max_cols=100)
# find columns with any null values

loan.columns[loan.isna().any()]
# find most common value in emp_length to impute

print(loan["emp_length"].mode())
# in 12 unique values we have 10+ years the most for emp_length, 

# but it is highly dependent variable so we will not impute the values 

# but remove the rows with null values which is around 2.5%

loan.dropna(axis=0, subset=["emp_length"], inplace=True)

print(loan.columns[loan.isna().any()])
# remove NA rows for revol_util as its dependent variable and is around 0.1%

loan.dropna(axis=0, subset=["revol_util"], inplace=True)

print(loan.columns[loan.isna().any()])
# update int_rate and revol_util without % sign and save them as numeric type

loan["int_rate"] = pd.to_numeric(loan["int_rate"].apply(lambda x:x.split('%')[0]))

loan["revol_util"] = pd.to_numeric(loan["revol_util"].apply(lambda x:x.split('%')[0]))

# remove text data from term feature and store as numerical

loan["term"] = pd.to_numeric(loan["term"].apply(lambda x:x.split()[0]))

loan[["int_rate", "revol_util", "term"]].head()
# remove the rows with loan_status as "Current"

loan = loan[loan["loan_status"].apply(lambda x:False if x == "Current" else True)]

print(loan["loan_status"].unique())

# update loan_status as Fully Paid to 0 and Charged Off to 1

loan["loan_status"] = loan["loan_status"].apply(lambda x: 0 if x == "Fully Paid" else 1)

print(loan["loan_status"].value_counts())
# update emp_length feature with continuous values as int

# where (< 1 year) is assumed as 0 and 10+ years is assumed as 10 and rest are stored as their magnitude

loan["emp_length"] = pd.to_numeric(loan["emp_length"].apply(lambda x:0 if "<" in x else (x.split('+')[0] if "+" in x else x.split()[0])))

loan["emp_length"].value_counts()
# look through the purpose value counts

loan_purpose_values = loan["purpose"].value_counts()*100/loan.shape[0]

print(loan_purpose_values)

# remove rows with less than 1% of value counts in paricular purpose 

loan_purpose_delete = loan_purpose_values[loan_purpose_values<1].index.values

print("Removing rows with purpose as",loan_purpose_delete)
loan = loan[[False if p in loan_purpose_delete else True for p in loan["purpose"]]]

print("Available purpose types:")

print(loan["purpose"].value_counts()*100/loan.shape[0])
# for annual_inc, the highest value is 6000000 where 75% quantile value is 83000, and is 100 times the mean

loan["annual_inc"].quantile(0.99)
# we need to remomve outliers from annual_inc i.e. 99 to 100%

annual_inc_q = loan["annual_inc"].quantile(0.99)

loan = loan[loan["annual_inc"] < annual_inc_q]

loan["annual_inc"].describe()
# for open_acc, the highest value is 44 where 75% quantile value is 12, and is 5 times the mean

loan["open_acc"].quantile(0.999)
# we need to remomve outliers from open_acc i.e. 99.9 to 100%

open_acc_q = loan["open_acc"].quantile(0.999)

loan = loan[loan["open_acc"] < open_acc_q]

loan["open_acc"].describe()
# for total_acc, the highest value is 90 where 75% quantile value is 29, and is 4 times the mean

loan["total_acc"].quantile(0.98)
# we need to remomve outliers from total_acc i.e. 98 to 100%

total_acc_q = loan["total_acc"].quantile(0.98)

loan = loan[loan["total_acc"] < total_acc_q]

loan["total_acc"].describe()
# for pub_rec, the highest value is 4 where 75% quantile value is 0, and is 4 times the mean

loan["pub_rec"].quantile(0.995)
# we need to remomve outliers from pub_rec i.e. 99.5 to 100%

pub_rec_q = loan["pub_rec"].quantile(0.995)

loan = loan[loan["pub_rec"] <= pub_rec_q]

loan["pub_rec"].describe()
# all values seems fine now, later will be checked while plotting boxplots

loan.describe()
def standerdisedate(date):

    year = date.split("-")[0]

    if(len(year) == 1):

        date = "0"+date

    return date

# this pattern works on mac but did not work on the windows 

# loan['issue_d'] = loan['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

# use this command on windows 

# loan['issue_d'] = loan['issue_d'].apply(lambda x: datetime.strptime(x, '%y-%b'))

from datetime import datetime

loan['issue_d'] = loan['issue_d'].apply(lambda x:standerdisedate(x))

loan['issue_d'] = loan['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

# extracting month and year from issue_date

loan['month'] = loan['issue_d'].apply(lambda x: x.month)

loan['year'] = loan['issue_d'].apply(lambda x: x.year)
# get year from issue_d and replace the same

loan["earliest_cr_line"] = pd.to_numeric(loan["earliest_cr_line"].apply(lambda x:x.split('-')[1]))

loan[["issue_d", "earliest_cr_line"]].head()
# create bins for loan_amnt range

bins = [0, 5000, 10000, 15000, 20000, 25000, 36000]

bucket_l = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000','25000+']

loan['loan_amnt_range'] = pd.cut(loan['loan_amnt'], bins, labels=bucket_l)
# create bins for int_rate range

bins = [0, 7.5, 10, 12.5, 15, 100]

bucket_l = ['0-7.5', '7.5-10', '10-12.5', '12.5-15', '15+']

loan['int_rate_range'] = pd.cut(loan['int_rate'], bins, labels=bucket_l)
# create bins for annual_inc range

bins = [0, 25000, 50000, 75000, 100000, 1000000]

bucket_l = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000+']

loan['annual_inc_range'] = pd.cut(loan['annual_inc'], bins, labels=bucket_l)
# create bins for installment range

def installment(n):

    if n <= 200:

        return 'low'

    elif n > 200 and n <=500:

        return 'medium'

    elif n > 500 and n <=800:

        return 'high'

    else:

        return 'very high'



loan['installment'] = loan['installment'].apply(lambda x: installment(x))
# create bins for dti range

bins = [-1, 5.00, 10.00, 15.00, 20.00, 25.00, 50.00]

bucket_l = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25%+']

loan['dti_range'] = pd.cut(loan['dti'], bins, labels=bucket_l)

loan[["loan_amnt_range", "annual_inc_range", "int_rate_range", "dti_range"]].head()
# check for amount of defaulters in the data using countplot

plt.figure(figsize=(14,5))

sns.countplot(y="loan_status", data=loan)

plt.show()
# function for plotting the count plot features wrt default ratio

def plotUnivariateRatioBar(feature, data=loan, figsize=(10,5), rsorted=True):

    plt.figure(figsize=figsize)

    if rsorted:

        feature_dimension = sorted(data[feature].unique())

    else:

        feature_dimension = data[feature].unique()

    feature_values = []

    for fd in feature_dimension:

        feature_filter = data[data[feature]==fd]

        feature_count = len(feature_filter[feature_filter["loan_status"]==1])

        feature_values.append(feature_count*100/feature_filter["loan_status"].count())

    plt.bar(feature_dimension, feature_values, color='orange', edgecolor='white')

    plt.title("Loan Defaults wrt "+str(feature)+" feature - countplot")

    plt.xlabel(feature, fontsize=16)

    plt.ylabel("defaulter %", fontsize=16)

    plt.show()



def plotUnivariateBar(x, figsize=(10,5)):

    plt.figure(figsize=figsize)

    sns.barplot(x=x, y='loan_status', data=loan)

    plt.title("Loan Defaults wrt "+str(x)+" feature - countplot")

    plt.xlabel(x, fontsize=16)

    plt.ylabel("defaulter ratio", fontsize=16)

    plt.show()
# check for defaulters wrt term in the data using countplot

plotUnivariateBar("term", figsize=(8,5))
# check for defaulters wrt grade in the data using countplot

plotUnivariateRatioBar("grade")
# check for defaulters wrt sub_grade in the data using countplot

plotUnivariateBar("sub_grade", figsize=(16,5))
# check for defaulters wrt home_ownership in the data using countplot

plotUnivariateRatioBar("home_ownership")
# check for defaulters wrt verification_status in the data using countplot

plotUnivariateRatioBar("verification_status")
# check for defaulters wrt purpose in the data using countplot

plotUnivariateBar("purpose", figsize=(16,6))
# check for defaulters wrt open_acc in the data using countplot

plotUnivariateRatioBar("open_acc", figsize=(16,6))
# check for defaulters wrt pub_rec in the data using countplot

plotUnivariateRatioBar("pub_rec")
# check for defaulters wrt emp_length in the data using countplot

plotUnivariateBar("emp_length", figsize=(14,6))
# check for defaulters wrt month in the data using countplot

plotUnivariateBar("month", figsize=(14,6))
# check for defaulters wrt year in the data using countplot

plotUnivariateBar("year")
# check for defaulters wrt earliest_cr_line in the data using countplot

plotUnivariateBar("earliest_cr_line", figsize=(16,10))
# check for defaulters wrt inq_last_6mths in the data using countplot

plotUnivariateBar("inq_last_6mths")
# check for defaulters wrt revol_util in the data using countplot

plotUnivariateRatioBar("revol_util", figsize=(16,6))
# check for defaulters wrt total_acc in the data using countplot

plotUnivariateRatioBar("total_acc", figsize=(14,6))
# check for defaulters wrt loan_amnt_range in the data using countplot

plotUnivariateBar("loan_amnt_range")
# check for defaulters wrt int_rate_range in the data using countplot

plotUnivariateBar("int_rate_range")
# check for defaulters wrt annual_inc_range in the data using countplot

plotUnivariateBar("annual_inc_range")
# check for defaulters wrt dti_range in the data using countplot

plotUnivariateBar("dti_range", figsize=(16,5))
# check for defaulters wrt installment range in the data using countplot

plotUnivariateBar("installment", figsize=(8,5))
# function to plot scatter plot for two features

def plotScatter(x, y):

    plt.figure(figsize=(16,6))

    sns.scatterplot(x=x, y=y, hue="loan_status", data=loan)

    plt.title("Scatter plot between "+x+" and "+y)

    plt.xlabel(x, fontsize=16)

    plt.ylabel(y, fontsize=16)

    plt.show()

    

def plotBivariateBar(x, hue, figsize=(16,6)):

    plt.figure(figsize=figsize)

    sns.barplot(x=x, y='loan_status', hue=hue, data=loan)

    plt.title("Loan Default ratio wrt "+x+" feature for hue "+hue+" in the data using countplot")

    plt.xlabel(x, fontsize=16)

    plt.ylabel("defaulter ratio", fontsize=16)

    plt.show()
# check for defaulters wrt annual_inc and purpose in the data using countplot

plotBivariateBar("annual_inc_range", "purpose")
# check for defaulters wrt term and purpose in the data using countplot

plotBivariateBar("term", "purpose")
# check for defaulters wrt grade and purpose in the data using countplot

plotBivariateBar("grade", "purpose")
# check for defaulters wrt loan_amnt_range and purpose in the data using countplot

plotBivariateBar("loan_amnt_range", "purpose")
# check for defaulters wrt loan_amnt_range and term in the data using countplot

plotBivariateBar("loan_amnt_range", "term")
# check for defaulters wrt annual_inc_range and purpose in the data using countplot

plotBivariateBar("annual_inc_range", "purpose")
# check for defaulters wrt annual_inc_range and purpose in the data using countplot

plotBivariateBar("installment", "purpose")
# check for defaulters wrt loan_amnt_range in the data using countplot

plotScatter("int_rate", "annual_inc")
# plot scatter for funded_amnt_inv with dti

plotScatter("funded_amnt_inv", "dti")
# plot scatter for funded_amnt_inv with annual_inc

plotScatter("annual_inc", "funded_amnt_inv")
# plot scatter for loan_amnt with int_rate

plotScatter("loan_amnt", "int_rate")
# plot scatter for int_rate with annual_inc

plotScatter("int_rate", "annual_inc")
# plot scatter for earliest_cr_line with int_rate

plotScatter("earliest_cr_line", "int_rate")
# plot scatter for annual_inc with emp_length

plotScatter("annual_inc", "emp_length")
# plot scatter for earliest_cr_line with dti

plotScatter("earliest_cr_line", "dti")
sorted(loan["grade"].unique())
# function to plot boxplot for comparing two features

def plotBox(x, y, hue="loan_status"):

    plt.figure(figsize=(16,6))

    sns.boxplot(x=x, y=y, data=loan, hue=hue, order=sorted(loan[x].unique()))

    plt.title("Box plot between "+x+" and "+y+" for each "+hue)

    plt.xlabel(x, fontsize=16)

    plt.ylabel(y, fontsize=16)

    plt.show()

    plt.figure(figsize=(16,8))

    sns.violinplot(x=x, y=y, data=loan, hue=hue, order=sorted(loan[x].unique()))

    plt.title("Violin plot between "+x+" and "+y+" for each "+hue)

    plt.xlabel(x, fontsize=16)

    plt.ylabel(y, fontsize=16)

    plt.show()
# plot box for term vs int_rate for each loan_status

plotBox("term", "int_rate")
# plot box for loan_status vs int_rate for each purpose

plotBox("loan_status", "int_rate", hue="purpose")
# plot box for purpose vs revo_util for each status

plotBox("purpose", "revol_util")
# plot box for grade vs int_rate for each loan_status

plotBox("grade", "int_rate", "loan_status")
# plot box for issue_d vs int_rate for each loan_status

plotBox("month", "int_rate", "loan_status")
# plot heat map to see correlation between features

continuous_f = ["funded_amnt_inv", "annual_inc", "term", "int_rate", "loan_status", "revol_util", "pub_rec", "earliest_cr_line"]

loan_corr = loan[continuous_f].corr()

sns.heatmap(loan_corr,vmin=-1.0,vmax=1.0,annot=True, cmap="YlGnBu")

plt.title("Correlation Heatmap")

plt.show()