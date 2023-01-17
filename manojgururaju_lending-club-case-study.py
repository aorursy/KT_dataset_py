# Importing libraries



import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.gridspec as gridspec

import matplotlib.pylab as pl

import seaborn as sns

import matplotlib.pyplot as plt

import os 

from datetime import datetime

from datetime import date



# Creating apth folder

DF = pd.read_csv('../input/loan-dataset/loan.csv', engine='python',encoding='ISO-8859-1')

DF.head()
# Checking number of rows corresponding to loan status



DF.groupby('loan_status')['id'].count()
# Creating dataframe where loan status is Fully Paid or Charged Off



DF=DF.loc[DF.loan_status.isin(['Fully Paid','Charged Off'])]



#Creating a flag for default status



DF['Default_Status']=DF['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
#Checking % of rows having NA in differt columns

Chk_1=round(100*(DF.isnull().sum()/len(DF.index)),2).reset_index()

Chk_1.rename(columns={'index': 'Column', 0:'Percent Missing'}, inplace=True)
# Saving Chk_1 as csv

# Chk_1.to_csv(os.path.join(path_folder,r'Check 1.csv'))
# Removing columns with all Null values

DF1=pd.concat([DF.iloc[:,0:50], DF.iloc[:,51:53], DF.iloc[:,56:57], DF.iloc[:,78:80], 

               DF.iloc[:,105:107], DF.iloc[:,111:112]],axis = 1)
#Checking % of rows having NA in differt columns

round(100*(DF1.isnull().sum()/len(DF1.index)),2)
# Removing columns related to customer beahviour varibales i.e. factors established after loan in granted

DF2=pd.concat([DF1.iloc[:,0:1], DF1.iloc[:,2:17], DF1.iloc[:,19:21], DF1.iloc[:,23:25], DF1.iloc[:,26:28], 

              DF1.iloc[:,30:32], DF1.iloc[:,34:36], DF1.iloc[:,48:49], DF1.iloc[:,55:58]], axis = 1)
DF2.head
#Checking % of rows having NA in differt columns

round(100*(DF2.isnull().sum()/len(DF2.index)),2)
# Creating a heatmap for visualizing Null values in dataset



pl.figure(figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')

sns.heatmap(DF2.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.plot
# Creating a histogram to check frequnecy in column pub_rec_bankruptcies



DF2['pub_rec_bankruptcies'].hist()

# Since 0 dominates the data, there is very little information that can be derived from this column

# We deem this column insgnificant and hence remove this column
# Creating a histogram to check frequnecy in column tax_liens



DF2['tax_liens'].hist()

# Since 0 dominates the data, there is very little information that can be derived from this column

# We deem this column insgnificant and hence remove this column
DF2=pd.concat([DF2.iloc[:,0:20], DF2.iloc[:,21:28], DF2.iloc[:,30:31]],axis = 1)
# Checking the dataset statistics



DF2.describe()
# Checking the data type of columns in DF2



DF2.dtypes
# Modifying column term to make it of type int64 from object type



DF2['term']=DF2['term'].str.split(' ').str[1].astype(np.int64)
# Modifying column int_rate to make it of type float64 from object type



DF2['int_rate']=DF2['int_rate'].str.split('%').str[0].astype(np.float64)
# Creating a new column Emp_Len from emp_length to make it of type float64 from object type



DF2_10=DF2.loc[DF2.emp_length.isin(['10+ years'])]

Emp=DF2_10['emp_length'].str.split(' ').str[0]

DF2_10['Emp_Len']=Emp.str.split('+').str[0].astype(np.float64)





DF2_1_9=DF2.loc[~DF2.emp_length.isin(['10+ years','< 1 year'])]

DF2_1_9['Emp_Len']=DF2_1_9['emp_length'].str.split(' ').str[0].astype(np.float64)





DF2_1=DF2.loc[DF2.emp_length.isin(['< 1 year'])]

DF2_1['Emp_Len']=DF2_1['emp_length'].str.split(' ').str[1].astype(np.float64)



DF3 = DF2_1.append(DF2_1_9.append(DF2_10, ignore_index = True), ignore_index = True)
# Creating datatype from object to datetime



DF3['today'] = datetime.today()

DF3['issue_d']=pd.to_datetime(DF3['issue_d'], format='%b-%y')

DF3['last_credit_pull_d']=pd.to_datetime(DF3['last_credit_pull_d'], format='%b-%y')
# Creating dervied columns from date columns



DF3['months_snce_issue_date'] = ((DF3['today'] - DF3['issue_d'])/np.timedelta64(1, 'M')).astype(np.float64)

DF3['months_snce_last_credit_pull'] = ((DF3['today'] - DF3['last_credit_pull_d'])/np.timedelta64(1, 'M')).astype(np.float64)
# Ploting a boxplot to check the derived columns



pl.figure(figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')

DF3.boxplot(['months_snce_issue_date','months_snce_last_credit_pull'])
# converting loan status to 0 and 1 in a new column Default

DF3['Default'] = DF3.loan_status

DF3[['Default','loan_status']]



DF3['Default'] = DF3['Default'].replace('Fully Paid', 0)

DF3['Default'] = DF3['Default'].replace('Charged Off', 1)

DF3[['Default','loan_status']].head(20)

DF3.Default = DF3.Default.astype('int64') 
#Checking % of rows having NA in differt columns

round(100*(DF3.isnull().sum()/len(DF3.index)),2)
# Imputing mean value in place of null in Emp Len column



DF3['Emp_Len'].fillna(DF3['Emp_Len'].mean(), inplace=True)
# Creating a new column month from issue date



DF3['issue_mnth']=DF3['issue_d'].dt.month
# deriving the data of loan amount requested - loan amount reciver from investor [ Business Driven ]

# derived for performing various numeric analysis between the numeric data



DF3['funded_amnt_inv_exact'] = DF3.loan_amnt-DF3.funded_amnt_inv

DF3['exact_funded_amnt_inv'] = DF3.funded_amnt_inv_exact.apply(lambda x: 0 if x > 0 else 1)
# deriving the data of loan amount requested - loan amount approved from LC [ Business Driven ]

# derived for performing various numeric analysis between the numeric data



DF3['funded_amnt_exact'] = DF3.loan_amnt-DF3.funded_amnt

DF3['exact_approved_funded_amnt'] = DF3.funded_amnt_exact.apply(lambda x: 0 if x > 0 else 1)
# dropping unwanted columns 



DF3 = DF3.drop(['funded_amnt_inv_exact', 'funded_amnt_exact'], axis=1)
# Checking distribution of annual income via histogram



plt.hist(DF3['annual_inc'], range=[0, 400000], facecolor='gray', align='mid')

plt.show()
# Binning annual income to create a new column Annual_Inc_Bin



labels = ['<20k', '20k-40k', '40k-60k', '60k-80k','80k-100k','>100k']

bins = [0, 20000, 40000, 60000, 80000, 100000, 10000000]

DF3['Annual_Inc_Bin'] = pd.cut(DF3['annual_inc'], bins=bins, labels=labels)
# Binning annual income to create a new column Annual_Inc_Bin



DF3['annual_inc_bin'] = pd.qcut(DF3['annual_inc'], q=4)
# Checking distribution of employee service length via histogram



plt.hist(DF3['Emp_Len'])

plt.show()
# Binning employee service length to create a new column Emp_Len_Bin



labels = ['<2', '2-4', '4-6', '6-8','>8']

bins = [0, 2, 4, 6, 8, 10]

DF3['Emp_Len_Bin'] = pd.cut(DF3['Emp_Len'], bins=bins, labels=labels)
# creating employee experience bins



DF3['emp_len_bin'] = DF3.Emp_Len

DF3.emp_len_bin = DF3.emp_len_bin.replace([1,2,3], 'Fresher')

DF3.emp_len_bin = DF3.emp_len_bin.replace([4,5,6,7], 'Moderately experienced')

DF3.emp_len_bin = DF3.emp_len_bin.replace([8,9,10], 'Experienced')
# Checking distribution of interest rates via histogram



plt.hist(DF3['int_rate'])

plt.show()
# Binning interest rates to create a new column Int_Rate_Bin



labels = ['<7.5%', '7.5% - 10.0%', '10.0% - 12.5%', '12.5% - 15.0%','15.0% - 17.5%', '>17.5%']

bins = [0, 7.5, 10, 12.5, 15, 17.5, 30]

DF3['Int_Rate_Bin'] = pd.cut(DF3['int_rate'], bins=bins, labels=labels)
# Checking distribution of dti via histogram



plt.hist(DF3['dti'])

plt.show()
# Binning debt to income ratio to create a new column dti_bin



labels = ['0-5', '5-10', '10-15', '15-20','20-25', '25-30']

bins = [0, 5, 10, 15, 20, 25, 30]

DF3['dti_bin'] = pd.cut(DF3['dti'], bins=bins, labels=labels)
# Checking distribution of loan amount via histogram



plt.hist(DF3['loan_amnt'])

plt.show()
# Binning loan amount to create a new column Loan_amnt_bin



labels = ['<5k', '5k-10k', '10k-15k', '15k-20k','>20k']

bins = [0, 5000, 10000, 15000, 20000, 100000]

DF3['Loan_amnt_bin'] = pd.cut(DF3['loan_amnt'], bins=bins, labels=labels)
# Checking distribution of loan amount via histogram



plt.hist(DF3['installment'])

plt.show()
# Binning installment size to create a new column installment_bin



labels = ['<100', '100-200', '200-300', '300-400','>400']

bins = [0, 100, 200, 300, 400, 1500]

DF3['installment_bin'] = pd.cut(DF3['installment'], bins=bins, labels=labels)
# getting Numeric data



numeric_data = DF3._get_numeric_data().columns.to_list()

numeric_data
# getting categorical data



categorical_data = DF3.loc[:,~DF3.columns.isin(numeric_data)].columns.to_list()

categorical_data
# Defining a function to add value labels to plots



def add_value_labels(ax, spacing=1):

    # For each bar: Place a label

    for rect in ax.patches:

        

        # Get X and Y placement of label from rect.

        y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2



        # Create annotation

        ax.annotate("{:.2f}".format(y_value), (x_value, y_value), ha='center', va='bottom')
# Creating a pie plot for distribution of all loan accounts vis-a-vis their current loan status of 100 percent



A2 = DF3.groupby(['loan_status'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Accounts"}, inplace=True)

A2['Total_Accounts']=A2['Accounts'].sum()

A2['Accounts_percent_of_total']=round(100*A2['Accounts']/A2['Total_Accounts'],2)





gs = gridspec.GridSpec(2, 2)

pl.figure(figsize=(10,5))

ax = pl.subplot(gs[:, 0:2])

ax.set_title('Distribution of loan accounts by current loan status (%)')

ax.pie(A2['Accounts_percent_of_total'],labels=A2['loan_status'],autopct='%1.2f',startangle=90)



plt.savefig("image.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their grades



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['grade'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['grade'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)





Grade=pd.merge(A2,A3,how='inner', on='grade')

Grade['Default_percent']=round(100*Grade['Num_defaults']/Grade['Total_accounts'],2)

Grade=Grade.sort_values(by=['grade'])

Grade['Total_defaults']=Grade['Num_defaults'].sum()

Grade['Default_percent_of_total']=round(100*Grade['Num_defaults']/Grade['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their grades



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of defaulting loans for each grade (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters in different loan grades (%)')

ax.pie(Grade['Default_percent_of_total'],labels=Grade['grade'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for each grade 

ax = pl.subplot(gs[:, 1:2]) 

ax.set_title('Default Status vis-a-vis loan grade')

ax.set_xlabel('Loan Grade')

ax.set_ylabel('Default Rate (%)')

plt.bar(Grade['grade'],Grade['Default_percent'])

add_value_labels(ax)



plt.savefig("image2.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their sub-grades



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['sub_grade'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['sub_grade'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)





SGrade=pd.merge(A2,A3,how='inner', on='sub_grade')

SGrade['Default_percent']=round(100*SGrade['Num_defaults']/SGrade['Total_accounts'],2)

SGrade=SGrade.sort_values(by=['sub_grade'])

SGrade['Total_defaults']=SGrade['Num_defaults'].sum()

SGrade['Default_percent_of_total']=round(100*SGrade['Num_defaults']/SGrade['Total_defaults'],2)
# # Plotting defaulter distribution vis-a-vis the high risk subordinate grades/tranches

gs = gridspec.GridSpec(2, 2)



SubGrade=SGrade.loc[SGrade['sub_grade'].isin(['F1','F2','F3','F4','F5','G1','G2','G3','G4','G5'])]



# Bar plot

# This plot indicates the default rate for high risk subgrades

ax = pl.subplot(gs[:, 0:2])

ax.set_title('Default status for high risk sub grades')

ax.set_xlabel('Loan Subgrade')

ax.set_ylabel('Default Rate (%)')

plt.bar(SubGrade['sub_grade'],SubGrade['Default_percent'])

add_value_labels(ax)



plt.savefig("image3.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their home ownership



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['home_ownership'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['home_ownership'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)





Home=pd.merge(A2,A3,how='inner', on='home_ownership')

Home['Default_percent']=round(100*Home['Num_defaults']/Home['Total_accounts'],2)

Home=Home.sort_values(by=['home_ownership'])

Home['Total_defaults']=Home['Num_defaults'].sum()

Home['Default_percent_of_total']=round(100*Home['Num_defaults']/Home['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their home ownership



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of defaulting loans for home ownership type (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters vis-a-vis home ownership status (%)')

ax.pie(Home['Default_percent_of_total'],labels=Home['home_ownership'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for home ownership type

ax = pl.subplot(gs[:, 1:2])

ax.set_title('Default Status vis-a-vis home ownership status')

ax.set_xlabel('Home Ownership')

ax.set_ylabel('Default Rate (%)')

plt.bar(Home['home_ownership'],Home['Default_percent'])

add_value_labels(ax)



plt.savefig("image4.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their employee service length



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['Emp_Len_Bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['Emp_Len_Bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



Emp=pd.merge(A2,A3,how='inner', on='Emp_Len_Bin')

Emp['Default_percent']=round(100*Emp['Num_defaults']/Emp['Total_accounts'],2)

Emp=Emp.sort_values(by=['Emp_Len_Bin'])

Emp['Total_defaults']=Emp['Num_defaults'].sum()

Emp['Default_percent_of_total']=round(100*Emp['Num_defaults']/Emp['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their employee service length



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of defaulting loans vs employee service length (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters on length of employment (%)')

ax.pie(Emp['Default_percent_of_total'],labels=Emp['Emp_Len_Bin'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for employee service length bins

ax = pl.subplot(gs[:, 1:2]) 

ax.set_title('Default Status vis-a-vis debtor\'s length of employment')

ax.set_xlabel('Employement Length')

ax.set_ylabel('Default Rate (%)')

plt.bar(Emp['Emp_Len_Bin'],Emp['Default_percent'])

add_value_labels(ax)



plt.savefig("image5.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their annual income



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['Annual_Inc_Bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['Annual_Inc_Bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



Inc=pd.merge(A2,A3,how='inner', on='Annual_Inc_Bin')

Inc['Default_percent']=round(100*Inc['Num_defaults']/Inc['Total_accounts'],2)

Inc=Inc.sort_values(by=['Annual_Inc_Bin'])

Inc['Total_defaults']=Inc['Num_defaults'].sum()

Inc['Default_percent_of_total']=round(100*Inc['Num_defaults']/Inc['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their annual income



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of defaulting loans vs annual income (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by annual income (%)')

ax.pie(Inc['Default_percent_of_total'],labels=Inc['Annual_Inc_Bin'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for annual income bins

ax = pl.subplot(gs[:, 1:2]) 

ax.set_title('Default Status vis-a-vis debtor\'s annual income')

ax.set_xlabel('Annual Income')

ax.set_ylabel('Default Rate (%)')

plt.bar(Inc['Annual_Inc_Bin'],Inc['Default_percent'])

plt.xticks(rotation=45)

add_value_labels(ax)



plt.savefig("image6.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their interest rates charged



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['Int_Rate_Bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['Int_Rate_Bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



Int=pd.merge(A2,A3,how='inner', on='Int_Rate_Bin')

Int['Default_percent']=round(100*Int['Num_defaults']/Int['Total_accounts'],2)

Int=Int.sort_values(by=['Int_Rate_Bin'])

Int['Total_defaults']=Int['Num_defaults'].sum()

Int['Default_percent_of_total']=round(100*Int['Num_defaults']/Int['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their interest rates charged



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of defaulting loans vs interest rates charged (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by interest rate charged (%)')

ax.pie(Int['Default_percent_of_total'],labels=Int['Int_Rate_Bin'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for interest rates bins

ax = pl.subplot(gs[:, 1:2])

ax.set_title('Default Status vis-a-vis interest rates charged on loan')

ax.set_xlabel('Interest Rate')

ax.set_ylabel('Default Rate (%)')

plt.bar(Int['Int_Rate_Bin'],Int['Default_percent'])

plt.xticks(rotation=45)

add_value_labels(ax)



# Creating a dataframe to plot defaulter distribution vis-a-vis their debt to income ratio



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['dti_bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['dti_bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



dti=pd.merge(A2,A3,how='inner', on='dti_bin')

dti['Default_percent']=round(100*dti['Num_defaults']/dti['Total_accounts'],2)

dti=dti.sort_values(by=['dti_bin'])

dti['Total_defaults']=dti['Num_defaults'].sum()

dti['Default_percent_of_total']=round(100*dti['Num_defaults']/dti['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their debt to income ratio



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of defaulting loans vs debt to income ratio (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by debt to income ratio (%)')

ax.pie(dti['Default_percent_of_total'],labels=dti['dti_bin'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for debt to income ratio bins

ax = pl.subplot(gs[:, 1:2]) 

ax.set_title('Default Status vis-a-vis debtor\'s debt to income ratio')

ax.set_xlabel('Debt to Income Ratio')

ax.set_ylabel('Default Rate (%)')

plt.bar(dti['dti_bin'],dti['Default_percent'])

plt.xticks()

add_value_labels(ax)



plt.savefig("image8.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their loan amount



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['Loan_amnt_bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['Loan_amnt_bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



loan=pd.merge(A2,A3,how='inner', on='Loan_amnt_bin')

loan['Default_percent']=round(100*loan['Num_defaults']/loan['Total_accounts'],2)

loan=loan.sort_values(by=['Loan_amnt_bin'])

loan['Total_defaults']=loan['Num_defaults'].sum()

loan['Default_percent_of_total']=round(100*loan['Num_defaults']/loan['Total_defaults'],2)

loan['All_accounts']=loan['Total_accounts'].sum()

loan['Accounts_percent_of_total']=round(100*loan['Total_accounts']/loan['All_accounts'],2)
# Plotting defaulter distribution vis-a-vis their loan amount



gs = gridspec.GridSpec(3, 3)



# Subplot 1 (Pie Chart)

# This plot indicates the share of all loans vs loan amount (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of all loans by loan amount (%)')

ax.pie(loan['Accounts_percent_of_total'],labels=loan['Loan_amnt_bin'],autopct='%1.2f',startangle=90)



# Subplot 2 (Pie Chart)

# This plot indicates the share of defaulting loans vs loan amount (total out of 100%)

ax = pl.subplot(gs[:, 1:2])

ax.set_title('Distribution of defaulters by loan amount (%)')

ax.pie(loan['Default_percent_of_total'],labels=loan['Loan_amnt_bin'],autopct='%1.2f',startangle=90)



# Subplot 3 (Bar plot)

# This plot indicates the default rate for debt to loan amount bins

ax = pl.subplot(gs[:, 2:3]) # row 0, col 0

ax.set_title('Default Status vis-a-vis debtor\'s loan amount')

ax.set_xlabel('Loan Amount')

ax.set_ylabel('Default Rate (%)')

plt.bar(loan['Loan_amnt_bin'],loan['Default_percent'])

plt.xticks(rotation=45)

add_value_labels(ax)



plt.savefig("image9.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their term



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['term'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['term'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



term=pd.merge(A2,A3,how='inner', on='term')

term['Default_percent']=round(100*term['Num_defaults']/term['Total_accounts'],2)

term=term.sort_values(by=['term'])

term['Total_defaults']=term['Num_defaults'].sum()

term['Default_percent_of_total']=round(100*term['Num_defaults']/term['Total_defaults'],2)
term
# Plotting defaulter distribution vis-a-vis their term



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of all loans vs term (total out of 100%)

Term=['36 months','60 months']

Default_percent_of_total=[57.35,42.65]



pl.figure(figsize=(10,5))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by loan term (%)')

ax.pie(Default_percent_of_total,labels=Term,autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for debt to term bins

Default_percent= [11.09,25.31]



ax = pl.subplot(gs[:, 1:2]) 

ax.set_title('Default Status vis-a-vis the loan term')

ax.set_xlabel('Loan Term')

ax.set_ylabel('Default Rate (%)')

plt.bar(Term,Default_percent)

plt.xticks()

add_value_labels(ax)



plt.savefig("image10.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their installment amount



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['installment_bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['installment_bin'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



ins=pd.merge(A2,A3,how='inner', on='installment_bin')

ins['Default_percent']=round(100*ins['Num_defaults']/ins['Total_accounts'],2)

ins=ins.sort_values(by=['installment_bin'])

ins['Total_defaults']=ins['Num_defaults'].sum()

ins['Default_percent_of_total']=round(100*ins['Num_defaults']/ins['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their installment amount



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of all loans vs installment amount (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by installment amount (%)')

ax.pie(ins['Default_percent_of_total'],labels=ins['installment_bin'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for debt to installment amount bins

ax = pl.subplot(gs[:, 1:2])

ax.set_title('Default Status vis-a-vis debtor\'s installment amount')

ax.set_xlabel('Installment Amount')

ax.set_ylabel('Default Rate (%)')

plt.bar(ins['installment_bin'],ins['Default_percent'])

plt.xticks()

add_value_labels(ax)



plt.savefig("image11.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their month of loan issue



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['issue_mnth'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['issue_mnth'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



iss=pd.merge(A2,A3,how='inner', on='issue_mnth')

iss['Default_percent']=round(100*iss['Num_defaults']/iss['Total_accounts'],2)

iss=iss.sort_values(by=['issue_mnth'])

iss['Total_defaults']=iss['Num_defaults'].sum()

iss['Default_percent_of_total']=round(100*iss['Num_defaults']/iss['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their month of loan issue



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of all loans vs month of loan issue (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by the month of loan issuance')

ax.pie(iss['Default_percent_of_total'],labels=iss['issue_mnth'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for debt to month of loan issue 

Default_percent= [13.49,12.3,12.89,13.07,15.96,15.19,14.29,13.81,15.64,15.42,14.93,16.09]

Mon=[1,2,3,4,5,6,7,8,9,10,11,12]



ax = pl.subplot(gs[:, 1:2])

ax.set_title('Default Status vis-a-vis the month of loan issuance')

ax.set_xlabel('Month of loan issue')

ax.set_ylabel('Default Rate (%)')

plt.bar(Mon,Default_percent)

plt.xticks()

add_value_labels(ax)



plt.savefig("image12.png")
# Creating a dataframe to plot defaulter distribution vis-a-vis their purpose of loan



A1= DF3.loc[DF3['loan_status'].isin(['Charged Off'])]

A2 = A1.groupby(['purpose'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A2.rename(columns={'id': "Num_defaults"}, inplace=True)

A3 = DF3.groupby(['purpose'])['id'].nunique().reset_index().sort_values(by='id',ascending=False)

A3.rename(columns={'id': "Total_accounts"}, inplace=True)



pps=pd.merge(A2,A3,how='inner', on='purpose')

pps['Default_percent']=round(100*pps['Num_defaults']/pps['Total_accounts'],2)

pps=pps.sort_values(by=['purpose'])

pps['Total_defaults']=pps['Num_defaults'].sum()

pps['Default_percent_of_total']=round(100*pps['Num_defaults']/pps['Total_defaults'],2)
# Plotting defaulter distribution vis-a-vis their purpose of loan



gs = gridspec.GridSpec(2, 2)



# Subplot 1 (Pie Chart)

# This plot indicates the share of all loans vs purpose of loan (total out of 100%)

pl.figure(figsize=(15,10))

ax = pl.subplot(gs[:, 0:1])

ax.set_title('Distribution of defaulters by debtor\'s purpose of loan')

ax.pie(pps['Default_percent_of_total'],labels=pps['purpose'],autopct='%1.2f',startangle=90)



# Subplot 2 (Bar plot)

# This plot indicates the default rate for debt to purpose of loan

ax = pl.subplot(gs[:, 1:2])

ax.set_title('Default Status vis-a-vis debtor\'s purpose of loan')

ax.set_xlabel('Loan Amount')

ax.set_ylabel('Default Rate (%)')

plt.bar(pps['purpose'],pps['Default_percent'])

plt.xticks(rotation=90)



add_value_labels(ax)



plt.savefig("image13.png")
# correlation matrix between all the numeric data



plt.subplots(figsize=(20,15))

sns.heatmap(DF3[numeric_data[1:]].corr(), annot=True, center=0,  vmin=-1, vmax=1, linewidths=5)
# influence of variables dti and term



df = DF3[['Default', 'term', 'dti']]

sns.barplot(x="term", y="dti", hue='Default', data=df, )
# influence of variables dti and home_ownership



sns.barplot('home_ownership', 'dti',hue='Default', data=DF3[['Default', 'home_ownership', 'dti']])
# influence of variables verified and annual_inc



sns.barplot(x='verification_status', y='annual_inc',hue='Default', data=DF3[['Default', 'verification_status', 'annual_inc']])
# influence of variables dti and purpose



plt.subplots(figsize=(20,10))

sns.barplot(x='purpose', y='annual_inc',hue='Default', data=DF3[['purpose', 'Default', 'annual_inc']])
# influence of variables dti and addr_state set 1



addr = DF3[['addr_state', 'Default', 'dti']]

plt.subplots(figsize=(25,5))

sns.barplot(x='addr_state', y='dti', hue='Default', data=addr)
# influence of funded_amnt_inv and annual_inc_bin



plt.subplots(figsize=(10,5))

sns.boxplot(y='funded_amnt_inv', x='annual_inc_bin',hue='Default', data=DF3[['funded_amnt_inv', 'Default', 'annual_inc_bin']])
# influence of funded_amnt_inv and grade



sns.boxplot(y='funded_amnt_inv', x='grade',hue='Default', data=DF3[['funded_amnt_inv', 'Default', 'grade']])
# influence of annual_inc_bin and grade



sns.barplot(y='annual_inc', x='grade',hue='Default', data=DF3[['annual_inc', 'Default', 'grade']])
# influence of funded_amnt_inv and sub_grade



data = DF3[['funded_amnt_inv', 'Default', 'sub_grade']]

data = data.sort_values('sub_grade', ascending=True)

y=data[data['sub_grade'].isin(['A1', 'A2', 'A3', 'A4','A5'])]

plt.subplots(figsize=(20,15))

sns.barplot(y='funded_amnt_inv', x='sub_grade', hue='Default', data=data)
# distrubution of the grade across the data



DF3 = DF3.sort_values('grade', ascending=True)

DF3.grade.hist(bins=7)