import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline
df = pd.read_csv('../input/prosperLoanData.csv')
# Shape of entire dataset

df.shape


df.dtypes
# Summary statistics

df.describe()
# Duplicates data entry in loan data

df.duplicated().sum()
df.isnull().sum()
df_loan = df.copy()
df_loan.info()
# Changing Loan orgination date into date time format

df_loan['LoanOriginationDate'] = pd.to_datetime(df_loan['LoanOriginationDate'])
df_loan.dtypes
# filter out loans without ProsperScores

df_loan = df_loan[df_loan['ProsperScore'].isnull()==False]
# Loan by term

base_color = sb.color_palette()[0]

sb.countplot(data=df_loan,x= 'Term',color=base_color);

plt.title('Terms of loan (Months)')

plt.xlabel('Term (Months)');
type_count = df_loan['LoanStatus'].value_counts()

type_order = type_count.index
# Count of Loan by Loan Status

n_loan =df_loan.shape[0]

max_type_count = type_count[0]

max_prop = max_type_count/n_loan
tick_props = np.arange(0,max_prop,0.1)

tick_names = ['{:0.2f}'.format(v) for v in tick_props]
tick_names
sb.countplot(data=df_loan,y='LoanStatus',color=base_color,order=type_order);

plt.xticks(tick_props*n_loan,tick_names)

plt.xlabel('proportion');

plt.title('Proportion of Loan Status')
df['ProsperScore'].describe()
# Distribution of Prosper rating

sb.countplot(data=df_loan,x='ProsperRating (numeric)',color=base_color);

plt.title('Count of Prosper ratings')
df_loan['Year'] = df_loan['LoanOriginationQuarter'].str[-4:]
# Number of loans per year

sb.countplot(data=df_loan,x='Year',color=base_color);

plt.title('Numner of Loans Sanctioned per year')

plt.xticks(rotation=90);
df['LoanOriginalAmount'].describe()
# Distribution of orginal Loan amount

bins = np.arange(1000,35000,2000)

plt.hist(data=df_loan,x='LoanOriginalAmount',color=base_color,bins=bins);

plt.title('Distribution of orginal Loan amount')

plt.xticks(rotation=90);
# Histogram for Credit Score ranges



plt.figure(figsize = [13, 5]) 





plt.subplot(1, 2, 1)

bins = np.arange(550, df_loan['CreditScoreRangeLower'].max(), 20)

plt.hist(data = df_loan, x = 'CreditScoreRangeLower', bins = bins)

plt.xticks(np.arange(550, 1000, 100))

plt.title('CreditScoreRangeLower Count')

plt.xlabel('CreditScoreRangeLower')

plt.ylabel('count');



plt.subplot(1, 2, 2)

bins = np.arange(550, df_loan['CreditScoreRangeUpper'].max(), 20)

plt.hist(data = df_loan, x = 'CreditScoreRangeUpper', bins = bins)

plt.xticks(np.arange(550, 1000, 100))

plt.title('CreditScoreRangeUpper Count')

plt.xlabel('CreditScoreRangeUpper')

plt.ylabel('count');
df_loan['LenderYield'].describe()
# Distribution of lender yield

bins = np.arange(.03,.34,.01)

plt.hist(data=df_loan,x='LenderYield',color=base_color,bins=bins);

plt.title('Distribution of Lender yield')

plt.xticks(rotation=90);
# Income range of borrower

order = ['$0','$1-24,999','$25,000-49,999','$50,000-74,999','$75,000-99,999','$100,000+']

sb.countplot(data=df_loan,x='IncomeRange',color=base_color,order=order);

plt.title('Count of Income Range')

plt.xticks(rotation=90);
# correlation plot 



num_vars = ['BorrowerAPR', 'ProsperScore', 'LenderYield', 

            'StatedMonthlyIncome',  'CreditScoreRangeUpper','ProsperRating (numeric)','DebtToIncomeRatio']

plt.figure(figsize = [8, 5])

sb.heatmap(df_loan[num_vars].corr(), annot = True, fmt = '.3f',

           cmap = 'vlag_r', center = 0)

plt.title('Correlation Plot') 

plt.show()
# plot matrix: only 300 random loans are used to see the pattern more clearer





num_vars = ['BorrowerAPR', 'ProsperScore', 'LenderYield', 

            'StatedMonthlyIncome',  'CreditScoreRangeUpper','ProsperRating (numeric)','DebtToIncomeRatio']



samples = np.random.choice(df_loan.shape[0], 300, replace = False)

loan_samp = df_loan.loc[samples,:]



g = sb.PairGrid(data = loan_samp, vars = num_vars)

g.map_offdiag(plt.scatter)

plt.title('Matrix Plot');
# scatter and heat plot for comparing ProsperScore and BorrowerAPR. 

plt.figure(figsize = [15, 5]) 



plt.subplot(1, 2, 1)

plt.scatter(data = df_loan, x = 'BorrowerAPR', y = 'ProsperScore', alpha =  0.005)

plt.yticks(np.arange(0, 12, 1))

plt.title('BorrowerAPR vs. ProsperScore')

plt.xlabel('BorrowerAPR')

plt.ylabel('ProsperScore')





plt.subplot(1, 2, 2)

bins_x = np.arange(0, df_loan['BorrowerAPR'].max()+0.05, 0.03)

bins_y = np.arange(0, df_loan['ProsperScore'].max()+1, 1)

plt.hist2d(data = df_loan, x = 'BorrowerAPR', y = 'ProsperScore', bins = [bins_x, bins_y], 

               cmap = 'viridis_r', cmin = 0.5)

plt.colorbar()

plt.title('BorrowerAPR vs. ProsperScore')

plt.xlabel('BorrowerAPR (l)')

plt.ylabel('ProsperScore');
# scatter and heat plot for comparing BorrowerAPR and credit score upper range. 

plt.figure(figsize = [15, 5]) 



plt.subplot(1, 2, 1)

plt.scatter(data = df_loan, x = 'CreditScoreRangeUpper', y = 'BorrowerAPR', alpha = 0.01)

plt.title('BorrowerAPR vs. CreditScoreRangeUpper')

plt.xlabel('BorrowerAPR')

plt.ylabel('CreditScoreRangeUpper');





plt.subplot(1, 2, 2)

bins_x = np.arange(0, df_loan['BorrowerAPR'].max()+0.05, 0.02)

bins_y = np.arange(500, df_loan['CreditScoreRangeUpper'].max()+100, 20)

plt.hist2d(data = df_loan, x = 'BorrowerAPR', y = 'CreditScoreRangeUpper', bins = [bins_x, bins_y], 

               cmap = 'viridis_r', cmin = 0.5)

plt.colorbar()

plt.title('BorrowerAPR vs. CreditScoreRangeUpper')

plt.xlabel('BorrowerAPR (l)')

plt.ylabel('CreditScoreRangeUpper');
# Stated MonthlyIncome vs Prosper Rating

plt.figure(figsize = [15, 5])



plt.subplot(1, 2, 1)

sb.boxplot(data=df_loan,x='ProsperScore',y='StatedMonthlyIncome',color=base_color);

plt.xlabel('Prosper Score');

plt.ylabel('Monthly Income');

plt.title('Box plot of monthly income Vs prosper Score');



plt.subplot(1, 2, 2)

plt.scatter(data=df_loan,x='BorrowerAPR',y='StatedMonthlyIncome',color=base_color);

plt.xlabel('BorrowerAPR');

plt.ylabel('Monthly Income');

plt.title('Scatter plot of monthly income Vs Borrower APR');
# Borrower APR vs Status of Loan and  Borrower APR vs Employment status

plt.figure(figsize = [15, 5])



plt.subplot(1, 2, 1)

sb.boxplot(data=df_loan,x='BorrowerAPR',y='LoanStatus',color=base_color);

plt.xlabel('Borrower APR');

plt.ylabel('Loan Status');

plt.title('Box plot of Borrower APR vs Status of Loan');



plt.subplot(1, 2, 2)

sb.boxplot(data=df_loan,x='BorrowerAPR',y='EmploymentStatus',color=base_color);

plt.xlabel('Borrower APR');

plt.ylabel('Employment Status');

plt.title('Box plot of Borrower APR vs Employment Status');
df_series = df_loan['BorrowerRate'].groupby(df_loan['LoanOriginationQuarter']).mean().reset_index()

df_series.LoanOriginationQuarter = pd.Categorical(df_series.LoanOriginationQuarter, sorted(df_series.LoanOriginationQuarter, key=lambda x: x.split(' ')[-1]), ordered = True)

df_series.sort_values('LoanOriginationQuarter', inplace=True)
# Mean Borrower rate over time

plt.errorbar(data=df_series,x='LoanOriginationQuarter',y='BorrowerRate');

plt.xticks(rotation = 90);

plt.xlabel('Quaerters');

plt.ylabel('Mean Borrower rate');

plt.title('Quarter and rate trends');
# LenderYield vs Borrower APR  vs ProsperRating

plt.figure(figsize = [10, 5])

plt.scatter(data=df_loan,x='LenderYield',y = 'BorrowerAPR',c='ProsperScore',cmap = 'viridis_r')

plt.colorbar(label = 'ProsperScore');

plt.xlabel('Lender Yield')

plt.ylabel('Borrower APR')

plt.title('LenderYield vs Borrower APR  vs ProsperRating');
# BorrowerAPR vs. CreditScoreRangeUpper & CreditScoreRangeUpper

plt.figure(figsize = [15, 5]) 

plt.scatter(data = df_loan, x = 'CreditScoreRangeUpper', y = 'BorrowerAPR', c ='ProsperScore', alpha = 0.3)

plt.colorbar(label = 'ProsperScore')

plt.title('BorrowerAPR vs. CreditScoreRangeUpper & CreditScoreRangeUpper')

plt.xlabel('BorrowerAPR')

plt.ylabel('CreditScoreRangeUpper');
# LoanStatus Vs BorrowerAPR VS EmploymentStatus

plt.figure(figsize=[12,10])

sb.boxplot(x="LoanStatus", y="BorrowerAPR", hue="EmploymentStatus", data=df_loan, palette="RdYlBu");

plt.xticks(rotation = 90);

plt.xlabel('Loan Status');

plt.ylabel('BorrowerAPR');

plt.title('LoanStatus Vs BorrowerAPR VS EmploymentStatus');