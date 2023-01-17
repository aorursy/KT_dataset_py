import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#read the loan data
#parse dates for easy time slicing
ld = pd.read_csv('../input/loan.csv',low_memory=False, parse_dates = True)

#determine the percet full for each variable
pct_full = ld.count()/len(ld)
names = list(pct_full[pct_full > 0.75].index)

#reduce to mostly full data
loan = ld[names]
import seaborn as sns
import matplotlib

#I swear it makes graphs more meaningful
plt.style.use('fivethirtyeight')

#lets look at the distirbution of the loan amount
amount_hist = loan.funded_amnt_inv.hist()
amount_hist.set_title('Histogram of Loan Amount')
#the average loan is a little less than $15,000.00
loan.funded_amnt_inv.describe()
#np.median(loan.funded_amnt)
#look at difference between the length of the loans 36 vs. 60 month loans
termGroup = loan.groupby('term')
termGroup['funded_amnt_inv'].agg([np.count_nonzero, np.mean, np.std])
#summarize loans by month

#hide the ugly warning
#!usually should set on copy of original data when creating variables!
pd.options.mode.chained_assignment = None 

#make new variable to groupby for month and year
loan['issue_mo'] = loan.issue_d.str[0:3]
loan['issue_year'] = loan.issue_d.str[4:]

loan_by_month = loan.groupby(['issue_year','issue_mo'])

avgLoanSizeByMonth = loan_by_month['funded_amnt_inv'].agg(np.mean).plot()
avgLoanSizeByMonth.set_title('Avg. Loan Size By Month')
NumLoansPerMo = loan_by_month.id.agg(np.count_nonzero).plot()
NumLoansPerMo.set_title('Number of Loans By Month')
NumLoansPerMo.set_xlabel('Issue Month')
#less granular look at loan volume
loanByYr = loan.groupby('issue_year')
loanYrPlt = loanByYr.id.agg(np.count_nonzero).plot(kind = 'bar')
loanYrPlt.set_title('Num Loans By Year')
loanYrPlt.set_xlabel('Issue Year')
import calendar
#get the counts by month
loanByMo = loan.groupby(['issue_d', 'issue_mo'])
numByDate = loanByMo.agg(np.count_nonzero).reset_index()

#average the monthly counts across years
counts_by_month = numByDate.groupby('issue_mo')
avg_loan_vol = counts_by_month.id.agg(np.mean)


moOrder = calendar.month_abbr[1:13]
mo_plt = sns.barplot(x = list(avg_loan_vol.index),y = avg_loan_vol, order = moOrder, palette = "GnBu_d")
mo_plt.set_title('Avg. Loan Volume Per Month')
#get the counts by mo/year
loanByMo = loan.groupby(['issue_d','issue_year','issue_mo'])
numByDate = loanByMo.agg(np.count_nonzero).reset_index()

#get the individual years
years = np.unique(loan.issue_year)

#just looking at the first year
tmp_agg = numByDate[numByDate.issue_year == '2007']
tmp_plt = sns.barplot(x = tmp_agg.issue_mo,y = tmp_agg.id, order = moOrder, palette = "GnBu_d")
tmp_plt.set_title('Loans By Month: 2007')
#plot the years in stacked graphs
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, figsize=(5, 8), sharex=True)

#being lazy and not wanting to figure out a better way to do this
#Please let me know if any of you know a better way
y1 = numByDate[numByDate.issue_year == '2007']
y2 = numByDate[numByDate.issue_year == '2008']
y3 = numByDate[numByDate.issue_year == '2009']
y4 = numByDate[numByDate.issue_year == '2010']
y5 = numByDate[numByDate.issue_year == '2011']
y6 = numByDate[numByDate.issue_year == '2012']
y7 = numByDate[numByDate.issue_year == '2013']
y8 = numByDate[numByDate.issue_year == '2014']
y9 = numByDate[numByDate.issue_year == '2015']

sns.barplot(y1.issue_mo, y1.id, order = moOrder, palette="BuGn_d", ax=ax1)
ax1.set_ylabel("2007")

sns.barplot(x = y2.issue_mo,y = y2.id, order = moOrder, palette="BuGn_d", ax=ax2)
ax2.set_ylabel("2008")

sns.barplot(x = y3.issue_mo,y = y3.id, order = moOrder, palette="BuGn_d", ax=ax3)
ax3.set_ylabel("2009")

sns.barplot(x = y4.issue_mo,y = y4.id, order = moOrder, palette="BuGn_d", ax=ax4)
ax4.set_ylabel("2010")

sns.barplot(x = y5.issue_mo,y = y5.id, order = moOrder, palette="BuGn_d", ax=ax5)
ax5.set_ylabel("2011")

sns.barplot(x = y6.issue_mo,y = y6.id, order = moOrder, palette="BuGn_d", ax=ax6)
ax6.set_ylabel("2012")

sns.barplot(x = y7.issue_mo,y = y7.id, order = moOrder, palette="BuGn_d", ax=ax7)
ax7.set_ylabel("2013")

sns.barplot(x = y8.issue_mo,y = y8.id, order = moOrder, palette="BuGn_d", ax=ax8)
ax8.set_ylabel("2014")

sns.barplot(x = y9.issue_mo, y = y9.id, order = moOrder, palette="BuGn_d", ax=ax9)
ax9.set_ylabel("2015")

#look better
sns.despine(bottom=True)
plt.setp(f.axes, yticks = [], xlabel = '')
plt.tight_layout()
loan['pct_paid'] = loan.out_prncp / loan.loan_amnt

loan[loan.loan_status == 'Current'].pct_paid.hist(bins = 50)