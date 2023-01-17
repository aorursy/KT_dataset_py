import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max.rows', 50)
# The Paycheck Protection Program is a $669-billion business loan program established by the 2020 US Federal government 
# Coronavirus Aid, Relief, and Economic Security Act to help certain businesses, self-employed workers, sole proprietors,
# certain nonprofit organizations, and tribal businesses continue paying their workers
# source: wikipedia.com

# The files

# File acquisition date - July 8, 2020
file_150kplus = "../input/ppp-data-150k-plus/PPP_Data_150k_plus.csv"

# File acquisition date - July 7, 2020
file_150kless = "../input/ppp-loan-data/foia_less_than_150k.csv"
df = pd.read_csv(file_150kplus)  # Loans above $150,000
df2 = pd.read_csv(file_150kless) # Loans below $150,000
# Adding BusinessName and Address column to df2
df2['BusinessName'] = ['-' for i in range(df2.count().max())]
df2['Address'] = ['-' for i in range(df2.count().max())]

# Renaming LoanAmount to LoanRange
df2.rename(columns={'LoanAmount': 'LoanRange'}, inplace=True)
# Dropping unnecessary column in df2
df2.drop('Unnamed: 0', axis=1,inplace=True)

# Reordering tables to suit df
df2 = df2[[i for i in df.columns]]
total = max(df.count())
print(f"The total number of items that should be in each df column is {total}.")

# Getting an idea of missing values
# print()
# print(df.count())
total2 = max(df2.count())
print(f"The total number of items that should be in each df2 column is {total2}.")

# Getting an idea of missing values
# print()
# print(df2.count())
# df.dtypes
# Missing business names are renamed 'ommited'.
df['BusinessName'].replace(np.NaN, 'Omitted', inplace=True)
df['BusinessName'].replace('-', 'Omitted', inplace=True)

df2['BusinessName'].replace(np.NaN, 'Omitted', inplace=True)
df2['BusinessName'].replace('-', 'Omitted', inplace=True)

# Missing JobRetained data is filled as 0.
df['JobsRetained'].replace(np.NaN, 0, inplace=True)
df2['JobsRetained'].replace(np.NaN, 0, inplace=True)

# Replacing 0 with them mean value in JobRetained column.
df['JobsRetained'].replace(0, round(df['JobsRetained'].mean()), inplace=True)
df2_mean = df2['JobsRetained'].mean()
print(df2_mean)
df2['JobsRetained'].replace(0, round(df2_mean), inplace=True)

df_new = pd.concat([df,df2])
df_new.count()
# Getting an understanding of missing data in RaceEthnicity column.

print(f"There are {df_new['RaceEthnicity'][df_new['RaceEthnicity'] == 'Unanswered'].count()} missing values in the RaceEthnicity column.")
# Unit testing if the columns below have no missing values
print(f'BussinessName: {df_new["BusinessName"].count()} items')
print(f'JobsRetained: {df_new["JobsRetained"].count()} items')

df["BusinessName"].count() == df["JobsRetained"].count() == total
notable = pd.concat([df[df['BusinessName'].str.contains('YEEZY')], 
                     df[df['BusinessName'].str.contains('CARDONE TRAINING')], 
                     df[df['BusinessName'].str.contains('CARDONE REAL ESTATE')]])
notable
df_new.dtypes
# New DataFrame
df_i = df_new[['LoanRange', 'JobsRetained']]
df_i['LoanRange'] = df_i['LoanRange'].astype(str)

# df_.sort_values(by='LoanRange', ascending=True, inplace=True)

df_ii = df_new[['LoanRange', 'JobsRetained']] 

# Renaming df_i's column
df_i.columns = ['LoanRange', 'Jobs Retained - AVG']
# Grouping data by Loan Category for data extraction
df_group = df_i.groupby(['LoanRange'], as_index=False)

# Extracting mean job(s) retained for each loan category.
mean_jobs_retained = df_group.mean()
mean_jobs_retained.sort_values(by='LoanRange', ascending=False, inplace=True) # sorted to ensure shell's LoanRange column matches
# print(mean_jobs_retained)

# Extracting sum of job(s) retained by each loan category.
sum_jobs_retained = df_group.sum()
sum_jobs_retained.columns = ['LoanRange', 'Total Jobs Retained']
sum_jobs_retained.sort_values(by='LoanRange', ascending=False, inplace=True) # sorted to ensure shell's LoanRange column matches
# print(sum_jobs_retained)

no_companies_eachcategory = df_i['LoanRange'].value_counts(ascending=True).to_frame()
no_companies_eachcategory.columns = ['No of Applicants | Recipients']
no_companies_eachcategory.index.name = 'index'
no_companies_eachcategory.sort_values(by='index', ascending=False, inplace=True) # sorted to ensure shell's LoanRange column matches
# print(no_companies_eachcategory)

# Checking if the above DataFrames are equally ordered
print(mean_jobs_retained.LoanRange.to_list() 
      == sum_jobs_retained.LoanRange.to_list() 
      == no_companies_eachcategory.index.to_list())

# Completing informative table as shell.
shell = pd.DataFrame({'LoanRange': mean_jobs_retained.LoanRange.to_list()})
shell['No of Applicants | Recipients'] = no_companies_eachcategory['No of Applicants | Recipients'].to_list()
shell['Jobs Retained - AVG'] = mean_jobs_retained['Jobs Retained - AVG'].to_list()
shell['Total Jobs Retained'] = sum_jobs_retained['Total Jobs Retained'].to_list()
above_150k = shell[:5][::-1].reset_index(drop=True)
above_150k
# Removing loan amount of '-199659'. It's possibly loan repayment or error in data entry.
shell.drop(424229, axis=0, inplace=True)

below_150k = shell[5:][::-1].reset_index(drop=True)
below_150k
summary_below_150k = pd.DataFrame() # summary of loans below $150k
summary_below_150k['LoanRange'] = ['f $1 - 150,000']
summary_below_150k['No of Applicants | Recipients'] = below_150k['No of Applicants | Recipients'].sum()

summary_below_150k['Jobs Retained - AVG'] = below_150k['Total Jobs Retained'].sum() / (df2['JobsRetained'].count() - 1)
# 1 was subtracted from the denominator due to entry drop in the cell above

summary_below_150k['Total Jobs Retained'] = below_150k['Total Jobs Retained'].sum()
# print(df2['JobsRetained'].sum())

summary_all_loans = pd.concat([above_150k, summary_below_150k])
summary_all_loans.reset_index(drop=True, inplace=True)
summary_all_loans
(summary_all_loans['No of Applicants | Recipients'].iloc[:3].sum()/summary_all_loans['No of Applicants | Recipients'].sum()) * 100
# QUICK STATS

# COVID-19 categorised as a pandemic = March 11, 2020 # source: https://time.com/5791661/who-coronavirus-pandemic-declaration/
# PPP Application Start Date = April 3, 2020          # source: https://home.treasury.gov/system/files/136/PPP--Fact-Sheet.pdf
# PPP Application Deadline (original) = June 30, 2020 # source: https://www.investopedia.com/your-guide-to-the-paycheck-protection-program-ppp-and-how-to-apply-4802195
# PPP Deadline Extension signed = July 4, 2020        # source: https://www.investopedia.com/your-guide-to-the-paycheck-protection-program-ppp-and-how-to-apply-4802195
# Date of data acquisition = July 8, 2020             # source: file directory info

us_population_jan_2020 = 329227746                    # source: https://en.wikipedia.org/wiki/Demographics_of_the_United_States
us_population_july10_2020 = 329926866                 ## preferred - source: https://www.census.gov/popclock/

us_labor_force_feb_2020 = 164546000                   # source: http://www.dlt.ri.gov/lmi/laus/us/usadj.htm
us_unemployment_rate_feb_2020 = 3.5/100               # source: http://www.dlt.ri.gov/lmi/laus/us/usadj.htm

us_labor_force_july_2020 = 159932000                  # source: http://www.dlt.ri.gov/lmi/laus/us/usadj.htm
us_unemployment_rate_july_2020 = 11/100               # source: http://www.dlt.ri.gov/lmi/laus/us/usadj.htm

us_unemployed_july_2020 = int(us_unemployment_rate_july_2020 * us_labor_force_july_2020)

# US STATS
print(f"US population           - July, 2020: {us_population_july10_2020:,}")
print(f"US labor force          - July, 2020: {us_labor_force_july_2020:,}")
print(f"Percentage unemployed   - July, 2020: {us_unemployment_rate_july_2020 * 100}%")
print(f"No of people unemployed - July, 2020: {us_unemployed_july_2020:,}")

# GUAGING PRE-PPP PLAUSIBLE LAYOFFS.
print()
print(f"Effects of loans received from Paycheck Protection Program:")
total = int(summary_all_loans['Total Jobs Retained'].sum())
print(f"1a. Total number of jobs retained - As at July 10, 2020: {total:,} jobs")

if_all_layedoff = ((us_unemployed_july_2020 + total)  / us_labor_force_july_2020) * 100

print(f"1b. If the Paycheck Protection Program wasn't created and all applicants layedoff all their employees, {if_all_layedoff:.2f}% "
      f"of the labor force would be unemployed. i.e {us_unemployed_july_2020 + total:,} people.")

# STATS OF RECIPIENTS OF $1M AND BELOW.
one_mill_below_rec = (summary_all_loans['No of Applicants | Recipients'].iloc[3:6].sum()  / summary_all_loans['No of Applicants | Recipients'].sum()) * 100

print()
print(f"2a. Small employers: {(one_mill_below_rec):.2f}% of applicants.")

retained_one_mill_below_recipients = int(summary_all_loans['Total Jobs Retained'].iloc[3:6].sum())

print(f"2b. Jobs retained by small employers: {retained_one_mill_below_recipients:,} jobs")

percentjobs_retained_one_mill_below = (summary_all_loans['Total Jobs Retained'].iloc[3:6].sum() / summary_all_loans['Total Jobs Retained'].sum()) * 100
print(f"2c. %Jobs retained by small employers: {percentjobs_retained_one_mill_below}%")


# STATS OF RECIPIENTS OF $1M AND ABOVE.
print()
print(f"3a. Large employers: {(100 - one_mill_below_rec):.2f}% of applicants.")
print(f"3b. Jobs retained by large employers: {int(total - retained_one_mill_below_recipients):,} jobs.")
print(f"3c. %Jobs retained by large employers: {100 - percentjobs_retained_one_mill_below}%")

print('Note that large employers refers to the large spenders who recieved loans beyond $1m and employ between 117 and 344 people on \n'
     'average while the small employers refers to the small spenders who received loans amounting to $1m & below and employ between 5 and 56 people on average.')

# JOBS RETAINED BY EACH CATEGORY OF LOAN RECIPIENT
print()
print(f"4. Jobs retained by each category of loan recipients:")

groups = summary_all_loans['LoanRange'].to_list()
retained = summary_all_loans['Total Jobs Retained'].to_list()
groups_dict = {}

for num, i in enumerate(groups):
    groups_dict[f"Category {i[:1].upper()} ({i[2:]})"] = f"{int(retained[num])}"
    
for i in groups_dict.items():
    print(f"{i[0]}: {int(i[1]):,} jobs i.e {(int(i[1]) / total) * 100:.2f}%")


# Jobs retained by Non Profit institutions
print()
non_profit_retained = df[df['NonProfit'] == 'Y']['JobsRetained'].sum()
print(f"NonProfits retained {int(non_profit_retained):,} jobs")

# Non Profit's percentage of total
non_profit_percentage = (non_profit_retained / total) * 100
print(f"Non Profit's retained {non_profit_percentage:.2f}% of jobs.")

# Number of recipients
print()
above150k_recipients = len(df['BusinessName'].unique())
print(f"More than {above150k_recipients:,} businesses have received loans above $150,000.")

applications_150kbelow = df2['BusinessName'].count()
applications_150kabove = df['BusinessName'].count()
print(f"There've been upto {applications_150kbelow:,} applications for Paycheck Protection Program loans below $150,000.")
print(f"Overall, {applications_150kbelow + applications_150kabove:,} applications were made for the PPP loans.")

# EMPLOYERS 
print()
large_loans_rec = summary_all_loans.loc[0:3, 'No of Applicants | Recipients'].sum()
small_loans_rec = summary_all_loans.loc[3:6, 'No of Applicants | Recipients'].sum()

print(f"Total number of large employers: {large_loans_rec:,}")
print(f"Total number of small employers: {small_loans_rec:,}")
print(f"Small employer:Large employers - There are {int(round(small_loans_rec / large_loans_rec))} small employers for every large employer. ")
# Functions to plot small or big sized charts

def small_chart(plot):
    plot.rcParams.update(plt.rcParamsDefault)
    
def big_chart(plot):
    parameters = {'xtick.labelsize': 50, 'ytick.labelsize': 50, 'axes.labelsize':100, 'legend.title_fontsize':50, 'axes.titlesize':150}
    plot.rcParams.update(parameters)
small_chart(plt)
sns.barplot('LoanRange', 'Jobs Retained - AVG', data=summary_all_loans)
plt.xticks(rotation=90)
plt.title('How many jobs were retained by each Loan Group on average?')
plt.xlabel('Loan Range')
plt.ylabel('Jobs Retained - AVG')
plt.show()
small_chart(plt)
sns.barplot('LoanRange', 'Total Jobs Retained', data=summary_all_loans)
plt.xticks(rotation=90)
plt.title('How many jobs in total were retained by each Loan Group?')
plt.xlabel('Loan Range')
plt.ylabel('Total Jobs Retained')
plt.show()
small_chart(plt)
sns.barplot('LoanRange', 'No of Applicants | Recipients', data=summary_all_loans)
plt.xticks(rotation=90)
plt.title('How many businesses applied for each Loan category?')
plt.xlabel('Loan Range')
plt.ylabel('No of Applicants | Recipients')
plt.show()
# Businesses whose names appear more than once.
# Note that they could be different businesses with the same name except in some:

all_businesses = df['BusinessName'].to_list()

# removing missing values
businesses_minus_omitted = df[df['BusinessName'] != 'Omitted']
businesses_minus_omitted = businesses_minus_omitted[businesses_minus_omitted['BusinessName'] != '-']


businesses_minus_omitted.columns.name = 'S/N'
businesses_duplicated = businesses_minus_omitted[businesses_minus_omitted.duplicated(subset=['BusinessName'], keep=False)]
businesses_duplicated.sort_values(by='BusinessName', ascending=True, inplace=True)

applied_more_than_once = len(businesses_duplicated['BusinessName'].unique())

print(f"Upto {applied_more_than_once} businesses (with the same name) applied more than one time.")

# businesses_duplicated.head(50)
num_nonprofit = df_new[df_new['NonProfit'] == 'Y']['NonProfit'].count()
print(f'Upto {num_nonprofit:,} NonProfits applied for the PPP loan.')
#### NON PROFITS AND FOR PROFITS

category_list = ['a $5-10 million', 'b $2-5 million', 'c $1-2 million', 'd $350,000-1 million', 'e $150,000-350,000', 'f $1 - 150,000']

nonprofit_no = {}

for num, i in enumerate(category_list):
    result = df[df['LoanRange'] == i]['NonProfit'].count()
    nonprofit_no[i[:1]] = result
    
print(nonprofit_no)

nonprofit_jobs_retained = {}

for num, i in enumerate(nonprofit_no):
    result_ = df[df['LoanRange'] == category_list[num]]
    result = int(result_[result_['NonProfit'] == 'Y']['JobsRetained'].sum())
    nonprofit_jobs_retained[i] = result
    
print(nonprofit_jobs_retained)
#Prepping rank DataFrame
lenders = df_new['Lender'].value_counts().to_frame()
lenders.columns = ['No of Loans']

df_lenders = pd.DataFrame({"Lender's Name": [i for i in lenders.index],
                         "No of Loans": [i for i in lenders['No of Loans']]})

# Getting number of lenders and loans.
num_of_lenders = len(df['Lender'].unique())
num_of_loans = df_new['Lender'].value_counts().sum()

# Presenting number of lenders and loans.
print(f"{num_of_lenders} lenders processed {num_of_loans} loans.")

print()
print('Top 5 lenders')
df_lenders.head()
big_chart(plt)
plt.figure(figsize=(100,75))
sns.barplot("Lender's Name", 'No of Loans', data=df_lenders[:51])
plt.xticks(rotation=90)
plt.title('How many Paycheck Protection loans did banks lend out?')
plt.xlabel('Bank Name')
plt.ylabel('No of loans')
plt.show()
group_a = df[df['LoanRange'] == 'a $5-10 million']
# group_a.describe(include='all')
highest_groupa_lenders = group_a['Lender'].value_counts().to_frame()
highest_groupa_lenders.columns = ['No of $5m+ loans']
top_5mplus = pd.DataFrame({"Lender's Name": [i for i in highest_groupa_lenders.index],
                          "No of Loans": [i for i in highest_groupa_lenders['No of $5m+ loans']]})
top_5mplus.head(15)
plt.figure(figsize=(100,75))
sns.barplot("Lender's Name", 'No of Loans', data=top_5mplus[:50])
plt.xticks(rotation=90)
plt.title('How many $5m+ Paycheck Protection loans did banks lend out?')
plt.xlabel('Bank Name')
plt.ylabel('No of loans')
plt.show()
group_b = df[df['LoanRange'] == 'b $2-5 million']
# group_b.describe(include='all')
highest_groupb_lenders = group_b['Lender'].value_counts().to_frame()
highest_groupb_lenders.columns = ['No of $2-5m loans']
top_2m5m = pd.DataFrame({"Lender's Name": [i for i in highest_groupb_lenders.index],
                          "No of Loans": [i for i in highest_groupb_lenders['No of $2-5m loans']]
                          })
top_2m5m.head(15)
plt.figure(figsize=(100,75))
sns.barplot("Lender's Name", 'No of Loans', data=top_2m5m[:50])
plt.xticks(rotation=90)
plt.title('How many $2m-5m Paycheck Protection loans did banks lend out?')
plt.xlabel('Bank Name')
plt.ylabel('No of loans')
plt.show()
approvals_general = pd.DataFrame({'Date of Approval': df_new['DateApproved'].value_counts().index.to_list(),
                                  'No of Approvals': df_new['DateApproved'].value_counts().to_list()})
plt.figure(figsize=(100,75))
plt.xticks(rotation=90)
sns.barplot(approvals_general['Date of Approval'], approvals_general['No of Approvals'])
plt.title('Determining the most active and least active days of approvals')
plt.ylabel('No of loans approved')
plt.show()
approvals_general.sort_values(by='Date of Approval', inplace=True)
plt.figure(figsize=(100,75))
plt.xticks(rotation=90)
sns.barplot(approvals_general['Date of Approval'], approvals_general['No of Approvals'])
plt.title('How all loan approvals played out over time.')
plt.ylabel('No of loans approved')
plt.show()
approvals_early = approvals_general[approvals_general['Date of Approval'] < '04/13/2020']['No of Approvals'].sum()
applications_early  = approvals_early

total_applications = approvals_general['No of Approvals'].sum()

percent_applications_early = (applications_early / total_applications) * 100

print(f"Overall, {percent_applications_early:.2f}% of all businesses applied early i.e within 9 days of the program's commencement.")
frame_after_412 = approvals_general['Date of Approval'] > '04/12/2020'
frame_before_503 = approvals_general['Date of Approval'] < '05/03/2020'

approvals_peak_periods = approvals_general[(frame_after_412) & (frame_before_503)]['No of Approvals'].sum()

total_applications = approvals_general['No of Approvals'].sum()

percent_applications_peakperiods = (approvals_peak_periods / total_applications) * 100

print(f"Overall, {percent_applications_peakperiods:.2f}% of all businesses applied during the peak periods i.e between April 13 and May 3rd.")
approvals_late = approvals_general[approvals_general['Date of Approval'] > '06/22/2020']['No of Approvals'].sum()
applications_late = approvals_late

total_applications = approvals_general['No of Approvals'].sum()

percent_applications_late = (applications_late / total_applications) * 100

print(f"Overall, {percent_applications_late:.2f}% of all businesses applied late i.e within 8 days to June 30.")
# Prepping group e DataFrame
e = df2.copy()
e['LoanRange'] = 'f $1 - 150,000'
# e.head()
# POPULATING doa_dict
doa_dict = {} # to contain DataFrames of each loan group

    # Using loan group names from category_list to map the DataFrames
for i in category_list:
    if i[0] in 'abcde':
        doa_dict[i] = df[df['LoanRange'] == i]
    elif i[0] == 'f':
        doa_dict[i] = e[e['LoanRange'] == i]

# Accounting for number of loans approved on the most active and least active days of approval.
approved_415 = 0
approved_607 = 0

for i in doa_dict.keys():
    in_sight_dates = doa_dict[i][['DateApproved','Lender']]
    in_sight_415_banks = in_sight_dates[in_sight_dates['DateApproved'] == '05/03/2020']
    approved_415 += in_sight_415_banks['DateApproved'].count()
    
    in_sight_607_banks = in_sight_dates[in_sight_dates['DateApproved'] == '06/07/2020']
    approved_607 += in_sight_607_banks['DateApproved'].count()
    
print(f"Overall, {approved_415:,} loans were approved on the peak day for approvals     - 04/15/2020")
print(f"Overall, {approved_607:,} loans were approved on the least active day for approvals   - 06/07/2020")
doa_refined = {}
for i in doa_dict.keys():
    
    doa_value_count = doa_dict[i]['DateApproved'].value_counts()
    
    doa_refined[i] = pd.DataFrame({i + ' loan dates': doa_value_count.index.to_list(),
                                          'No of Group ' + i + ' loans approved': doa_value_count.to_list()})
    
percent_early = []
percent_peak = []
percent_late = []

for i in doa_refined.keys():
    
    # APPLICATIONS
    # early applications
    approvals_early = doa_refined[i][doa_refined[i][i + ' loan dates'] < '04/13/2020']['No of Group ' + i + ' loans approved'].sum()
    applications_early  = approvals_early
    total_applications = doa_refined[i]['No of Group ' + i + ' loans approved'].sum()
    percent_applications_early = (applications_early / total_applications) * 100
    percent_early.append(percent_applications_early)
    print(f"{percent_applications_early:.2f}% of {i[2:]} PPP loan applicants applied early i.e within 9 days of the program's commencement.")
    
    # peak periods applications
    frame_after_412 = doa_refined[i][i + ' loan dates'] > '04/12/2020'
    frame_before_503 = doa_refined[i][i + ' loan dates'] < '05/03/2020'
    approvals_peak_periods = doa_refined[i][(frame_after_412) & (frame_before_503)]['No of Group ' + i + ' loans approved'].sum()    
    total_applications = doa_refined[i]['No of Group ' + i + ' loans approved'].sum()
    percent_applications_peakperiods = (approvals_peak_periods / total_applications) * 100
    percent_peak.append(percent_applications_peakperiods)
    print(f"{percent_applications_peakperiods:.2f}% of {i[2:]} PPP loan applicants applied for the PPP loans during the peak periods i.e between April 13 and May 3rd.")
    
    
    # late applications
    approvals_late = doa_refined[i][doa_refined[i][i + ' loan dates'] > '06/22/2020']['No of Group ' + i + ' loans approved'].sum()
    applications_late = approvals_late
    total_applications = doa_refined[i]['No of Group ' + i + ' loans approved'].sum()
    percent_applications_late = (applications_late / total_applications) * 100
    percent_late.append(percent_applications_late)
    print(f"{percent_applications_late:.2f}% of {i[2:]} PPP loan applicants applied late i.e within 8 days to June 30.")
    
    doa_refined[i].sort_values(by=i + ' loan dates', inplace=True)
    plt.figure(figsize=(100,75))
    sns.barplot(doa_refined[i][i + ' loan dates'], doa_refined[i]['No of Group ' + i + ' loans approved'])
    plt.xlabel('Dates of Approval')
    plt.ylabel('No of loans approved')
    plt.xticks(rotation=90)
    plt.title('How approvals played out for ' + i[2:] + ' loans')
    plt.show()
    
pd.set_option('display.width', 3000)
summary_all_loans['% Early Applications'] = percent_early
summary_all_loans['% Peak Period Applications'] = percent_peak
summary_all_loans['% Late Applications'] = percent_late

summary_all_loans.head(6)
titles = ['How many businesses made early loan applications?', 
          'What percentage of businesses applied during the peak period',
         '% late applications for each loan group?']

small_chart(plt)
for num, i in enumerate(summary_all_loans.columns[4:7]):
    sns.barplot('LoanRange', i, data=summary_all_loans)
    plt.title(titles[num])
    plt.xticks(rotation=90)
    plt.show()
lenders_pd_dict = {} # to be populated with 'Approval Dates' DataFrames of the top 5 lenders

# Populating the lenders_pd_dict
for i in df_lenders["Lender's Name"].to_list()[0:5]:

    df_JP = df_new[df_new['Lender'] == i]
    approvals_JP = pd.DataFrame({'Date of Approval': df_JP['DateApproved'].value_counts().index.to_list(),
                                  'No of Approvals': df_JP['DateApproved'].value_counts().to_list()})
    approvals_JP.sort_values(by='Date of Approval', inplace=True)
    
    
    lenders_pd_dict[i] = approvals_JP
    
    peak_approval_day_info = lenders_pd_dict[i][lenders_pd_dict[i]['No of Approvals'] == lenders_pd_dict[i]['No of Approvals'].max()]
    peak_approval_day = peak_approval_day_info.loc[0, 'Date of Approval']
    peakday_no_approvals = peak_approval_day_info.loc[0, 'No of Approvals']
    
    print(f"'{i}' approved {peakday_no_approvals:,} Paycheck Protection Program loans on it's peak day for approvals - {peak_approval_day}")
    
    big_chart(plt)
    
    # charts
    plt.figure(figsize=(100, 75))
    sns.barplot(lenders_pd_dict[i]['Date of Approval'], lenders_pd_dict[i]['No of Approvals'])
    plt.xticks(rotation=90)
    plt.title('How ' + i + ' processed loans.')
    plt.show()
    
    
# approvals_JP.head(50)
# STATES LOANS BELOW $150,000k WENT - TOP 100 CITIES.

cities_informative = df_new['State'].value_counts().to_frame()
cities_informative.columns = ['No of Loans']

cities_no_loans = pd.DataFrame({'States': cities_informative.index.to_list()[:51],
                                'No of Loans': cities_informative['No of Loans'].to_list()[:51]})

cities_total_loans_below150k_50 = df2[['State', 'LoanRange']].groupby(['State'], as_index=False).sum()[:51]
cities_total_loans_below150k_50.sort_values(by='LoanRange', ascending=False, inplace=True)
# cities_no_loans_below150k

cities_info = [cities_no_loans, cities_total_loans_below150k_50]
titles = ['No of loans that went to each state - Top 50', '$1-150,000 loans that went to each state - Top 50']


big_chart(plt)

for num, i in enumerate(cities_info):
    plt.figure(figsize=(100,75))
    sns.barplot(i[i.columns[0]], i[i.columns[1]], data=i)
    plt.title(titles[num])
    plt.xticks(rotation=90)

    plt.xlabel(i.columns[0])
    plt.ylabel(i.columns[1])
    if num == 1:
        plt.ylabel('Loan Amount')
    plt.show()
df_new['NAICSCode'] = df_new['NAICSCode'].astype(str)
df_new['NAICSCode'] = [i[:2] for i in df_new['NAICSCode'].to_list()]
df_new[df_new['NAICSCode'].isnull()] = '99'
df_n = pd.read_csv('../input/2017-naics-codes-summary/2017_NAICS_Structure_Summary_Table .csv')
# Including Unclassified businesses in NAICSCodes  file. Source: https://classcodes.com/naics-code-list/
df_n.loc[25] = [99, 'Unclassified']
# Delicate
df_n.drop(0, axis=0, inplace=True)
df_n.rename({'Sector': 'NAICSCode', 'Name': 'Industry'}, axis=1, inplace=True)
df_n['NAICSCode'] = df_n['NAICSCode'].astype(str)
df_n['NAICSCode'] = [i[:2] for i in df_n['NAICSCode'].to_list()]
df_n.reset_index(drop=True, inplace=True)
# Including the industries each business belongs to in df_new_industry
df_new_industry = pd.merge(df_new, df_n, on='NAICSCode', how='left')
df_new_industry = df_new_industry[['LoanRange', 'BusinessName', 'Address', 'City', 'State', 'Zip', 'NAICSCode', 'Industry', 'BusinessType', 'RaceEthnicity', 'Gender', 'Veteran', 'NonProfit', 'JobsRetained', 'DateApproved', 'Lender', 'CD']]

df_new_industry['NAICSCode'].count()
df_new_industry_ = df_new_industry['Industry'].value_counts().to_frame()
df_new_industry_.columns = ['No of loans']
# Plotting 
big_chart(plt)
plt.figure(figsize=(100,75))
plot = sns.barplot(df_new_industry_.index, df_new_industry_['No of loans'])
plt.title('How many loans went to each industry?')
plt.xticks(rotation=90)
plt.xlabel('No of loans')
plot.xaxis.label.set_size(50)
plt.show()
# Another delicate

df2_industry = df2.copy()

# converting null values to 99 i.e unclassified
df2_industry[df2_industry['NAICSCode'].isnull()] = '99'

df2_industry['LoanRange'] = df2_industry['LoanRange'].astype(float)
df2_industry['NAICSCode'] = df2_industry['NAICSCode'].astype(str)
df2_industry['NAICSCode'] = [i[:2] for i in df2_industry['NAICSCode'].to_list()]

# Merging files
df2_industry_ = pd.merge(df2_industry, df_n, on='NAICSCode', how='left')

# Including the industries each business belongs to in df2_industry
df2_industry_ = df2_industry_[['LoanRange', 'BusinessName', 'Address', 'City', 'State', 'Zip', 'NAICSCode', 'Industry', 'BusinessType', 'RaceEthnicity', 'Gender', 'Veteran', 'NonProfit', 'JobsRetained', 'DateApproved', 'Lender', 'CD']]
df2_industry_ii = df2_industry_[['Industry', 'LoanRange']].groupby(['Industry'], as_index=False).sum()
df2_industry_ii.sort_values(by='LoanRange', ascending=False, inplace=True)
# Plotting 
big_chart(plt)
plt.figure(figsize=(100,75))
plot = sns.barplot(df2_industry_ii.Industry, df2_industry_ii.LoanRange)
plt.title('How much loans upto $150,000 went to each industry?')
plt.xticks(rotation=90)
plt.xlabel('Loan Amount')
plot.xaxis.label.set_size(50)
plt.show()