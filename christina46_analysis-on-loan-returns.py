import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import matplotlib 

matplotlib.style.use('ggplot')

import seaborn as sns

import datetime as dt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#Read CSV file into DataFrame

#Convert missing values to NaN

#Set column id as index

nan_values = ['nan','N.A','NaN','n/a']

data = pd.read_csv('../input/loan.csv',na_values=nan_values, encoding = "ISO-8859-1", index_col='id')  
#briefly examine our data

data.info()
pd.set_option('display.max_columns',100)

data.head(3)
#1. make sure that all loans have been funded 

data = data[data['funded_amnt']>0]

#2. columns such as issue_d, loan_states and last_pymnt_d are essential for calculating the loan period. remove rows with missing data

data = data[(data['issue_d'].notnull()) & (data['loan_status'].notnull())]

data = data[(data['last_pymnt_d'].notnull() | (data['loan_status']!='Fully Paid'))]

#3. convert a string date to datetime formate

def str_to_dt(a_string,conv_format='%b-%Y'):

    try:

        return dt.datetime.strptime(a_string,conv_format)

    except:

        return None    

data['issue_d'] = data['issue_d'].apply(str_to_dt)

data['last_pymnt_d'] = data['last_pymnt_d'].apply(str_to_dt)
compounded_nar_li = list()

simple_nar_li = list()



update_d = str_to_dt('Jan-2016') # last update time of our data 



# calculate net annualized return for each loan (row-wise)

for index, row in data.iterrows():

    net_interest = row['total_rec_int'] + row['total_rec_late_fee'] - 0.01*row['total_pymnt_inv']

    net_charge_offs = 0



    # specify loan period based on the current loan status

    if row['loan_status'] == 'Fully Paid':

        loan_period = (row['last_pymnt_d'] - row['issue_d'])/ np.timedelta64(1,'D')/30

    elif row['loan_status'] == 'Charged Off':

        net_charge_offs = row['funded_amnt_inv'] - row['total_rec_prncp'] - row['recoveries'] + row['collection_recovery_fee']

        active_period = 0

        if row['last_pymnt_d']>dt.datetime(2006,12,30,0,0):

            active_period = (row['last_pymnt_d'] - row['issue_d'])/  np.timedelta64(1,'D')/30

        up_to_now = (update_d-row['issue_d'])/ np.timedelta64(1,'D')/30 -1

        #Charge off typically occurs when a loan is no later than 5 months past due

        loan_period = min(active_period+6, up_to_now)

    else:

        loan_period = (update_d-row['issue_d'])/ np.timedelta64(1,'D')/30 - 1

 

    loan_period = int(loan_period)  

    if loan_period>0:

        t=12/loan_period

    else:   #occasionally, the last repayment occured in the month of issuance

        t=12



    #calculate both compounded returns and simple returns

    compounded_nar = (1 + (net_interest-net_charge_offs) / row['funded_amnt'])**t -1 

    simple_nar = t*(net_interest-net_charge_offs)/row['funded_amnt']



    compounded_nar_li.append(compounded_nar)

    simple_nar_li.append(simple_nar)
data['simple_nar'] = simple_nar_li

data['compounded_nar'] =  compounded_nar_li

data = data[data['compounded_nar'].notnull()]
#plot histogram of annualized return

plt.hist(data['simple_nar'],color='salmon', bins=24, range=(-1,1), label='simple_nar')

plt.xlabel('simple annualized return')

plt.ylabel('frequency')
grade_group = data.groupby('grade')

grade_group.mean()['simple_nar'].plot(kind='bar', color='salmon')

plt.title('average return grouped by loan grade')
grade_group = data.groupby('sub_grade', as_index=False)

grade_group.mean()['simple_nar'].plot()

plt.title('average return grouped by sub-grade')
purpose_group = data.groupby('purpose')

purpose_group.mean()['simple_nar'].plot(kind='bar',figsize=(10,6),color='salmon')

plt.title('average return grouped by purpose')
gra_pur_group = data.groupby(['purpose','grade'])

gra_pur_group.mean()['simple_nar'].unstack().plot(kind='bar',figsize=(9,6))

plt.title('average return grouped by grade and purpose')
term_group = data.groupby('term')

term_group.mean()['simple_nar'].plot(kind='bar',figsize=(5,5))

plt.title('average return grouped by loan term')