# Imports:

import matplotlib.pyplot as plt

from matplotlib import font_manager as fm

%matplotlib inline

import pandas as pd

import numpy as np

import sqlite3 as sql

# import seaborn as sns
# Connect to database and pull in raw data

db_con = sql.connect('../input/database.sqlite')

data = pd.read_sql_query('SELECT * FROM loan', db_con)



# Print Sample

print(data.head())
# Summary of loan 'status' data

summary = data.groupby(['loan_status']).count()['index'].sort_values(ascending=False)

print(summary)
# Clean the raw data:



# To ensure that a 'loan_status' value is available for each row, we get the totals of rows with/ without the status value:

print('Out of {} records, {} have a status, and {} do not'.format(len(data['loan_status']),

                                                             data['loan_status'].notnull().sum(),

                                                             data['loan_status'].isnull().sum()))



# Remove rows with no 'loan_status'

data = data[data.loan_status.notnull()]



# This function characterizes loans



def clean_status(raw_status):

    

    status = ""

    raw_status = str(raw_status).lower().strip()

    

    if 'charged' in raw_status:

        status = 'charged_off'        

    elif 'default' in raw_status:

        status = 'default'

    elif 'paid' in raw_status:

        status = 'paid'   

    elif ('grace' and 'period') in raw_status:

        status = 'grace_per'

    elif 'current' in raw_status:

        status = 'current' 

    elif 'issued' in raw_status:

        status = 'current' 

    elif ('late' and '16-30') in raw_status:

        status = 'late16_30' 

    elif ('late' and '31-120') in raw_status:

        status = 'late31_120' 

    else:

        # There shouldn't be any 'uncategorized' loans, but we'll be able to find them if there are any

        # using this label:

        status = 'uncategorized'

        

    return status



# Add a 'clean_status' column

data['clean_status'] = data.apply(lambda row:clean_status(row["loan_status"]) , axis = 1)



print('{} records are Uncategorized'.format(len(data[data['clean_status'] == 'uncategorized'])))
# Next, we'll see the overall performance of Lending Club loans



# Total value of loans issued:

val_issued = data['loan_amnt'].sum()



# Total values for each status category. To make ploting graphs easier, we create a Dict for each:

val_current = {'value': data[data['clean_status'] == 'current']['loan_amnt'].sum(), 

               'label': 'Current', 'color': 'green'}

val_paid = {'value': data[data['clean_status'] == 'paid']['loan_amnt'].sum(), 

               'label': 'Paid', 'color': 'green'}

val_grace_per = {'value': data[data['clean_status'] == 'grace_per']['loan_amnt'].sum(), 

               'label': 'In Grace Period', 'color': 'green'}

val_late16_30 = {'value': data[data['clean_status'] == 'late16_30']['loan_amnt'].sum(), 

               'label': 'Late 16-30 Days', 'color': 'yellow'}

val_late31_120 = {'value': data[data['clean_status'] == 'late31_120']['loan_amnt'].sum(), 

               'label': 'Late 31-120 Days', 'color': 'yellow'}

val_default = {'value': data[data['clean_status'] == 'default']['loan_amnt'].sum(), 

               'label': 'Default', 'color': 'red'}

val_charged_off = {'value': data[data['clean_status'] == 'charged_off']['loan_amnt'].sum(), 

               'label': 'Charged Off', 'color': 'red'}

val_uncat = {'value': data[data['clean_status'] == 'uncategorized']['loan_amnt'].sum(), 

               'label': 'Uncategorized', 'color': 'gray'}
# Plot a pie chart:



fig = plt.figure(1, figsize=(8,8))

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

plt.title("Overall Loan Status",y=1.08,fontweight="bold")



labels = [val_current.get('label'), 

          val_paid.get('label'), 

          val_late16_30.get('label'), 

          val_late31_120.get('label'), 

          val_default.get('label'), 

          val_charged_off.get('label'), 

          val_uncat.get('label')]



values = [val_current.get('value'), 

          val_paid.get('value'), 

          val_late16_30.get('value'), 

          val_late31_120.get('value'), 

          val_default.get('value'), 

          val_charged_off.get('value'), 

          val_uncat.get('value')]



colors = [val_current.get('color'), 

          val_paid.get('color'), 

          val_late16_30.get('color'), 

          val_late31_120.get('color'), 

          val_default.get('color'), 

          val_charged_off.get('color'), 

          val_uncat.get('color')]



patches, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors,

                                  pctdistance=1.1, labeldistance=1.18)



proptease = fm.FontProperties()

proptease.set_size('x-small')

plt.setp(autotexts, fontproperties=proptease)

plt.setp(texts, fontproperties=proptease)



plt.show()
'''

# View overall performance of all loans

funded_amnt = data['funded_amnt'].sum() # Amt. issued to borrower

total_pymnt = data['total_pymnt'].sum() # Payments received to date for total amount funded

out_prncp = data['out_prncp'].sum() # Remaining outstanding principal

recoveries = data['recoveries'].sum() # post charge off gross recovery



# We will be using the following data to find correlation with defaults:



loan_amnt

total_bal_il

revol_bal

grade

delinq_2yrs

average of: fico_range_high, fico_range_low



'''