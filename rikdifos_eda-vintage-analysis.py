%matplotlib inline

%config InlineBackend.figure_format = 'svg'



import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd    

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

credit = pd.read_csv('../input/credit-card-approval-prediction/credit_record.csv')  

application = pd.read_csv('../input/credit-card-approval-prediction/application_record.csv') 

credit
application
len(set(application['ID'])) # how many unique ID in application record?
len(set(credit['ID'])) # how many unique ID in credit record?
len(set(application['ID']).intersection(set(credit['ID']))) # how many IDs do two tables share?
grouped = credit.groupby('ID')

### convert credit data to wide format which every ID is a row

pivot_tb = credit.pivot(index = 'ID', columns = 'MONTHS_BALANCE', values = 'STATUS')

pivot_tb['open_month'] = grouped['MONTHS_BALANCE'].min() # smallest value of MONTHS_BALANCE, is the month when loan was granted

pivot_tb['end_month'] = grouped['MONTHS_BALANCE'].max() # biggest value of MONTHS_BALANCE, might be observe over or canceling account

pivot_tb['ID'] = pivot_tb.index

pivot_tb = pivot_tb[['ID', 'open_month', 'end_month']]

pivot_tb['window'] = pivot_tb['end_month'] - pivot_tb['open_month'] # calculate observe window

pivot_tb.reset_index(drop = True, inplace = True)

credit = pd.merge(credit, pivot_tb, on = 'ID', how = 'left') # join calculated information

credit0 = credit.copy()

credit = credit[credit['window'] > 20] # delete users whose observe window less than 20

credit['status'] = np.where((credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 1, 0) # analyze > 60 days past due 

credit['status'] = credit['status'].astype(np.int8) # 1: overdue 0: not

credit['month_on_book'] = credit['MONTHS_BALANCE'] - credit['open_month'] # calculate month on book: how many months after opening account

credit.sort_values(by = ['ID','month_on_book'], inplace = True)



##### denominator

denominator = pivot_tb.groupby(['open_month']).agg({'ID': ['count']}) # count how many users in every month the account was opened

denominator.reset_index(inplace = True)

denominator.columns = ['open_month','sta_sum']



##### ventage table

vintage = credit.groupby(['open_month','month_on_book']).agg({'ID': ['count']}) 

vintage.reset_index(inplace = True)

vintage.columns = ['open_month','month_on_book','sta_sum'] 

vintage['due_count'] = np.nan

vintage = vintage[['open_month','month_on_book','due_count']] # delete aggerate column

vintage = pd.merge(vintage, denominator, on = ['open_month'], how = 'left') # join sta_sum colun to vintage table

vintage
for j in range(-60,1): # outer loop: month in which account was opened

    ls = []

    for i in range(0,61): # inner loop time after the credit card was granted

        due = list(credit[(credit['status'] == 1) & (credit['month_on_book'] == i) & (credit['open_month'] == j)]['ID']) # get ID which satisfy the condition

        ls.extend(due) # As time goes, add bad customers

        vintage.loc[(vintage['month_on_book'] == i) & (vintage['open_month'] == j), 'due_count'] = len(set(ls)) # calculate non-duplicate ID numbers using set()

        

vintage['sta_rate']  = vintage['due_count'] / vintage['sta_sum'] # calculate cumulative % of bad customers

vintage        
### Vintage wide table

vintage_wide = vintage.pivot(index = 'open_month',

                             columns = 'month_on_book',

                             values = 'sta_rate')

vintage_wide
# plot vintage line chart

vintage0 = vintage_wide.replace(0,np.nan)

lst = [i for i in range(0,61)]

vintage_wide[lst].T.plot(legend = False, grid = True, title = 'Cumulative % of Bad Customers (> 60 Days Past Due)')

#plt.axvline(30)

#plt.axvline(25)

#plt.axvline(20)

plt.xlabel('Months on Books')

plt.ylabel('Cumulative % > 60 Days Past Due')

plt.show()
lst = []

for i in range(0,61):

    ratio = len(pivot_tb[pivot_tb['window'] < i]) / len(set(pivot_tb['ID']))

    lst.append(ratio)

    

pd.Series(lst).plot(legend = False, grid = True, title = ' ')

plt.xlabel('Observe Window')

plt.ylabel('account ratio')

plt.show()
def calculate_observe(credit, command):

    '''calculate observe window

    '''

    id_sum = len(set(pivot_tb['ID']))

    credit['status'] = 0

    exec(command)

    #credit.loc[(credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1

    credit['month_on_book'] = credit['MONTHS_BALANCE'] - credit['open_month']

    minagg = credit[credit['status'] == 1].groupby('ID')['month_on_book'].min()

    minagg = pd.DataFrame(minagg)

    minagg['ID'] = minagg.index

    obslst = pd.DataFrame({'month_on_book':range(0,61), 'rate': None})

    lst = []

    for i in range(0,61):

        due = list(minagg[minagg['month_on_book']  == i]['ID'])

        lst.extend(due)

        obslst.loc[obslst['month_on_book'] == i, 'rate'] = len(set(lst)) / id_sum 

    return obslst['rate']



command = "credit.loc[(credit['STATUS'] == '0') | (credit['STATUS'] == '1') | (credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"   

morethan1 = calculate_observe(credit, command)

command = "credit.loc[(credit['STATUS'] == '1') | (credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"   

morethan30 = calculate_observe(credit, command)

command = "credit.loc[(credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"

morethan60 = calculate_observe(credit, command)

command = "credit.loc[(credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"

morethan90 = calculate_observe(credit, command)

command = "credit.loc[(credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"

morethan120 = calculate_observe(credit, command)

command = "credit.loc[(credit['STATUS'] == '5'), 'status'] = 1"

morethan150 = calculate_observe(credit, command)
obslst = pd.DataFrame({'past due more than 30 days': morethan30,

                       'past due more than 60 days': morethan60,

                       'past due more than 90 days': morethan90,

                       'past due more than 120 days': morethan120,

                       'past due more than 150 days': morethan150

                        })

obslst.plot(grid = True, title = 'Cumulative % of Bad Customers Analysis')

plt.xlabel('Months on Books')

plt.ylabel('Cumulative %')

plt.show()
def calculate_rate(pivot_tb, command): 

    '''calculate bad customer rate

    '''

    credit0['status'] = None

    exec(command) # excuate input code

    sumagg = credit0.groupby('ID')['status'].agg(sum)

    pivot_tb = pd.merge(pivot_tb, sumagg, on = 'ID', how = 'left')

    pivot_tb.loc[pivot_tb['status'] > 1, 'status'] = 1

    rate = pivot_tb['status'].sum() / len(pivot_tb)

    return round(rate, 5)



command = "credit0.loc[(credit0['STATUS'] == '0') | (credit0['STATUS'] == '1') | (credit0['STATUS'] == '2') | (credit0['STATUS'] == '3' )| (credit0['STATUS'] == '4' )| (credit0['STATUS'] == '5'), 'status'] = 1"   

morethan1 = calculate_rate(pivot_tb, command)

command = "credit0.loc[(credit0['STATUS'] == '1') | (credit0['STATUS'] == '2') | (credit0['STATUS'] == '3' )| (credit0['STATUS'] == '4' )| (credit0['STATUS'] == '5'), 'status'] = 1"   

morethan30 = calculate_rate(pivot_tb, command)

command = "credit0.loc[(credit0['STATUS'] == '2') | (credit0['STATUS'] == '3' )| (credit0['STATUS'] == '4' )| (credit0['STATUS'] == '5'), 'status'] = 1"

morethan60 = calculate_rate(pivot_tb, command)

command = "credit0.loc[(credit0['STATUS'] == '3' )| (credit0['STATUS'] == '4' )| (credit0['STATUS'] == '5'), 'status'] = 1"

morethan90 = calculate_rate(pivot_tb, command)

command = "credit0.loc[(credit0['STATUS'] == '4' )| (credit0['STATUS'] == '5'), 'status'] = 1"

morethan120 = calculate_rate(pivot_tb, command)

command = "credit0.loc[(credit0['STATUS'] == '5'), 'status'] = 1"

morethan150 = calculate_rate(pivot_tb, command)



summary_dt = pd.DataFrame({'situation':['past due more than 1 day',

                               'past due more than 30 days',

                               'past due more than 60 days',

                               'past due more than 90 days',

                               'past due more than 120 days',

                               'past due more than 150 days'],

                      'bad customer ratio':[morethan1,

                               morethan30,

                               morethan60,

                               morethan90, 

                               morethan120,

                               morethan150, 

                      ]})

summary_dt