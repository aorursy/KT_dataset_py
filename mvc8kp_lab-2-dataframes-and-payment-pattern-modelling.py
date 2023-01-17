'''

cmi = Monthly Installment (to be calculated)

s = Amount to Settle

r = Monthly Interest Rate

n = Months in the term

'''



s = 100000

r = 0.005

n = 180
def cmi_calculator(rate, nper, fv):

    cmi = fv * ((rate * (1 + rate) ** nper) / ((1 + rate) ** nper - 1))

    return cmi
#run recalculator and define interesting variables

cmi = cmi_calculator(r, n, s)

total_paid = cmi * n

total_interest = total_paid - s
#Think of pandas as a nice table-like representation of your data, on steroids

import pandas as pd

help(pd)
'''

We can create a DataFrame directly from a dictionary

Dictionaries are complex data type. 

They are conformed by key : value pairs and help keep data organised an easily accessible.

Just think of it, as a normal dictionary :)

'''



#The values for our dictionary will be lists of the same length. This will represent the rows in our table

data = {'metrics' : ['total_paid', 'total_interest'],

       'values' : [total_paid, total_interest]}



df = pd.DataFrame(data)



print(f'Your dictionary looks like this: ')

print(data)

print('')



print(f'Your best friend looks like this: ')

df
'''

You can easily select only the values of interest from your DataFrames 

    df['metrics']: Select only the 'metrics' column

    df['values']: Select only the 'values' column

    df[df['metrics'] == 'total_paid']: Select the slice of the table where the 'metrics' is called 'total_paid'

'''



n_dashes = 30



print('Printing Metrics')

print(df['metrics'])

print('-'*n_dashes)



print('Printing Values')

print(df['values'])

print('-'*n_dashes)



print('Printing Total Paid Values')

print(df[df['metrics'] == 'total_paid'])

print('-'*n_dashes)





print('When you select a single column (or row...) your output will be a "pandas Series" rather than a DataFrame')

print(type(df['metrics']))
#You can select data by row index

print('Row at index 0')

display(df.iloc[0])



print('Row at index 1')

display(df.iloc[1])
#You can replace the values of that row

df_copy = df.copy()

df_copy.iloc[1] = 50

display(df_copy)



#You can replace a specific cell

df_copy2 = df.copy()

df_copy2.at[1, 'values'] = 50

display(df_copy2)
#You can add new columns by 'Selecting' a (new) column and 'Assigning' a value to it

df_copy3 = df.copy()

df_copy3['Added_Column'] = 'Assigned Value'

df_copy3.head()
#You can remove a column with the 'drop' function

try:

    df_copy3 = df_copy3.drop(columns='Added_Column')

except KeyError:

    print('Column to be dropped does not exists')

display(df_copy3.head())
try:

    del df_copy, df_copy2, df_copy3

    del s, r, n

    del cmi, total_paid, total_interest

except NameError as e:

    print(f'At least one of specified variables no longer exists')

    print(e)
#Define the core mortgage inputs

amount_borrowed = 100000.00

periods = 180

monthly_rate = 0.06 / 12



#Create one data point per period

list_period = [x for x in range(periods)]

list_monthly_rate = [0.005 for x in range(periods)]

list_remaining_months = [periods - x for x in range(periods)]



print(f'First five periods: {list_period[:5]}')

print(f'First five rates: {list_monthly_rate[:5]}')

print(f'First five remaining months: {list_remaining_months[:5]}')
#Create a dataframe with the mortgage timeline

data = {'period' : list_period,

        'remaining_months' : list_remaining_months,

        'monthly_rate' : list_monthly_rate,

        'amount_borrowed' : amount_borrowed}



df_original = pd.DataFrame(data).set_index('period')

df_original.head()
df = df_original.copy()



#We can now calculate the CMI for every month 

cmi = cmi_calculator(df['monthly_rate'], df['remaining_months'], df['amount_borrowed'])

print(cmi[:5])

print('Total CMI Over Term {}'.format(sum(cmi)))
def payment_pattern_calculator(df):



    #Make copy of original df to avoid overwrites

    df = df.copy()



    #Define placeholders for the CMI values

    cmi_breakdown = (0.00, 0.00, 0.00)

    df['cmi_total'] = cmi_breakdown[0]

    df['cmi_interest'] = cmi_breakdown[1]

    df['cmi_settle'] = cmi_breakdown[2]

    

    #Set default value for remaining principal

    df['closing_amount_to_settle'] = df['amount_borrowed']

    

    for index, row in df.iterrows():

        

        #Define opening amount to settle

        if index == 0: #Set default to starting value

            opening_amount_to_settle = row['closing_amount_to_settle']

        else: #Set to closing value from previous period

            opening_amount_to_settle = closing_amount_to_settle

        

        #Define input values for CMI calculation

        rate = row['monthly_rate']

        nper = row['remaining_months']



        #Run CMI calculation and recalculations

        cmi_total = cmi_calculator(rate, nper, opening_amount_to_settle)



        #Define CMI Components

        cmi_interest = opening_amount_to_settle * rate

        cmi_settle = cmi_total - cmi_interest

        cmi_breakdown = (cmi_total, cmi_interest, cmi_settle)

        

        #Update closing_amount_to_settle

        closing_amount_to_settle = opening_amount_to_settle - cmi_settle



        #Update new values to the dataframe

        df.at[index, 'cmi_total'] = cmi_breakdown[0]

        df.at[index, 'cmi_interest'] = cmi_breakdown[1]

        df.at[index, 'cmi_settle'] = cmi_breakdown[2]   

        

        df.at[index, 'closing_amount_to_settle'] =  closing_amount_to_settle

    return df
df_pattern = df_original.copy()



#run recalculator

df_pattern = payment_pattern_calculator(df_pattern)



print('Calculating payment pattern on below starting table')

display(df_original.head())

    

print('Resulting Payment Pattern')

display(df_pattern.head())
df_change = df_original.copy()



#Let's assume there is an interest rate change at n = 2

df_change.at[2:, 'monthly_rate'] = 0.01

df_change = payment_pattern_calculator(df_change)



print('No interest rate change')

display(df_pattern.head())



print('Interest rate change')

display(df_change.head())
print('No Change - Total Interest Paid: {}'.format(round(df_pattern['cmi_interest'].sum(),2)))

print('Interest Change - Total Interest Paid: {}'.format(round(df_change['cmi_interest'].sum(),2)))
# Define a list for trigger events

trigger_flag = [0 for x in list_period]

trigger_flag[:4] = [1, 0, 1, 0]

print(trigger_flag)
#Define DataFrame and change inputs

df_trigger = df_original.copy()

df_trigger['trigger_flag'] = trigger_flag

df_trigger.at[2:, 'monthly_rate'] = 0.01
'''--------------Current Exercise Section begins----------------------------------------'''

'''     In the payment_pattern_recalculator, change the None to correct statements      '''

'''     TIP: If a variable has not been updated, it will keep it's previous value       '''

'''--------------Current Exercise Section ends------------------------------------------'''



def payment_pattern_recalculator(df):



    #Make copy of original df to avoid overwrites

    df = df.copy()



    #Define placeholders for the CMI values

    cmi_breakdown = (0.00, 0.00, 0.00)

    df['cmi_total'] = cmi_breakdown[0]

    df['cmi_interest'] = cmi_breakdown[1]

    df['cmi_settle'] = cmi_breakdown[2]

    

    #Set default value for remaining principal

    df['closing_amount_to_settle'] = df['amount_borrowed']

    

    for index, row in df.iterrows():

        

        #Define opening amount to settle

            #Note that now we don't refer to an opening_amount_to_settle, as logically it's not needed

        if index == 0: #Set default to starting value

            opening_amount_to_settle = row['closing_amount_to_settle']

        else:

            closing_amount_to_settle = closing_amount_to_settle            

        

        '''--------------Current Exercise Section begins----------------------------------------'''

        #Run CMI calculation and recalculations

        if row['trigger_flag'] == 1:

            #Define input values for CMI calculation

            rate = row['monthly_rate']

            nper = row['remaining_months']

            cmi_total = cmi_calculator(rate, nper, opening_amount_to_settle)

        else:

            cmi_total = cmi_total

        '''--------------Current Exercise Section ends------------------------------------------'''



        #Define CMI Components

        cmi_interest = opening_amount_to_settle * rate

        cmi_settle = cmi_total - cmi_interest

        cmi_breakdown = (cmi_total, cmi_interest, cmi_settle)



        #Update closing_amount_to_settle

        closing_amount_to_settle = opening_amount_to_settle - cmi_settle

        

        #Update new values to the dataframe

        df.at[index, 'cmi_total'] = cmi_breakdown[0]

        df.at[index, 'cmi_interest'] = cmi_breakdown[1]

        df.at[index, 'cmi_settle'] = cmi_breakdown[2]   

        

        df.at[index, 'closing_amount_to_settle'] =  closing_amount_to_settle

    return df
print('Trigger Event: Payment Pattern')

df_trigger = payment_pattern_recalculator(df_trigger)

display(df_trigger.head())



print('No Trigger Event: Payment Pattern')

df_no_trigger = df_trigger.copy()

df_no_trigger['trigger_flag'] = 0

df_no_trigger.at[0, 'trigger_flag'] = 1

df_no_trigger = payment_pattern_recalculator(df_no_trigger)

display(df_no_trigger.head())
print('Trigger Event, Total Paid: {}'.format(round(df_trigger['cmi_total'].sum(), 2)))

print('No Trigger Event, Total Paid: {}'.format(round(df_no_trigger['cmi_total'].sum(), 2)))
#configure the inputs

df_missed = df_original.copy()



df_missed['trigger_flag'] = 0

df_missed['missed_flag'] = 0



df_missed.at[3:, 'monthly_rate'] = 0.01

df_missed.at[[0,3], 'trigger_flag'] = 1

df_missed.at[:2, 'missed_flag'] = 1



df_missed.head()
#run recalculator

df_missed = payment_pattern_recalculator(df_missed)

df_missed.head()
'''--------------Current Exercise Section begins----------------------------------------'''

'''     In the payment_pattern_recalculator, change the None to correct statements      '''

'''     TIP: If a variable has not been updated, it will keep it's previous value       '''

'''--------------Current Exercise Section ends------------------------------------------'''



def balance_profile_recalculator(df):



    #Make copy of original df to avoid overwrites

    df = df.copy()



    #Define placeholders for the CMI values

    cmi_breakdown = (0.00, 0.00, 0.00)

    df['cmi_total'] = cmi_breakdown[0]

    df['cmi_interest'] = cmi_breakdown[1]

    df['cmi_settle'] = cmi_breakdown[2]

    

    #Set default value for remaining principal

    df['closing_amount_to_settle'] = df['amount_borrowed']

    

    '''--------------New Section begins----------------------------------------'''

    arrears_breakdown = (0.00, 0.00, 0.00)

    df['arrears_total'] = arrears_breakdown[0]

    df['arrears_interest'] = arrears_breakdown[1]

    df['arrears_settle'] = arrears_breakdown[2]

    '''--------------New Section ends------------------------------------------'''

    

    for index, row in df.iterrows():

        

        #Define opening amount to settle

        if index == 0: #Set default to starting value

            opening_amount_to_settle = row['closing_amount_to_settle']

            '''--------------New Section begins----------------------------------------'''

            arrears_total, arrears_interest, arrears_settle = arrears_breakdown

            '''--------------New Section ends------------------------------------------'''

        else:

            opening_amount_to_settle = closing_amount_to_settle

        

        #Run CMI calculation and recalculations

        if row['trigger_flag'] == 1:

            #Define input values for CMI calculation

            rate = row['monthly_rate']

            nper = row['remaining_months']

            cmi_total = cmi_calculator(rate, nper, opening_amount_to_settle)#Copy/Paste solution from previous exercise

        else:

            cmi_total = cmi_total#Copy/Paste solution from previous exercise

            



        #Define CMI Components

        cmi_interest = opening_amount_to_settle * rate

        cmi_settle = cmi_total - cmi_interest

        cmi_breakdown = (cmi_total, cmi_interest, cmi_settle)



        '''--------------Current Exercise Section begins----------------------------------------'''

        #Update closing_amount_to_settle and Arrears values

        

        closing_amount_to_settle = opening_amount_to_settle - cmi_settle + (row['missed_flag'] * cmi_total)

        

        arrears_total = arrears_total + (row['missed_flag'] * cmi_total)

        arrears_interest = arrears_interest + (row['missed_flag'] * cmi_interest)

        arrears_settle = arrears_settle + (row['missed_flag'] * cmi_settle)

        arrears_breakdown = (arrears_total, arrears_interest, arrears_settle)

        '''--------------Current Exercise Section ends------------------------------------------'''

        

        #Update new values to the dataframe

        df.at[index, 'cmi_total'] = cmi_breakdown[0]

        df.at[index, 'cmi_interest'] = cmi_breakdown[1]

        df.at[index, 'cmi_settle'] = cmi_breakdown[2]   

        

        df.at[index, 'closing_amount_to_settle'] = closing_amount_to_settle

        

        '''--------------New Section begins----------------------------------------'''

        df.at[index, 'arrears_total'] = arrears_breakdown[0]

        df.at[index, 'arrears_interest'] = arrears_breakdown[1]

        df.at[index, 'arrears_settle'] = arrears_breakdown[2] 

        '''--------------New Section ends------------------------------------------'''

        

    return df
#Configure the inputs



df_balance = df_original.copy()



df_balance['trigger_flag'] = 0

df_balance['missed_flag'] = 0



df_balance.at[3:, 'monthly_rate'] = 0.01

df_balance.at[[0,3], 'trigger_flag'] = 1

df_balance.at[:2, 'missed_flag'] = 1
#run the recalculator

df_balance = balance_profile_recalculator(df_missed)

df_balance.head()
'''

Assume a recalculation at period 2

Note that the CMI in this case will not reflect missed payments

This is due to our definition of closing_amount_to_settle

'''

df_missed_example = df_balance.copy()

df_missed_example.at[2, 'trigger_flag'] = 1

df_missed_example = balance_profile_recalculator(df_missed_example)

df_missed_example.head()
df_exercise = df_balance.copy()

df_exercise.at[2, 'trigger_flag'] = 1

df_exercise = balance_profile_recalculator(df_missed_example)

df_exercise.head()