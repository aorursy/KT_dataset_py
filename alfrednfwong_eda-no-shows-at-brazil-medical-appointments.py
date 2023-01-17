import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import datetime

import collections

import statsmodels.stats.proportion

%matplotlib inline
def bold(text):

    '''Returns the input-ted text in bold'''

    return '\033[1m' + text + '\033[0;0m'
# Load the data from the CSV and print some basic info and the 

# first entry as a sample



appointments_df = pd.read_csv('../input/KaggleV2-May-2016.csv')



print(appointments_df.info())

print(appointments_df.describe())

print(appointments_df.iloc[0])
appointments_df.columns = [

    'patient_ID', 'appointment_ID', 'gender', 'scheduled_day',

    'appointment_day', 'age', 'neighbourhood', 'scholarship',

    'hypertension', 'diabetes', 'alcoholism', 'handicap', 'SMS_received',

    'no_show'

    ]
appointments_df.scheduled_day = pd.to_datetime(appointments_df.scheduled_day)

appointments_df['scheduled_hour'] = (

    appointments_df.scheduled_day.apply(lambda x : x.hour)

    )
def get_day(datetime_object):

    '''This function returns only the date part of a datetime'''

    return datetime_object.date()



# Parse the date from string

appointments_df.appointment_day = pd.to_datetime(appointments_df.appointment_day)

# Get lead_days column in timedelta64[ns] format

appointments_df['lead_days'] = (

    appointments_df.appointment_day.apply(get_day)

    - appointments_df.scheduled_day.apply(get_day)

    )

# Change the datatype into integer

appointments_df.lead_days = (

    (appointments_df.lead_days.astype('timedelta64[D]')).astype(int)

    )

# Create the appointment day-of-week column

appointments_df['appointment_DOW'] = (

    appointments_df.appointment_day.dt.dayofweek

    )
# Create lead_days_category column

lead_days_labels = pd.Series([

    'A: Same day',

    'B: 1-2 days',

    'C: 3-7 days',

    'D: 8-31 days',

    'E: 32+ days'

    ])

appointments_df['lead_days_category'] = pd.cut(

    appointments_df.lead_days, bins = [-1, 0, 2, 7, 31, 999],

    labels = lead_days_labels,

    )
# To make sure decimals get displayed

pd.set_option('display.float_format', '{:9f}'.format)

print(

    bold('Number of unique patient IDs:'),

    len(appointments_df.patient_ID.unique())

    )

print(' ')

print(bold('Patient IDs with decimal places: '))

print(

    appointments_df[~ appointments_df.patient_ID

    .apply(lambda x: x.is_integer())]

    )
# Take a look at the patients with the most number of appointments

# in our dataset.

# To trim the 0s in the IDs

pd.set_option('display.float_format', '{:.0f}'.format) 

print(bold('Patients with the most appointments: '))

print(appointments_df.patient_ID.value_counts().iloc[0:10])

pd.set_option('display.float_format', '{:4f}'.format) 
len(appointments_df.appointment_ID.unique())
appointments_df.set_index('appointment_ID', inplace = True)
print(bold('Counts by gender: '))

appointments_df.gender.value_counts()
appointments_df['is_female'] = (appointments_df.gender == 'F')
pd.options.display.max_rows = 10

print(bold('Counts by age: '))

print(appointments_df.age.value_counts().sort_index())

print(appointments_df.age.value_counts().sort_values(ascending = False).head())
appointments_df['age_group'] = (

    appointments_df.age.apply(lambda x: min(int(x / 10) , 9))

    )

appointments_df.age_group.value_counts().sort_index()
# number of unique neighbourhood values

len(appointments_df.neighbourhood.unique())
columns_to_change = [

    'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'SMS_received'

    ]

for column in columns_to_change:

    appointments_df[column] = (appointments_df[column] == 1)
appointments_df['is_handicapped'] = (appointments_df.handicap > 0)
boolean_replacement = {'Yes': True, 'No': False}

appointments_df.no_show.replace(boolean_replacement, inplace = True)
# create a new column no_show_last_time that takes the value of no_show in

# the previous appointment of the same patient

appointments_df = appointments_df.sort_values(

    by = ['appointment_day', 'scheduled_day'], axis = 0

    )

appointments_df['no_show_last_time'] = (

    appointments_df.groupby('patient_ID')['no_show'].apply(lambda x : x.shift(1))

    )
def check_patient_status_consistency(column_to_check):

    '''

    Takes in a column (meant to be grouped by patient ID), and returns

    False if not all the values are the same

    '''

    if len(column_to_check.unique()) > 1:

        return False

    return True



def check_age_consistency(age_series):

    '''

    Takes in a series of (age) values and returns False if any two of the

    values differ by more than 1

    '''

    if age_series.max() - age_series.min() > 1:

        return False

    return True



def print_problematic_patients(column_name, problematic_patient_ID_list):

    '''

    Takes a column name and the list of patient IDs with inconsistencies 

    in that column (empty list if none), and prints the entries with those

    patient_IDs

    '''

    if problematic_patient_ID_list.size > 0:

        print(bold('For the column ' + column + ':'))

        for patient in problematic_patient_ID_list:

            print(bold('    Patient ID:', str(int(patient))))

            print(appointments_df[appointments_df.patient_ID.isin

                                  (problematic_patient_ID_list)])

    else:

        print(bold('No inconsistency found in the column ' + column))



# For each of the columns we check, print the entries that have 

# inconsistent values in those columns

grouped_df = appointments_df.groupby('patient_ID')

for column in (

    'gender', 'hypertension', 'diabetes', 'alcoholism', 'scholarship', 'handicap'

    ):

    ID_check_results = grouped_df[column].apply(check_patient_status_consistency)

    problematic_patient_ID_list = (

        ID_check_results[ID_check_results == False].index.values

        )

    print_problematic_patients(column, problematic_patient_ID_list)



# Ditto for age, but with a different test function

for column in ('age',):

    ID_check_results = grouped_df[column].apply(check_age_consistency)

    problematic_patient_ID_list = (

        ID_check_results[ID_check_results == False].index.values

        )

    print_problematic_patients(column, problematic_patient_ID_list)



del grouped_df
# See if any entries have the appointment_day earlier than the scheduled_day

print(appointments_df[appointments_df.lead_days < 0])
# Dropping the rows where the scheduled_day is later than the appointment_day

appointments_df = (

    appointments_df.drop(appointments_df[appointments_df.lead_days < 0].index)

    )
def compare_by_column(df, column_name):

    '''

    Returns comparison_df and the overall no-show rate for a single column.

    

    This function takes in a datraframe and a column name within it, 

    and returns another dataframe with the frequencies of all the

    possible values in that column by show up status, as well as the

    percentage of no shows, for use in the plot_no_show_rates(comparison_df)

    function. Also returns the overall no-show rate of the df.

    The column can have more than 2 values.

    '''

    comparison_df = pd.DataFrame()

    comparison_df['no_show'] = (

        df[df.no_show == True][column_name].value_counts()

        )

    comparison_df['show_up'] = (

        df[df.no_show == False][column_name].value_counts()

        )

    # In case some for column_name values, there isn't a single True or

    # a single False in no_show, they'll come up as NaN in the 

    # comparison_df. We fill those with zeros.

    comparison_df = comparison_df.fillna(0)

    comparison_df['sample_size'] = comparison_df.no_show + comparison_df.show_up

    comparison_df['no_show_rate'] = (

        comparison_df.no_show / (comparison_df.no_show + comparison_df.show_up)

        )

    comparison_df.sort_index(inplace = True)

    comparison_df.name = column_name

    overall_no_show_rate = (

        comparison_df.no_show.sum() / comparison_df.sample_size.sum()

        )

    return (comparison_df, overall_no_show_rate)



def compare_by_multiple_booleans(df, column_list):

    '''

    Returns a comparison_df for a list of boolean columns and the overall

    no-show rate.

    

    Takes a dataframe and a list of column names, returns a comparison_df for

    use in the plot_no_show_rates(comparison_df) function, together with an

    overall no-show rate of the df. The columns have to be boolean.

    

    '''

    comparison_df = (pd.DataFrame(columns = [

        'no_show', 'show_up', 'sample_size', 'no_show_rate'

        ]))

    # loop through the columns (of the input df), do the no-show counts for 

    # each of the column values True and False

    # Then get the sample_size and no_show_rate

    # Then put those numbers as a row in the comparison_df for output

    for column in column_list:

        no_show_count = len(df[df[column] & df['no_show']])

        show_up_count = len(df[df[column] & (~ df['no_show'])])

        sample_size = no_show_count + show_up_count

        no_show_rate = no_show_count / (no_show_count + show_up_count)

        comparison_df.loc[column] = ([

            no_show_count, show_up_count, sample_size, no_show_rate

            ])

    comparison_df[['no_show', 'show_up', 'sample_size']] = (

        comparison_df[['no_show', 'show_up', 'sample_size']].astype(int)

        )

    overall_no_show_rate = len(df[df.no_show]) / len(df)

    comparison_df.name = 'patient attributes'

    return comparison_df, overall_no_show_rate
def plot_no_show_rates(

    comparison_df, overall_no_show_rate, graph_width, graph_height,

    title_suffix = None

    ):

    '''

    Plot the no-show rates in a horizontal bars.

    

    Takes in a comparison_df and the overall_no_show_rate. 

    graph_width and graph_height are the width and height of the graph. 

    title_suffix is an optional string appended to the end of the default

    graph title

    '''

    fig = plt.figure(figsize=(graph_width, graph_height))

    ax = fig.add_subplot(111)

    height = 0.4

    bins = np.arange(len(comparison_df.index))

    bars = ax.barh(

        bins, comparison_df.no_show_rate.values, height, align = 'center',

        color = 'red'

        )

    

    # Add a vertical line indicating the overall no-show rates

    y1 = - (height / 2)

    y2 = len(bins) - 1 + (height / 2)

    x = overall_no_show_rate

    overall_line = ax.plot(

        (x, x), (y1, y2), color = 'yellow', label = 'Overall no-show rate'

        )

    ax.legend(handles = overall_line)

    

    # Labels and title. Title suffix is appended here if not none.

    ax.set_xlabel('No-show rates')

    ax.set_ylabel(comparison_df.name + ' (n=sample size)')

    ax.set_title('No-show rates by ' + comparison_df.name)

    if title_suffix is None:

        ax.set_title(''.join(['No-show rates by ', comparison_df.name]))

    else:

        ax.set_title(''.join(['No-show rates by ', comparison_df.name,

                             title_suffix]))

    ax.set_yticks(bins)

    

    # Create yticklabels. Each label contains the index of the row in the 

    # comparison_df it represents, as well as the sample_size in that row

    labels = []

    for row_number in bins:

        labels.append(

            str(comparison_df.index[row_number]) + ' (n='

            + str(comparison_df.sample_size.iloc[row_number]) + ')'

            )

    ax.set_yticklabels(labels, ha = 'right')



    # label the values of each bar

    for bar in bars:

        width = bar.get_width()

        ax.text(

            width * 1.02, bar.get_y() + (height / 2.), round(width, 3),

            ha = 'left', va = 'center'

            )

    plt.show()

    return
def proportions_test(df1, df2, success_column):

    '''

    Z-test for difference in proportions of success_column values

    between samples df1 and df2.

    

    Returns only the p-value

    '''

    success_count1 = len(df1[df1[success_column] == True])

    success_count2 = len(df2[df2[success_column] == True])

    num_of_obs1 = len(df1)

    num_of_obs2 = len(df2)

    results = statsmodels.stats.proportion.proportions_ztest(

        [success_count1, success_count2], [num_of_obs1, num_of_obs2],

        alternative = 'two-sided'

        )

    return results[1]
# Plot a pie chart for appointment counts by lead day categories

pie_data = appointments_df.lead_days_category.value_counts().sort_index()

label_list = pie_data.index

sizes = pie_data

explode_list = (0.1, 0, 0, 0, 0) 

fig1, ax1 = plt.subplots(figsize=(6, 6))

ax1.pie(

    sizes, explode = explode_list, labels = label_list, autopct =' %1.1f%%',

    pctdistance = 0.6, labeldistance = 1.05, shadow = True, startangle = 0,

    )

ax1.axis('equal') 

ax1.set_title('Appointment counts by lead_days_category')

plt.show()
type(compare_by_column(appointments_df, 'lead_days_category'))
comparison_df, overall_no_show_rate = (

    compare_by_column(appointments_df, 'lead_days_category')

    )

plot_no_show_rates(comparison_df, overall_no_show_rate, 6, 6)
# Create a line plot of SMS_received by lead_days, up to lead_days == 30

# then print the number for the rest of the data so that they don't obscure

# the focus.

x = appointments_df.groupby('lead_days')['SMS_received'].mean()[:31].index

y = appointments_df.groupby('lead_days')['SMS_received'].mean()[:31].values

plt.plot(x,y, 'bo')

plt.plot(x,y)

plt.title('Proportion of SMS_received by lead_days')

plt.xlabel('lead_days')

plt.ylabel('Proportion of SMS_received')

plt.show()



print(bold('Proportion of SMS_received for lead_days > 30'))

print(appointments_df[appointments_df.lead_days > 30]['SMS_received'].mean())
# Call the compare and plot functions on the SMS_received column, with 

# a trimmed dataframe that has only the entries with lead_days > 3, to 

# get around the problem that same-day appointments have a much lower

# no-show rate, and SMS messages are never sent for same-day appointments.

lead_days_3plus_df = appointments_df[appointments_df.lead_days > 3]

comparison_df, overall_no_show_rate = (

    compare_by_column(lead_days_3plus_df, 'SMS_received')

    )

plot_no_show_rates(

    comparison_df, overall_no_show_rate, 5, 5, ' when lead_days > 3'

    )
print(bold('p-value: '))

print(proportions_test(

    lead_days_3plus_df[lead_days_3plus_df.SMS_received],

    lead_days_3plus_df[~ lead_days_3plus_df.SMS_received], 'no_show'

    ))
comparison_df, overall_no_show_rate = (

    compare_by_column(appointments_df, 'age_group')

    )

plot_no_show_rates(comparison_df, overall_no_show_rate, 5, 7)
# Create a bar chart for all booleans

column_list = [

    'is_female', 'scholarship', 'hypertension', 'diabetes', 'alcoholism',

    'is_handicapped'

    ]

comparison_df, overall_no_show_rate  = (

    compare_by_multiple_booleans(appointments_df, column_list)

    )

plot_no_show_rates(comparison_df, overall_no_show_rate, 5, 7)
def booleans_by_age_group(column, grid_number):

    '''

    Takes a column name and subplot grid number, and plots the column

    by age_group

    '''

    ax = fig.add_subplot(grid_number)

    red_x = appointments_df[appointments_df[column]].age_group

    blue_x = appointments_df[~ appointments_df[column]].age_group

    ax.hist(

        [red_x, blue_x], normed = True, color = ['red', 'blue'],

        bins = np.arange(12), align = 'left'

        )

    ax.set_xticks(np.arange(1,11))

    ax.set_ylabel('Normalized frequency')

    ax.set_xlabel(('Age group'))

    ax.set_title('Age group distributions by ' + column)

    ax.legend((column + ' = True', column + ' = False'))

    return



fig = plt.figure(figsize=(20,20))

booleans_by_age_group('is_female', 321)

booleans_by_age_group('scholarship', 322)

booleans_by_age_group('hypertension', 323) 

booleans_by_age_group('diabetes', 324)

booleans_by_age_group('alcoholism', 325)

booleans_by_age_group('is_handicapped', 326)

plt.show()
comparison_df, overall_no_show_rate = (

    compare_by_column(appointments_df, 'no_show_last_time')

    )

plot_no_show_rates(comparison_df, overall_no_show_rate, 5, 5)
print(bold('p-value: '))

print(proportions_test(

    appointments_df[appointments_df.no_show_last_time == True],

    appointments_df[appointments_df.no_show_last_time == False], 

    'no_show'

    ))
# Plot no-show proportions by scheduled_hour, only for rows where

# lead_days > 0

non_same_day_df = appointments_df[appointments_df.lead_days > 0]

comparison_df, overall_no_show_rate = (

    compare_by_column(non_same_day_df, 'scheduled_hour')

    )

plot_no_show_rates(

    comparison_df, overall_no_show_rate, 6, 10, 

    title_suffix = ', excluding same-day appointments'

    )
# Create a comparison_df with the neighbourhood column

neighbourhood_df = compare_by_column(appointments_df, 'neighbourhood')[0]

# Take away rows with sample_size less than 100 to get rid of possible 

# extreme values that could be caused by small sample sizes, then sort

# to make the graph look better.

neighbourhood_df = (

    neighbourhood_df[neighbourhood_df.sample_size >= 100]

    .sort_values('no_show_rate')

    )

neighbourhood_df.sample_size = neighbourhood_df.sample_size.astype(int)

neighbourhood_df.name = 'neighbourhood'

# Compute the overall rate using the new trimmed df because if we trim away

# the neighbourhoods with too few samples, we cant include those samples in 

# the overall rate.

overall_no_show_rate = (

    neighbourhood_df.no_show.sum() / neighbourhood_df.sample_size.sum()

    )

plot_no_show_rates(

    neighbourhood_df, overall_no_show_rate, 8, 16,

    ', in neighbourhoods with 100 appointments or more'

    )
plt.hist(neighbourhood_df.no_show_rate)

plt.yticks(range(0,20))

plt.title('Frequencies of neighbourhoods in no-show rate bins')

plt.ylabel('Number of neighbourhoods')

plt.xlabel('No-show rate')

plt.show()