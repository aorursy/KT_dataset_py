import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from datetime import datetime



# print(os.listdir("../input"))
## STEP

# Create list of date_range



date_range = pd.date_range(start='01/01/2018', end='31/12/2020')

print(len(date_range))

date_range
## STEP

# Generate a dataset for two countries 'France' and 'England'



def add_country_record(date, country, value):

    dataset['Date'].append(date)

    dataset['Country'].append(country)

    dataset['Value'].append(value)



def is_record_duplicate(date, country):

    return True if date in dataset['Date'] and country in dataset['Country'] else False



dataset = {

    'Date': [],

    'Country': [],

    'Value': []

}



france_range = range(0, 150)

england_range = range(0, 230)



# Generating range for 'France'

for i in france_range:

    random_index = np.random.randint(0, 91)

    random_value = np.random.randint(5, 100)

    date = date_range[random_index]

    country = 'France'

    

    if not is_record_duplicate(date, country):

        add_country_record(date, country, random_value)



# Generating range for 'England'

for i in england_range:

    random_index = np.random.randint(0, 91)

    random_value = np.random.randint(5, 100)

    date = date_range[random_index]

    country = 'England'

    

    if not is_record_duplicate(date, country):

        add_country_record(date, country, random_value)

        

dataset
## STEP

# Create dataframe from the dataset and sort it by the 'Date'



df = pd.DataFrame(dataset)

df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', inplace=True)

df.set_index('Date', inplace=True)

df.info()
df.tail()
## STEP

# Calculate the required date range



# Select the date you want from the table, and assign it to the variable

date = '2018-03-01'



# Set the required range of days withint which you want the search

days_ago_start = 20

days_ago_end = 3



selected_date = pd.to_datetime(date)

start_date = selected_date - pd.Timedelta(days_ago_start, unit='D')

end_date = selected_date - pd.Timedelta(days_ago_end, unit='D')





if start_date > end_date:

    raise Exception("days_ago_start '{}' should be greater than days_ago_end '{}'".format(start_date, end_date))
## STEP

# Run the required calculation for this task



# Find all records for 'France' country and date is between start_date and end_date

result = df[(df['Country'] == 'France') & (df.index >= start_date) & (df.index <= end_date)].reset_index()



# Sum all records of 'Value' column

result_sum = result['Value'].sum()



# Print the final result

print("#### Records in range '{}' and '{}' ####".format(start_date.date(), end_date.date()))

print(result)

print()

print("Sum of all values: {}".format(result_sum))