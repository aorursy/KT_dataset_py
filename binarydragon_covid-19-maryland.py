# 2.1 Importing necessary libraries 



import pandas as pd # Data manipulation 

import numpy as np # lin. algebra

import seaborn as sns # Data visualization 

# Set up matplot 

%matplotlib inline

import matplotlib.pyplot as plt # Data graph manipulation 
# 2.2 Loading the data, cutting it down to only include Maryland. 



full_raw_data = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv') # Keep the entire dataset in a variable... in case it is needed.



maryland_raw_data = full_raw_data[full_raw_data.state == "Maryland"]

maryland_raw_data = maryland_raw_data[full_raw_data.county != "Washington"] # Cut out Washington, as D.C is NOT part of Maryland. 
maryland_raw_data.head()
maryland_raw_data.info()
# Total number of entries

total_entries = len(maryland_raw_data.index) # Keep track of the number of total entries. 

print("Total number of entries: %s" % total_entries)



# Total number of missing values from the fips category

total_fips_null = maryland_raw_data['fips'].isna().sum()

print("Total missing values: %s" % total_fips_null)



# Thus... total valid entries

print("Total valid entries: %s" % (total_entries - total_fips_null))



# Let's compile a list of all the rows with missing values, and take a look at it

missing_value_rows = maryland_raw_data[maryland_raw_data.isnull().any(axis=1)]
missing_value_rows.head()
maryland_raw_data = maryland_raw_data.dropna()

# Verify it worked... check that the total length is = (total_entries - total_fips_null)

print("(New length):%s =? (total - null):%s" % (len(maryland_raw_data), total_entries - total_fips_null))

if len(maryland_raw_data) != (total_entries - total_fips_null):

    # An error has occurred! Let's make it clear

    print("Procedural error! Mistake handling missing values, total entries - removed missing values != length of new raw_data array")

else:

    print("Correct! Missing values removed!")
maryland_data = maryland_raw_data.copy()

maryland_data.index = maryland_raw_data.date
maryland_data.head()
zero_case_set = maryland_data[maryland_data.cases == 0]

print("Any rows with zero cases...? :: %s" % len(zero_case_set))

if len(zero_case_set) == 0:

    print("No rows with zero cases")

else:

    print("%s rows with zero cases" % len(zero_case_set))
# Let's get the first date present in the data by checking the first row.

print(maryland_data.iloc[0]['date'])
# Define our date class...

class Date: 

    def __init__(self, date_string): # format for the date_string is "yyyy-mm-dd", dashes included!

        self.raw_date_string = date_string

        self.year = int(date_string[:4])

        self.month = int(date_string[5:7])

        self.day = int(date_string[8:])

        

print(type(maryland_data.iloc[0]['date'])) # Proof that the type is a string...

test_date = Date(maryland_data.iloc[0]['date'])

print([test_date.year, test_date.month, test_date.day]) # Test that our date class is working
# We need to set the x-axis to be our dates and the y-axis to be our case numbers. To make things readable, we'll create a seperate line for each county. 

plt.figure(figsize=(32, 16))

county_names = maryland_data.county.unique() # Get all of the different county names, to plot each county

counter = 0 # Prepare a counter variable, in order to print our progress

for county in county_names: 

    counter += 1 # Update our counter. 1 will be the first value

    print("Processing: %s --> %s / %s" % (county, counter, len(county_names))) # Print progress

    temp_data = maryland_data[maryland_data.county == county] # get a subset of the data that ONLY INCLUDES THE CURRENT COUNTY

    graph = sns.lineplot(x='date', y='cases', data=temp_data, err_style="bars", label=county, legend=False) # plot the data for that county

    

plt.legend(loc='upper left') # place the legend, update it

graph = graph.set_xticklabels(temp_data['date'], rotation=90) # make sure the date values are rotated
#  Calculate percentages of population infected

population_values_exact = [75300, 556100, 842600, 100450, 40300, 197400,

                          130350, 177200, 36300, 287900, 31600, 276500, 

                          312900, 22200, 1075000, 921900, 55650, 130100,

                          28300, 40050, 170950, 107450, 56250] # Create a list of all the population values (predicted) for 2020 Maryland, by county (alphabetical)

county_names_alph = sorted(county_names) # Sort the county names alphabetically 

population_numbers_predicted_2020 = {county_names_alph[i] : population_values_exact[i] for i in range(len(population_values_exact))} # Create a dictionary...

#  Form: {'County name' : Population value...}



#  For each county, get the temporary data for ONLY THAT COUNTY, and print/store the percentages of the population infected

for county in county_names: 

    temp_data = maryland_data[maryland_data.county == county]

    print("%s --> Percentage of population infected is : %s / %s ... or %s%%\n" % (county, temp_data.iloc[len(temp_data) - 1]['cases'], 

                                                                                population_numbers_predicted_2020[county], 

                                                                                (temp_data.iloc[len(temp_data) - 1]['cases'] / 

                                                                                population_numbers_predicted_2020[county]) * 100))
#  Calculate instaneous derivatives for each county's cases on different dates. 



plt.figure(figsize=(32, 16)) # Prepare our plot... size it so it's readable





from numpy import diff # Import a function to help us calculate the derivative 

dx = 1.0 # The change on the x axis will be ONE DAY (the meaning of 1.0). Thus, dy/dx will be dy change for each DAY. 

counter = 0 # Prepare a counter variable, in order to print our progress

for county in county_names: 

    counter += 1 # Update our counter. 1 will be the first value

    print("Processing: %s --> %s / %s" % (county, counter, len(county_names))) # Print progress

    temp_data = maryland_data[maryland_data.county == county] # get a subset of the data that ONLY INCLUDES THE CURRENT COUNTY, this will be our y.

    y = temp_data['cases'].tolist() # Convert our cases to a list so the numpy.diff function will work

    dy = diff(y) / dx # REMEMBER, when we're calculating derivates simply by taking differences, the last value in the data will NOT BE INCLUDED

    #  This is handled below... in the graphing function. y = derivative of the cases, and the data includes all values but the last. 

    graph = sns.lineplot(x='date', y=dy, data=temp_data[:len(temp_data.index) - 1], err_style="bars", label=county, legend=False) # plot the data for that county



plt.legend(loc='upper left') # place the legend, update it

graph = graph.set_xticklabels(temp_data['date'], rotation=90) # make sure the date values are rotated
#### IMPORTANT !!! #####

derivative_smoothed_graphs = [] # List to hold each derivative smoothed graph. 

smoothed_d_graph = None # Variable to hold each respective graph

#  Method 1, average all 7 days' values.



plt.figure(figsize=(32, 16)) # Prepare our plot... size it so it's readable



### DECLARE VARIABLES ###



counter = 0 # reset our counter (for printing progress) to zero. 

averaged_derivatives_cases_dataframes = [] # create a list of dataframes for the derivative data, averaged, from each county.

averaged_derivatives_cases_dataframe = None # create a dataframe for ALL the derivative data, averaged, to be sorted by date (not county)



### ###



### PROCESS DERIVATIVES AND AVERAGES### 

for county in county_names: # For each county...

    counter += 1 # Update our counter. 1 will be the first value

    

    method_counter = 0 # Variable to help us keep track of our every 7 days averaging requirement 

    dy_averaged = [] # list of derivatives, averaged for 7 days.

    averaged_dates = [] # list of every 7 days. 

    continous_average = 0 # The average of the past 7 days (with the exception of the FIRST entry)

    average_denominator = 1 # The denominator for the average calculation. Set to 1 for the first entry

    

    print("Processing Derivative: %s --> %s / %s" % (county, counter, len(county_names))) # Print progress



    

    temp_data = maryland_data[maryland_data.county == county] # get a subset of the data that ONLY INCLUDES THE CURRENT COUNTY, this will be our y.

    y = temp_data['cases'].tolist() # Convert our cases to a list so the numpy.diff function will work

    dy = diff(y) / dx # REMEMBER, when we're calculating derivates simply by taking differences, the last value in the data will NOT BE INCLUDED

    

    for i in range(len(dy)): # for each day and each derivative

        if method_counter == 0: # if we're on the first value, or the start of a week...

            continous_average = sum(dy[:(i+1)]) / (i + 1) # calculate the average

            dy_averaged.append(continous_average) # append the current average 

            averaged_dates.append(temp_data.iloc[i]['date']) # append the current date so that the average is tracked

            method_counter += 1 # Next day!

        elif method_counter == 6: # Last day of the week, reset to zero so the first day of the NEXT week is recorded

            method_counter = 0 

        else: # Next day... next day... etc.

            method_counter += 1

            

    

    df = pd.DataFrame({'date' : averaged_dates, 'averaged derivatives of cases' : dy_averaged, 'county' : [

        county for c in range(len(averaged_dates))

    ]}) # Create a dataframe to hold all our dates, and all the averaged derivatives for EACH couonty

    df.index = df.date # set the index to date

    averaged_derivatives_cases_dataframes.append(df) # Append it to our list



### ###



### MERGE DATAFRAMES ###

    

print("Progress: Merging county derivative data")

averaged_derivatives_cases_dataframe = pd.concat(averaged_derivatives_cases_dataframes, axis=0) # conc. all the different counties' data. 

averaged_derivatives_cases_dataframe.index = averaged_derivatives_cases_dataframe['date'] # set the index (be sure the index is correct)



averaged_derivatives_cases_dataframe['date'] = pd.to_datetime(averaged_derivatives_cases_dataframe.date) # convert the date to datetime, so we can sort

averaged_derivatives_cases_dataframe.sort_index() # sort based on the datetime index



### ###



### GRAPH EVERYTHING! ###



counter = 0

for county in county_names:

    counter += 1 # Update our counter. 1 will be the first value

    

    print("Processing: %s --> %s / %s" % (county, counter, len(county_names))) # Print progress

    temp_data = averaged_derivatives_cases_dataframe[averaged_derivatives_cases_dataframe.county == county] # get a subset of the data that 

        #  ONLY INCLUDES THE CURRENT COUNTY

        

    smoothed_d_graph = sns.lineplot(x='date', y='averaged derivatives of cases', data=temp_data, err_style="bars", label=county, legend='full') # plot the data for that 

        #  county

### ###



derivative_smoothed_graphs.append(smoothed_d_graph) # Add the final version of the graph to the list for plotting later

plt.legend(loc='upper left') # place the legend, update it

#  Method 2, pick the median value for all 7 days of the week.



from statistics import median

plt.figure(figsize=(32, 16)) # Prepare our plot... size it so it's readable



### DECLARE VARIABLES ###



counter = 0 # reset our counter (for printing progress) to zero. 

median_derivatives_cases_dataframes = [] # create a list of dataframes for the derivative data, median, from each county.

median_derivatives_cases_dataframe = None # create a dataframe for ALL the derivative data, median, to be sorted by date (not county)



### ###



### PROCESS DERIVATIVES AND MEDIAN ### 

for county in county_names: # For each county...

    counter += 1 # Update our counter. 1 will be the first value

    

    method_counter = 0 # Variable to help us keep track of our every 7 days median requirement 

    dy_median = [] # list of derivatives, median for 7 days.

    median_dates = [] # list of every 7 days. 

    continous_median = 0 # The median of the past 7 days (with the exception of the FIRST entry)

    

    print("Processing Derivative: %s --> %s / %s" % (county, counter, len(county_names))) # Print progress



    

    temp_data = maryland_data[maryland_data.county == county] # get a subset of the data that ONLY INCLUDES THE CURRENT COUNTY, this will be our y.

    y = temp_data['cases'].tolist() # Convert our cases to a list so the numpy.diff function will work

    dy = diff(y) / dx # REMEMBER, when we're calculating derivates simply by taking differences, the last value in the data will NOT BE INCLUDED

    

    for i in range(len(dy)): # for each day and each derivative

        if method_counter == 0: # if we're on the first value, or the start of a week...

            continous_median = median(dy[:(i+1)]) # calculate the median

            dy_median.append(continous_median) # append the current median 

            median_dates.append(temp_data.iloc[i]['date']) # append the current date so that the median is tracked

            method_counter += 1 # Next day!

        elif method_counter == 6: # Last day of the week, reset to zero so the first day of the NEXT week is recorded

            method_counter = 0 

        else: # Next day... next day... etc.

            method_counter += 1

            

    

    df = pd.DataFrame({'date' : median_dates, 'median derivatives of cases' : dy_median, 'county' : [

        county for c in range(len(median_dates))

    ]}) # Create a dataframe to hold all our dates, and all the median derivatives for EACH couonty

    df.index = df.date # set the index to date

    median_derivatives_cases_dataframes.append(df) # Append it to our list



### ###



### MERGE DATAFRAMES ###

    

print("Progress: Merging county derivative data")

median_derivatives_cases_dataframe = pd.concat(median_derivatives_cases_dataframes, axis=0) # conc. all the different counties' data. 

median_derivatives_cases_dataframe.index = median_derivatives_cases_dataframe['date'] # set the index (be sure the index is correct)



median_derivatives_cases_dataframe['date'] = pd.to_datetime(median_derivatives_cases_dataframe.date) # convert the date to datetime, so we can sort

median_derivatives_cases_dataframe.sort_index() # sort based on the datetime index



### ###



### GRAPH EVERYTHING! ###



counter = 0

for county in county_names:

    counter += 1 # Update our counter. 1 will be the first value

    

    print("Processing: %s --> %s / %s" % (county, counter, len(county_names))) # Print progress

    temp_data = median_derivatives_cases_dataframe[median_derivatives_cases_dataframe.county == county] # get a subset of the data that 

        #  ONLY INCLUDES THE CURRENT COUNTY

        

    smoothed_d_graph = sns.lineplot(x='date', y='median derivatives of cases', data=temp_data, err_style="bars", label=county, legend='full') # plot the data for that 

        #  county

### ###



derivative_smoothed_graphs.append(smoothed_d_graph) # Add the final version of the graph to the list in case it's needed

plt.legend(loc='upper left') # place the legend, update it

fig_avg, ax_avg = plt.subplots()

fig_avg = plt.gcf()

fig_avg.set_size_inches(32, 16)

# Method 1

for county in county_names:    

    temp_data = median_derivatives_cases_dataframe[median_derivatives_cases_dataframe.county == county] # get a subset of the data that 

        #  ONLY INCLUDES THE CURRENT COUNTY

    temp_graph = sns.lineplot(x='date', y='median derivatives of cases', data=temp_data, err_style="bars", 

                              ax=ax_avg, 

                              label=county, legend='full').set_title("Averaged Derivatives for Cases (Every 7 days)")# plot the data for that county

        

##

fig_med, ax_med = plt.subplots()

fig_med = plt.gcf()

fig_med.set_size_inches(32, 16)## 

# Method 2

for county in county_names:    

    temp_data = averaged_derivatives_cases_dataframe[averaged_derivatives_cases_dataframe.county == county] # get a subset of the data that 

        #  ONLY INCLUDES THE CURRENT COUNTY

    temp_graph = sns.lineplot(x='date', y='averaged derivatives of cases', data=temp_data, err_style="bars", 

                              ax=ax_med, 

                              label=county, legend='full').set_title("Median Derivatives for Cases (Every 7 days)") # plot the data for that county

plt.legend(loc='upper left') # place the legend, update it

fig = plt.figure(figsize=(15, 15))

cases_ax = fig.add_subplot(1, 2, 1)

deaths_ax = fig.add_subplot(1, 2, 2)



final_date = maryland_data.iloc[len(maryland_data) - 1]['date']

print("Final date in the dataset is: %s" % final_date)



temp_data = maryland_data[maryland_data.date == final_date]

cases_data = {'Counties' : temp_data['county'], 'Cases' : temp_data['cases']}

deaths_data = {'Counties' : temp_data['county'], 'Deaths' : temp_data['deaths']}



cases_df = pd.DataFrame(cases_data)

cases_df.index = cases_df.Counties

del cases_df['Counties']

deaths_df = pd.DataFrame(deaths_data)

deaths_df.index = deaths_df.Counties

del deaths_df['Counties']



ax_1 = sns.heatmap(data=cases_df, ax=cases_ax)

ax_2 = sns.heatmap(data=deaths_df, ax=deaths_ax)

plt.tight_layout()