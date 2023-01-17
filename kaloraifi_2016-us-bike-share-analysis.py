## import all necessary packages and functions.
import csv # read and write csv files
from datetime import datetime # operations to parse dates
from pprint import pprint # use to print data structures like dictionaries in
                          # a nicer way than the base print function.
def print_first_point(filename):
    """
    This function prints and returns the first data point (second row) from
    a csv file that includes a header row.
    """
    # print city name for reference
    city = filename.split('-')[0].split('/')[-1]
    print('\nCity: {}'.format(city))
    
    with open(filename, 'r') as f_in:
        ## TODO: Use the csv library to set up a DictReader object. ##
        ## see https://docs.python.org/3/library/csv.html           ##
        trip_reader = csv.DictReader(f_in)
        
        ## TODO: Use a function on the DictReader object to read the     ##
        ## first trip from the data file and store it in a variable.     ##
        ## see https://docs.python.org/3/library/csv.html#reader-objects ##
        first_trip = next(trip_reader)
        
        ## TODO: Use the pprint library to print the first trip. ##
        ## see https://docs.python.org/3/library/pprint.html     ##
        pprint(first_trip)
        
    # output city name and first trip for later testing
    return (city, first_trip)
def duration_in_mins(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the trip duration in units of minutes.
    
    Remember that Washington is in terms of milliseconds while Chicago and NYC
    are in terms of seconds. 
    
    HINT: The csv module reads in all of the data as strings, including numeric
    values. You will need a function to convert the strings into an appropriate
    numeric type when making your transformations.
    see https://docs.python.org/3/library/functions.html
    """
    duration = 0
    if city == "Washington":
        duration = float(datum['Duration (ms)']) / 60000
    else:
        duration = float(datum['tripduration']) / 60
    
    return duration
def time_of_trip(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the month, hour, and day of the week in
    which the trip was made.
    
    Remember that NYC includes seconds, while Washington and Chicago do not.
    
    HINT: You should use the datetime module to parse the original date
    strings into a format that is useful for extracting the desired information.
    see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    """
    month = ""
    hour = ""
    day_of_week = ""
    
    if city == "NYC":
        nyc_datetime = datetime.strptime(datum['starttime'], "%m/%d/%Y %H:%M:%S")
        
        month = int(nyc_datetime.strftime("%-m"))
        hour = int(nyc_datetime.strftime("%-H"))
        day_of_week = nyc_datetime.strftime("%A")
        
    elif city == "Chicago":
        chicago_datetime = datetime.strptime(datum['starttime'], "%m/%d/%Y %H:%M")
        
        month = int(chicago_datetime.strftime("%-m"))
        hour = int(chicago_datetime.strftime("%-H"))
        day_of_week = chicago_datetime.strftime("%A")
        
    elif city == "Washington":
        washington_datetime = datetime.strptime(datum['Start date'], "%m/%d/%Y %H:%M")
        
        month = int(washington_datetime.strftime("%-m"))
        hour = int(washington_datetime.strftime("%-H"))
        day_of_week = washington_datetime.strftime("%A")
    
    return (month, hour, day_of_week)
def type_of_user(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the type of system user that made the
    trip.
    
    Remember that Washington has different category names compared to Chicago
    and NYC. 
    """
    user_type = ""
    if city == "Washington":
        user_type = datum['Member Type']
        if user_type == 'Registered':
            user_type = 'Subscriber'
        else:
            user_type = 'Customer'
    else:
        user_type = datum['usertype']
        
    return user_type
def condense_data(in_file, out_file, city):
    """
    This function takes full data from the specified input file
    and writes the condensed data to a specified output file. The city
    argument determines how the input file will be parsed.
    
    HINT: See the cell below to see how the arguments are structured!
    """
    
    with open(out_file, 'w') as f_out, open(in_file, 'r') as f_in:
        # set up csv DictWriter object - writer requires column names for the
        # first row as the "fieldnames" argument
        out_colnames = ['duration', 'month', 'hour', 'day_of_week', 'user_type']        
        trip_writer = csv.DictWriter(f_out, fieldnames = out_colnames)
        trip_writer.writeheader()
        
        ## TODO: set up csv DictReader object ##
        trip_reader = csv.DictReader(f_in)
        # collect data from and process each row
        for row in trip_reader:
            # set up a dictionary to hold the values for the cleaned and trimmed
            # data point
            new_point = {}
            ## TODO: use the helper functions to get the cleaned data from  ##
            ## the original data dictionaries.                              ##
            ## Note that the keys for the new_point dictionary should match ##
            ## the column names set in the DictWriter object above.         ##
            new_point['duration'] = duration_in_mins(row, city)
            new_point['month'] = time_of_trip(row, city)[0]
            new_point['hour'] = time_of_trip(row, city)[1]
            new_point['day_of_week'] = time_of_trip(row, city)[2]
            new_point['user_type'] = type_of_user(row, city)

            ## TODO: write the processed information to the output file.     ##
            ## see https://docs.python.org/3/library/csv.html#writer-objects ##
            trip_writer.writerow(new_point)
# Run this cell to check your work
city_info = {'Washington': {'in_file': '../input/Washington-CapitalBikeshare-2016.csv',
                            'out_file': '../input/Washington-2016-Summary.csv'},
             'Chicago': {'in_file': '../input/Chicago-Divvy-2016.csv',
                         'out_file': '../input/Chicago-2016-Summary.csv'},
             'NYC': {'in_file': '../input/NYC-CitiBike-2016.csv',
                     'out_file': '../input/NYC-2016-Summary.csv'}}

for city, filenames in city_info.items():
    condense_data(filenames['in_file'], filenames['out_file'], city)
    print_first_point(filenames['out_file'])
def number_of_trips(filename):
    """
    This function reads in a file with trip data and reports the number of
    trips made by subscribers, customers, and total overall.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        
        # initialize count variables
        n_subscribers = 0
        n_customers = 0
        
        # tally up ride types
        for row in reader:
            if row['user_type'] == 'Subscriber':
                n_subscribers += 1
            else:
                n_customers += 1
        
        # compute total number of rides
        n_total = n_subscribers + n_customers
        
        # return tallies as a tuple
        subs_prop = n_subscribers/n_total
        customers_prop = n_customers/n_total
        return('Number of trips = {}\nSubscribers proportion = {:.2f}\nCustomers proportion = {:.2f}'.format(n_total, subs_prop, customers_prop))
## Modify this and the previous cell to answer Question 4a. Remember to run ##
## the function on the cleaned data files you created from Question 3.      ##
washington_summary = '../input/Washington-2016-Summary.csv'
print("Washington Summary:\n{}\n".format(number_of_trips(washington_summary)))

chicago_summary = '../input/Chicago-2016-Summary.csv'
print("Chicago Summary:\n{}\n".format(number_of_trips(chicago_summary)))

nyc_summary = '../input/NYC-2016-Summary.csv'
print("NYC Summary:\n{}".format(number_of_trips(nyc_summary)))
## Use this and additional cells to answer Question 4b.                 ##
##                                                                      ##
## HINT: The csv module reads in all of the data as strings, including  ##
## numeric values. You will need a function to convert the strings      ##
## into an appropriate numeric type before you aggregate data.          ##
## TIP: For the Bay Area example, the average trip length is 14 minutes ##
## and 3.5% of trips are longer than 30 minutes.                        ##
def trips_duration(filename, free_duration = 30):
    """
    This function reads in a file with trip data and reports the duration of
    trips made on average and whether or not they are extra charged.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        
        # initialize total trips length, number of trips, and counters for whether or not the trip is extra charged. 
        total_trips_len = 0
        total_trips_count = 0
        extra_charge = 0
        no_extra_charge = 0
        
        for row in reader:
            current_row = float(row['duration'])
            
            if current_row > free_duration:
                extra_charge+=1
            else:
                no_extra_charge+=1
                
            total_trips_len+=current_row
            total_trips_count+=1
            
        average_trip_len = total_trips_len/total_trips_count
        extra_charge_prop = extra_charge/total_trips_count
        return ("Average trip length = {:.0f} minutes\nPropotion of extra charged trips = {:.4f}"
                .format(average_trip_len, extra_charge_prop))
washington_summary = '../input/Washington-2016-Summary.csv'
print("Washington Summary:\n{}\n".format(trips_duration(washington_summary)))

chicago_summary = '../input/Chicago-2016-Summary.csv'
print("Chicago Summary:\n{}\n".format(trips_duration(chicago_summary)))

nyc_summary = '../input/NYC-2016-Summary.csv'
print("NYC Summary:\n{}".format(trips_duration(nyc_summary)))
## Use this and additional cells to answer Question 4c. If you have    ##
## not done so yet, consider revising some of your previous code to    ##
## make use of functions for reusability.                              ##
##                                                                     ##
## TIP: For the Bay Area example data, you should find the average     ##
## Subscriber trip duration to be 9.5 minutes and the average Customer ##
## trip duration to be 54.6 minutes. Do the other cities have this     ##
## level of difference?                                                ##
def trips_duration2(filename):
    """
    This function reads in a file with trip data and reports the duration of
    trips made on average by subscribers and customers.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        
        # initialize subscriber trips length, customer trips length and their counters
        sub_trips_len = 0
        cust_trips_len = 0
        sub_trips_count = 0
        cust_trips_count = 0
        
        for row in reader:
            current_row = float(row['duration'])
            
            if row['user_type'] == 'Subscriber':
                sub_trips_len+=current_row
                sub_trips_count+=1
            else:
                cust_trips_len+=current_row
                cust_trips_count+=1
            
        average_sub_trip_len = sub_trips_len/sub_trips_count
        average_cust_trip_len = cust_trips_len/cust_trips_count
        return ("Average trip length for subscribers = {:.1f} minutes\nAverage trip length for customers = {:.1f} minutes."
                .format(average_sub_trip_len, average_cust_trip_len))
nyc_summary = '../input/NYC-2016-Summary.csv'
print("NYC Summary:\n{}".format(trips_duration2(nyc_summary)))
# load library
import calendar
import matplotlib.pyplot as plt
import numpy as np

# this is a 'magic word' that allows for plots to be displayed
# inline with the notebook. If you want to know more, see:
# http://ipython.readthedocs.io/en/stable/interactive/magics.html
%matplotlib inline 

# example histogram, data taken from bay area sample
data = [ 7.65,  8.92,  7.42,  5.50, 16.17,  4.20,  8.98,  9.62, 11.48, 14.33,
        19.02, 21.53,  3.90,  7.97,  2.62,  2.67,  3.08, 14.40, 12.90,  7.83,
        25.12,  8.30,  4.93, 12.43, 10.60,  6.17, 10.88,  4.78, 15.15,  3.53,
         9.43, 13.32, 11.72,  9.85,  5.22, 15.10,  3.95,  3.17,  8.78,  1.88,
         4.55, 12.68, 12.38,  9.78,  7.63,  6.45, 17.38, 11.90, 11.52,  8.63,]
plt.hist(data)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (m)')
plt.show()
def get_data(filename):
    """
    This function returns a list of all trips durations.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        
        dataset = []
        for row in reader:
            dataset.append(round(float(row['duration']), 2))
            
        return dataset
## Use this and additional cells to collect all of the trip times as a list ##
nyc_data = get_data(nyc_summary)

## and then use pyplot functions to generate a histogram of trip times.     ##
plt.hist(nyc_data, range=(0,60))
plt.title('Distribution of NYC Trip Durations')
plt.xlabel('Duration (m)')
plt.show()
## Use this and additional cells to answer Question 5. ##
def get_subscriber_data(filename):
    """
    This function returns a list of all trips durations by subscribers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        
        subscriber_dataset = []
        for row in reader:
            if row['user_type'] == 'Subscriber':
                subscriber_dataset.append(round(float(row['duration']), 2))
            
        return subscriber_dataset

def get_customer_data(filename):
    """
    This function returns a list of all trips durations by customers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        
        customer_dataset = []
        for row in reader:
            if row['user_type'] == 'Customer':
                customer_dataset.append(round(float(row['duration']), 2))
                
        return customer_dataset
nyc_subscriber_data = get_subscriber_data(nyc_summary)
plt.hist(nyc_subscriber_data, range=(0,75), width=5)
plt.title('Distribution of NYC\'s Subscriber Trip Durations')
plt.xlabel('Duration (m)')
plt.ylabel('Number of Trips')
plt.show()

nyc_customer_data = get_customer_data(nyc_summary)
plt.hist(nyc_customer_data, range=(0,75), width=5)
plt.title('Distribution of NYC\'s Customer Trip Durations')
plt.xlabel('Duration (m)')
plt.ylabel('Number of Trips')
plt.show()
## Use this and additional cells to continue to explore the dataset. ##
## Once you have performed your exploration, document your findings  ##
## in the Markdown cell above.##
def get_subscriber_monthly_data(filename):
    """
    This function returns a dictionary of number of ridership per month of subscribers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        trips_per_month = {}
        for row in reader:
            if row['user_type'] == 'Subscriber': 
                month_name = calendar.month_name[int(row['month'])]
                if trips_per_month.get(month_name) == None:
                    trips_per_month[month_name] = 1
                else:
                    trips_per_month[month_name] += 1
        return trips_per_month
    
    
def get_customer_monthly_data(filename):
    """
    This function returns a dictionary of number of ridership per month of customers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        trips_per_month = {}
        for row in reader:
            if row['user_type'] == 'Customer': 
                month_name = calendar.month_name[int(row['month'])]
                if trips_per_month.get(month_name) == None:
                    trips_per_month[month_name] = 1
                else:
                    trips_per_month[month_name] += 1
        return trips_per_month
        
nyc_sub_monthly_data = get_subscriber_monthly_data(nyc_summary)
nyc_cust_monthly_data = get_customer_monthly_data(nyc_summary)

bar_width = 0.3
x = np.arange(len(nyc_sub_monthly_data))
subscriber_bars = nyc_sub_monthly_data.values()
customer_bars = nyc_cust_monthly_data.values()

plt.figure(figsize=(14,7))
plt.bar(x, subscriber_bars, width=bar_width, label='Subscribers', color='#0D47A1')
plt.bar(x + bar_width, customer_bars, width=bar_width, label='Customers', color='#4DB6AC')

plt.xticks(x + bar_width/2, nyc_sub_monthly_data.keys())
plt.title('Monthly Ridership Per User Type in NYC')
plt.xlabel('Month')
plt.ylabel('Ridership')
plt.legend()
plt.show()