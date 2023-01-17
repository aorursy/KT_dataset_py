## import all necessary packages and functions.
import csv # read and write csv files
from datetime import datetime # operations to parse dates
from pprint import pprint # use to print data structures like dictionaries in
                          # a nicer way than the base print function.
import seaborn as sns
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
        trip_reader =csv.DictReader(f_in)
        
        ## TODO: Use a function on the DictReader object to read the     ##
        ## first trip from the data file and store it in a variable.     ##
        ## see https://docs.python.org/3/library/csv.html#reader-objects ##
        first_trip = next(trip_reader)
        
        ## TODO: Use the pprint library to print the first trip. ##
        ## see https://docs.python.org/3/library/pprint.html     ##
        pprint(first_trip)
        
    # output city name and first trip for later testing
    return (city, first_trip)

# list of files for each city
data_files = ['../input/NYC-CitiBike-2016.csv',
              '../input/Chicago-Divvy-2016.csv',
              '../input/Washington-CapitalBikeshare-2016.csv',]

# print the first trip from each file, store in dictionary
example_trips = {}
for data_file in data_files:
    city, first_trip = print_first_point(data_file)
    example_trips[city] = first_trip
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
    if city=="Washington":
        duration = (float(datum['Duration (ms)'])*(0.001))/60
    else:
        duration = (float(datum['tripduration']))/60
    # YOUR CODE HERE
    
    return duration


# Some tests to check that your code works. There should be no output if all of
# the assertions pass. The `example_trips` dictionary was obtained from when
# you printed the first trip from each of the original data files.
tests = {'NYC': 13.9833,
         'Chicago': 15.4333,
         'Washington': 7.1231}

for city in tests:
    assert abs(duration_in_mins(example_trips[city], city) - tests[city]) < .001
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
    
    # YOUR CODE HERE
    if city=="NYC":
        starttime = datum['starttime']
        s1 = datetime.strptime(starttime, '%m/%d/%Y %H:%M:%S')
        return(s1.month,s1.hour,s1.strftime('%A'))
    
    elif city=="Washington":
        starttime = datum['Start date']
        s1 = datetime.strptime(starttime, '%m/%d/%Y %H:%M')
        return(s1.month,s1.hour,s1.strftime('%A'))
    
    elif city=="Chicago":
        starttime = datum['starttime']
        s1 = datetime.strptime(starttime, '%m/%d/%Y %H:%M')
        return(s1.month,s1.hour,s1.strftime('%A'))
    


# Some tests to check that your code works. There should be no output if all of
# the assertions pass. The `example_trips` dictionary was obtained from when
# you printed the first trip from each of the original data files.
tests = {'NYC': (1, 0, 'Friday'),
         'Chicago': (3, 23, 'Thursday'),
         'Washington': (3, 22, 'Thursday')}

for city in tests:
    assert time_of_trip(example_trips[city], city) == tests[city]
def type_of_user(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the type of system user that made the
    trip.
    
    Remember that Washington has different category names compared to Chicago
    and NYC. 
    """
    
    # YOUR CODE HERE
    if city=="NYC" or city=="Chicago":
        user_type = datum['usertype']
    elif city=="Washington":
        if datum['Member Type']=="Registered":
            user_type = "Subscriber"
        else:
            user_type = "Customer"
    
    return user_type


# Some tests to check that your code works. There should be no output if all of
# the assertions pass. The `example_trips` dictionary was obtained from when
# you printed the first trip from each of the original data files.
tests = {'NYC': 'Customer',
         'Chicago': 'Subscriber',
         'Washington': 'Subscriber'}

for city in tests:
    assert type_of_user(example_trips[city], city) == tests[city]
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
            new_point['duration'] = duration_in_mins(row,city)
            new_point['month'],new_point['hour'],new_point['day_of_week'] = time_of_trip(row,city)
            new_point['user_type'] = type_of_user(row,city)

            ## TODO: write the processed information to the output file.     ##
            ## see https://docs.python.org/3/library/csv.html#writer-objects ##
            f_out.write(str(new_point['duration']))
            f_out.write(",")
            f_out.write(str(new_point['month']))
            f_out.write(",")
            f_out.write(str(new_point['hour']))
            f_out.write(",")
            f_out.write(str(new_point['day_of_week']))
            f_out.write(",")
            f_out.write(str(new_point['user_type']))
            f_out.write("\n")      
# Generate summary files for each of the cities
city_info = {'Washington': {'in_file': '../input/Washington-CapitalBikeshare-2016.csv',
                            'out_file': '../working/Washington-2016-Summary.csv'},
             'Chicago': {'in_file': '../input/Chicago-Divvy-2016.csv',
                         'out_file': '../working/Chicago-2016-Summary.csv'},
             'NYC': {'in_file': '../input/NYC-CitiBike-2016.csv',
                     'out_file': '../working/NYC-2016-Summary.csv'}}

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
        return(n_subscribers, n_customers, n_total)
## Modify this and the previous cell to answer Question 4a. Remember to run ##
## the function on the cleaned data files you created from Question 3.      ##

data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    n_subscribers,n_customers,n_total = number_of_trips(datafile)
    sub_p = (n_subscribers / n_total)*100
    cus_p = (n_customers / n_total)*100
    # print city name for reference
    city = datafile.split('-')[0].split('/')[-1]
    print('In ' + city +' city, the total no. of Subscribers are ' + str(n_subscribers) +' ('+ str(abs(sub_p)) +'%)' + ', Customers are ' + str(n_customers)+' ('+ str(abs(cus_p)) +'%)' + ' and total no. of rides is ' + str(n_total) + '\n')

#data_file = './examples/BayArea-Y3-Summary.csv'
#print(number_of_trips(data_file))
## Use this and additional cells to answer Question 4b.                 ##
##                                                                      ##
## HINT: The csv module reads in all of the data as strings, including  ##
## numeric values. You will need a function to convert the strings      ##
## into an appropriate numeric type before you aggregate data.          ##
## TIP: For the Bay Area example, the average trip length is 14 minutes ##
## and 3.5% of trips are longer than 30 minutes.                        ##

def trip_length(filename):
    total_ride = 0
    s = 0
    t_time = 0
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            total_ride = total_ride + 1
            duration = float(row['duration'])
            t_time = t_time + duration
            if duration>30:
                s = s + 1
            
        avg = (t_time/total_ride)
        p = (s/total_ride)
        
        return (t_time,total_ride,avg,p)
data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    t_time,total,avg,p  = trip_length(datafile)
    # print city name for reference
    city = datafile.split('-')[0].split('/')[-1]
    print("In "+city+" City: ")
    print("The Average Trip Length Is: "+ str(avg))
    print("Proportion of rides longer than 30 min: "+ str(p)+"\n")
    
## Use this and additional cells to answer Question 4c. If you have    ##
## not done so yet, consider revising some of your previous code to    ##
## make use of functions for reusability.                              ##
##                                                                     ##
## TIP: For the Bay Area example data, you should find the average     ##
## Subscriber trip duration to be 9.5 minutes and the average Customer ##
## trip duration to be 54.6 minutes. Do the other cities have this     ##
## level of difference?                                                ##
def ridership(filename):
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        s = 0
        c = 0
        s_duration = 0
        c_duration = 0
        for row in reader:
            user_type = row['user_type']
            duration = float(row['duration'])
            if user_type == 'Subscriber':
                s = s + 1
                s_duration = s_duration + duration
            else:
                c = c + 1
                c_duration = c_duration + duration
            
        s_avg = s_duration/s
        c_avg = c_duration/c
        return (s,c,s_avg,c_avg)

data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    s,c,s_avg,c_avg  = ridership(datafile)
    # print city name for reference
    city = datafile.split('-')[0].split('/')[-1]
    print("In "+city+" City: ")
    print("Average Ride Taken By Subscribers: "+ str(s_avg))
    print("Average Ride Taken By Customers  : "+ str(c_avg)+"\n")
    
# load library
import matplotlib.pyplot as plt

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
sns.set_style('whitegrid')
## Use this and additional cells to collect all of the trip times as a list ##
## and then use pyplot functions to generate a histogram of trip times.     ##
def plot_trip(filename):
    #totaltime,total_trip,avg,p = trip_length(filename)
    data = []
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            data.append(float(row['duration']))
            #plt.hist(reader)
        plt.hist(data)
        city = datafile.split('-')[0].split('/')[-1]
        plt.title(city)
        plt.xlabel("Trip Duration")
        plt.ylabel("No of Users")
        plt.show()
data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    plot_trip(datafile)
#Again plot the graph for different cities
#change the bin size
import numpy as py
def plot_trip_again(filename):
    #totaltime,total_trip,avg,p = trip_length(filename)
    data = []
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            data.append(float(row['duration']))
            #plt.hist(reader)
        bins = py.arange(0,100,5)
        plt.hist(data,bins)
        city = datafile.split('-')[0].split('/')[-1]
        plt.title(city)
        plt.xlabel("Trip Duration")
        plt.ylabel("No of Users")
        plt.show()
data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    plot_trip_again(datafile)
## Use this and additional cells to answer Question 5. ##
## Plot a graph 
def plot_cus(filename):
    data1 = []
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if row['user_type']=='Customer':
                if float(row['duration'])<75:
                    data1.append(float(row['duration']))
        
        bins = py.arange(0,80,5)
        plt.hist(data1,bins,histtype='bar',rwidth=0.8)
        plt.xticks(bins)
        city = datafile.split('-')[0].split('/')[-1]
        plt.title("Distribution of Trip Durations for Customer in " + city)
        plt.xlabel("Trip Duration")
        plt.ylabel("No of Users")
        plt.show()
data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    plot_cus(datafile)
## Use this and additional cells to continue to explore the dataset. ##
## Once you have performed your exploration, document your findings  ##
## in the Markdown cell above.                                       ##
def monthly_analysis(filename):
    month_trip_sub={}
    month_trip_cus={}
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        data=[]
        data1=[]
        for i in range(1,13):
            month_trip_sub[str(i)]=0
            month_trip_cus[str(i)]=0 
        for row in reader:
            if row['user_type']=='Subscriber':
                month_trip_sub[row['month']]+=1
            else:
                month_trip_cus[row['month']]+=1
        data = month_trip_sub.values()
        data1 = month_trip_cus.values()
        bins = py.arange(1,13,1)
        plt.bar(bins+0,data,width=0.4,label ='Subscriber')
        plt.bar(bins+.40,data1,width=0.4,label ='Customer')
        city = datafile.split('-')[0].split('/')[-1]
        plt.title("Monthly Analysis Of City " + city + " (Subscribers/Customer)")
        plt.xticks(bins,['Jan','Feb','Mar','Apr','may','Jun','July','Aug','Sep','Oct','Nov','Dec'])
        plt.xlabel("Month")
        plt.ylabel("No of Users")
        plt.legend()
        plt.show()
##Count Total Trips in Each Month and plot a bar graph
def total_trips_monthly(filename):
    total_trip_monthly = []
    total={}
    for i in range(1,13):
        total[str(i)]=0
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            total[row['month']]+=1
            #total_trip_monthly.append(int(row['month']))
        total_trip_monthly = total.values()
        bins = py.arange(1,13,1)
        #plt.hist(total_trip_monthly,bins,histtype='bar',rwidth=0.8)
        plt.bar(bins,total_trip_monthly,width=0.9)
        city = datafile.split('-')[0].split('/')[-1]
        plt.title("Monthly Analysis Of City " + city+ " (Total Users)")
        plt.xticks(bins,['Jan','Feb','Mar','Apr','may','Jun','July','Aug','Sep','Oct','Nov','Dec'])
        plt.xlabel("Month")
        plt.ylabel("No of Users")
        plt.show()
    
##Calculate ratio of subscriber and customer in each month
def ratio(filename):
    month_trip_sub={}
    month_trip_cus={}
    with open(filename,'r') as fin:
        reader = csv.DictReader(fin)
        data=[]
        data1=[]
        data2=[]
        for i in range(1,13):
            month_trip_sub[str(i)]=0
            month_trip_cus[str(i)]=0 
        for row in reader:
            if row['user_type']=='Subscriber':
                month_trip_sub[row['month']]+=1
            else:
                month_trip_cus[row['month']]+=1
        for i in range(1,13):
            data.append(month_trip_sub[str(i)])
            data1.append(month_trip_cus[str(i)])
        for j in range(12):
            data2.append(data[j]/data1[j])
        bins = py.arange(1,13,1)
        #plt.hist(data2,bins,histtype='bar',rwidth=0.8)
        plt.bar(bins,data2,width=0.6)
        city = datafile.split('-')[0].split('/')[-1]
        plt.title("Ratio Of (Subscriber/Customer) in " + city)
        plt.xticks(bins,['Jan','Feb','Mar','Apr','may','Jun','July','Aug','Sep','Oct','Nov','Dec'])
        plt.xlabel("Month")
        plt.ylabel("Ratio")
        plt.show()
        
data_file = ['../working//Washington-2016-Summary.csv', '../working//Chicago-2016-Summary.csv', '../working//NYC-2016-Summary.csv']
for datafile in data_file:
    total_trips_monthly(datafile)
    monthly_analysis(datafile)
    ratio(datafile)
