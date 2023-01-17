import numpy as np
import pandas as pd
import datetime
from scipy.spatial import distance

import ipywidgets as widgets
from ipywidgets import IntSlider

from datascience import *
from datetime import timedelta
from datetime import date
from datetime import datetime
import time

from IPython.display import display
from IPython.display import HTML

# from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

%matplotlib inline
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')


import locale
%load_ext line_profiler
import os
import xlrd as xlrd
raw_directory = "../input/"

# Declare column names to allow for auto completion :)

COL_SALE_DATE = 'SALE DATE'
COL_PURCHASE_DATE = 'PURCHASE DATE'
COL_SOLD_DATE = 'SOLD DATE'
COL_PURCHASE_PRICE = 'PURCHASE PRICE'
COL_SOLD_PRICE = 'SOLD PRICE'
COL_FULL_ADDRESS = 'FULL ADDRESS'
COL_PRICE_CHANGE = 'PRICE CHANGE'
COL_PERIOD = 'PERIOD'

COL_SALE_YEAR = 'SALE_YEAR'
COL_SALE_MONTH = 'SALE_MONTH'
COL_SALE_PRICE = 'SALE PRICE'

# Daily price change column name
COL_DAILY_PRICE_CHANGE = 'DAILY PRICE CHANGE'

COL_PURCHASE_DATE_SU = 'PURCHASE DATE SU'
COL_PURCHASE_PRICE_SU = 'PURCHASE PRICE SU'
COL_SOLD_DATE_SU = 'SOLD DATE SU'
def standard_units(any_numbers):
    "Convert any array of numbers to standard units."
    return (any_numbers - np.mean(any_numbers)) / np.std(any_numbers)


def correlation(t, x, y):
    return np.mean(standard_units(t.column(x)) * standard_units(t.column(y)))


def slope(table, x, y):
    r = correlation(table, x, y)
    return r * np.std(table.column(y)) / np.std(table.column(x))


def intercept(table, x, y):
    a = slope(table, x, y)
    return np.mean(table.column(y)) - a * np.mean(table.column(x))


def fit(table, x, y):
    a = slope(table, x, y)
    b = intercept(table, x, y)
    return a * table.column(x) + b


def residual(table, x, y):
    return table.column(y) - fit(table, x, y)


def scatter_fit(table, x, y):
    plots.scatter(table.column(x), table.column(y), s=20)
    plots.plot(table.column(x), fit(table, x, y), lw=2, color='gold')
    plots.xlabel(x)
    plots.ylabel(y)


# Helpers Functions

def print_stats(data):
    '''Prints common stats for a data array'''

    data_mean = np.mean(data)
    data_std = np.std(data)
    data_min = min(data)
    data_max = max(data)

    percent_5 = percentile(5, data)
    percent_95 = percentile(95, data)
    percent_1 = percentile(1, data)
    percent_99 = percentile(99, data)

    percent_25 = percentile(25, data)
    percent_50 = percentile(50, data)
    percent_75 = percentile(75, data)

    print("Avg:", data_mean, "\tStd:", data_std, "\tMin:", data_min, "\tMax:", data_max)
    print(" 5%:", percent_5, "\t95%:", percent_95)
    print(" 1%:", percent_1, "\t99%:", percent_99)
    print("25%:", percent_25, "\t50%:", percent_50, '\t75%', percent_75)


def print_col_stats(table, col_name):
    ''' Print the stats For column named'''

    print(col_name, "Stats")
    data = table.column(col_name)
    print_stats(data)


def draw_hist(table: Table, col_name, offset_percent=0):
    ''' Draw a histogram for table with an additional offset percent'''
    data = table.column(col_name)
    offset_start = percentile(offset_percent, data)
    offset_end = percentile(100 - offset_percent, data)
    table.hist(col_name, bins=np.arange(offset_start, offset_end, (offset_end - offset_start) / 20))


def col_stats(table, col_name):
    ''' Prints state for a column in table'''
    print_col_stats(table, col_name)
    draw_hist(table, col_name)

# List data files and directories in current directory
excel_files = os.listdir(raw_directory)

# Select only tje xls files
excel_files = [k for k in excel_files if '.xls' in k]
# Create an data frame to store
all_sales_data = pd.DataFrame()

# Load individual excel files. 
for excel_file in excel_files:
    print(excel_file)
    
    # Read excel, Note the headers could in row 4 or row 5 (index=3 or 4). 
    yearly_sales_data = pd.read_excel(raw_directory+excel_file, header=3, encoding='sys.getfilesystemencoding()')
   
    # Check if the first column is "BOROUGH"
    if not yearly_sales_data.columns[0].startswith('BOROUGH'):
        # Otherwise the data starts from row 5.
         yearly_sales_data = pd.read_excel(raw_directory+excel_file, header=4, encoding='sys.getfilesystemencoding()')
    
    yearly_sales_data.rename(columns=lambda x: x.strip(), inplace=True)
    
    all_sales_data = all_sales_data.append(yearly_sales_data)
all_sales_data.sample(5)
# Check for duplicate entries
print('Duplicate rows:', sum(all_sales_data.duplicated(all_sales_data.columns)))

#Delete the duplicates and check that it worked
all_sales_data = all_sales_data.drop_duplicates(all_sales_data.columns, keep='last')
sum(all_sales_data.duplicated(all_sales_data.columns))
#SALE DATE is object but should be datetime
all_sales_data[COL_SALE_DATE] = pd.to_datetime(all_sales_data[COL_SALE_DATE], errors='coerce')
all_sales_data['APARTMENT NUMBER'] = all_sales_data['APARTMENT NUMBER'].astype(str)

# remove additional whitespace in strings
all_sales_data = all_sales_data.applymap(lambda x: x.strip() if type(x) is str else x)
all_sales_data = Table.from_df(all_sales_data)
all_sales_data = all_sales_data.where(COL_SALE_PRICE, are.above(10000))
all_sales_data.labels
# Remove columns we don't actually need. e.g. lot, block etc

all_sales_data = all_sales_data.select(['SALE DATE', 'SALE PRICE', 'ADDRESS','APARTMENT NUMBER', 'YEAR BUILT', 'NEIGHBORHOOD', 'ZIP CODE', 'BUILDING CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY'])
def combine_address(address, aptNo):
    """Combine the address and Apartment into a single result"""
    temp = address.strip()
    if len(aptNo.strip()) > 0:
        temp = temp + ', ' + aptNo.strip()
    return temp

full_address = all_sales_data.apply(combine_address, ['ADDRESS', 'APARTMENT NUMBER'])

# Add a Full Address column
all_sales_data =  all_sales_data.with_column(COL_FULL_ADDRESS, full_address)
all_sales_data.group(['BUILDING CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY']).sort('count', descending=True).show(20)
condos = all_sales_data.where('BUILDING CLASS AT TIME OF SALE', are.contained_in("R1R2R3R4R6"))

# Spot Check condo data
condos.sample(5)
condos.group(COL_FULL_ADDRESS).sort(1, descending=True)
multi_sale_condos = condos.where(COL_FULL_ADDRESS, are.containing(',')).group(COL_FULL_ADDRESS).sort(1, descending=True).where('count', are.above(1))
multi_sale_condos
multi_sale_condos = multi_sale_condos.join(COL_FULL_ADDRESS, condos)
purchase_dates = multi_sale_condos.select(COL_FULL_ADDRESS, COL_SALE_DATE).group([0], min)
sold_dates = multi_sale_condos.select(COL_FULL_ADDRESS, COL_SALE_DATE).group([0], max)

# Note for the purposes of this analysis, we can ignore any additnal sales between min and max dates

# Spot check data
purchase_dates.show(5)
sold_dates.show(5)
# Update Labels

purchase_dates = purchase_dates.relabel(1, COL_PURCHASE_DATE)
sold_dates = sold_dates.relabel(1, COL_SOLD_DATE)
# Join with Condos to get the sale price
purchase_dates = purchase_dates.join(COL_FULL_ADDRESS, condos, COL_FULL_ADDRESS).where(COL_SALE_DATE, are.equal_to, COL_PURCHASE_DATE)
purchase_dates = purchase_dates.select( COL_FULL_ADDRESS, COL_PURCHASE_DATE, COL_SALE_PRICE)


sold_dates = sold_dates.join(COL_FULL_ADDRESS, condos, COL_FULL_ADDRESS).where(COL_SALE_DATE, are.equal_to, COL_SOLD_DATE)
sold_dates = sold_dates.select( COL_FULL_ADDRESS, COL_SOLD_DATE, COL_SALE_PRICE)

purchase_dates.show(5)
sold_dates.show(5)

# Hmm earlier we had 15871 now we have more! Could we have multiple sale records for the same date??
purchase_dates.groups([COL_FULL_ADDRESS, COL_PURCHASE_DATE]).sort(2, descending=True).where(2, are.above(1))
condos.where(COL_FULL_ADDRESS, are.equal_to('2 EAST 55 STREET, 921'))
purchase_dates = purchase_dates.group([COL_FULL_ADDRESS, COL_PURCHASE_DATE], min)
sold_dates = sold_dates.group([COL_FULL_ADDRESS, COL_SOLD_DATE], max)
# Relabel and join the first and last sale tables to create a new condo sales table

purchase_dates = purchase_dates.relabel(2, COL_PURCHASE_PRICE)
sold_dates = sold_dates.relabel(2, COL_SOLD_PRICE)

condo_sales = purchase_dates.join(COL_FULL_ADDRESS, sold_dates, COL_FULL_ADDRESS)
condo_sales
price_diffs = condo_sales.column(COL_SOLD_PRICE) - condo_sales.column(COL_PURCHASE_PRICE)
date_diffs = condo_sales.column(COL_SOLD_DATE) - condo_sales.column(COL_PURCHASE_DATE)

date_diffs = [ d.days for d in date_diffs ]

condo_sales = condo_sales.with_column( COL_PRICE_CHANGE, price_diffs, COL_PERIOD, date_diffs)
condo_sales.set_format(COL_PRICE_CHANGE, NumberFormatter())
col_stats(condo_sales, COL_PRICE_CHANGE)
# strip out the Price chnage outliers . 
percent_1 = percentile(1, price_diffs)
percent_99 = percentile(99, price_diffs)

largest_losses = condo_sales.where(COL_PRICE_CHANGE, are.below_or_equal_to(percent_1))
largest_gains = condo_sales.where(COL_PRICE_CHANGE, are.above_or_equal_to(percent_99))

condo_sales = condo_sales.where(COL_PRICE_CHANGE, are.between(percent_1, percent_99))
col_stats(condo_sales, COL_PRICE_CHANGE)
condo_sales.where( COL_PURCHASE_DATE, are.equal_to, COL_SOLD_DATE )

# Let's spot check these

all_sales_data.where(COL_FULL_ADDRESS, are.equal_to('100 CENTRAL PARK SOUTH, 4B')).sort(0)

# These look to be duplicate records, Let's ignore them
# Ignore multiple sales on same date
condo_sales = condo_sales.where( COL_PURCHASE_DATE, are.not_equal_to, COL_SOLD_DATE )
condo_sales.sort(COL_PERIOD)
condo_sales = condo_sales.where( COL_PERIOD, are.above(90) )
col_stats(condo_sales, COL_PERIOD)
daily_change = condo_sales.column(COL_PRICE_CHANGE) / condo_sales.column(COL_PERIOD) 

condo_sales = condo_sales.with_column(COL_DAILY_PRICE_CHANGE , daily_change ).sort(COL_DAILY_PRICE_CHANGE, descending=True)
col_stats(condo_sales, COL_DAILY_PRICE_CHANGE)

# strip out the Dailys Price change outliers. 
price_change_diffs = condo_sales.column(COL_DAILY_PRICE_CHANGE)

percent_1 = percentile(1, price_change_diffs)
percent_99 = percentile(99, price_change_diffs)

condo_sales = condo_sales.where(COL_DAILY_PRICE_CHANGE, are.between(percent_1, percent_99))

years = [ d.year for d in condos.column(COL_SALE_DATE) ]

months = [ d.month for d in condos.column(COL_SALE_DATE) ]

condos = condos.with_column('SALE_YEAR', years, 'SALE_MONTH', months)

condo_mean = condos.select(COL_SALE_YEAR, COL_SALE_PRICE).group(COL_SALE_YEAR, np.mean).sort(0)
condo_mean.plot(COL_SALE_YEAR)


neighborhoods = condos.group('NEIGHBORHOOD').sort(0).column(0)

def plot_neighborhood(neighborhood:str):
    '''Plot the average sale for a specified neign'''
    condos.where('NEIGHBORHOOD', are.equal_to(neighborhood)).select(COL_SALE_YEAR, COL_SALE_PRICE).group(0, np.mean).plot(0, label=neighborhood)
    plots.title = neighborhood
    print(neighborhood)
    plots.plot(condo_mean.column(0), condo_mean.column(1), color='gold', label=neighborhood )
    return

# ignore = interact(plot_neighborhood, neighborhood=neighborhoods)

plot_neighborhood('ALPHABET CITY')

plot_neighborhood('MIDTOWN EAST')
sales_2010 = condo_sales.where(COL_PURCHASE_DATE, are.between( datetime(2010, 1, 1), datetime(2010, 12, 31)))

Table().with_columns(
    'PERIOD',  sales_2010.column(COL_PERIOD), 
    'PRICE CHANGE', sales_2010.column(COL_PRICE_CHANGE)
).scatter(0, 1, fit_line=True)

print('Correlation betweeen Price Change and Time: ', correlation(sales_2010, COL_PERIOD, COL_PRICE_CHANGE))
Table().with_columns(
    'PURCHASED PRICE',  sales_2010.column(COL_PURCHASE_PRICE), 
    'SOLD PRICE', sales_2010.column(COL_SOLD_PRICE)
).scatter(0, fit_line=True)

print('Correlation betweeen Purchase Price and Sold Price: ', correlation(sales_2010, COL_PURCHASE_PRICE, COL_SOLD_PRICE))
a = slope(condo_sales, COL_PURCHASE_PRICE, COL_SOLD_PRICE)
b = intercept(condo_sales, COL_PURCHASE_PRICE, COL_SOLD_PRICE)

first_prices = condo_sales.column(COL_PURCHASE_PRICE)

predicted = first_prices * a + b

errors = condo_sales.column(COL_SOLD_PRICE) - predicted


Table().with_columns(
    'PURCHASE PRICE',  condo_sales.column(COL_PURCHASE_PRICE), 
    'ERRORS', errors
).scatter(0, fit_line=True)


price_band = condo_sales.where(COL_PURCHASE_PRICE, are.between(750000, 1000000))

price_band.scatter(COL_PURCHASE_PRICE, COL_SOLD_PRICE, fit_line=True)
print('Correlation betweeen Purchase Price and Sold Price for apts between 750K-1MM: ', correlation(price_band, COL_PURCHASE_PRICE, COL_SOLD_PRICE))
col_stats(condo_sales, COL_PURCHASE_PRICE)
purchase_prices = condo_sales.column(COL_PURCHASE_PRICE)

percent_1 = percentile(1, purchase_prices)
percent_99 = percentile(99, purchase_prices)

condo_sales = condo_sales.where(COL_PURCHASE_PRICE, are.between(percent_1, percent_99))
col_stats(condo_sales, COL_PURCHASE_PRICE)
percents = sales_2010.column(COL_PRICE_CHANGE) / sales_2010.column(COL_PURCHASE_PRICE) * 100

COL_PRICE_PERCENT = 'PRICE CHANGE %'

sales_2010 = sales_2010.with_column(COL_PRICE_PERCENT, percents)
col_stats(sales_2010, COL_PRICE_PERCENT)
# draw_hist(sales_2010, COL_PRICE_PERCENT, 2)
# strip out the Price Percent change outliers. 

price_changes = sales_2010.column(COL_PRICE_PERCENT)

percent_2 = percentile(2, price_changes)
percent_98 = percentile(98, price_changes)

sales_2010 = sales_2010.where(COL_PRICE_PERCENT, are.between(percent_2, percent_98))
sales_2010.scatter(COL_PERIOD, COL_PRICE_PERCENT, fit_line=True)
print('Correlation Price Change, Time Period', correlation(sales_2010, COL_PERIOD, COL_PRICE_PERCENT))

periods = sales_2010.column(COL_PERIOD)

min_period = min(periods)
max_period = max(periods)


period_groups = []
period_sales = []

for i in np.arange(min_period, max_period, 30 ):
    period_groups.append(i)
    period_sales.append(np.mean(sales_2010.where(COL_PERIOD, are.between(i, i+30)).column(COL_PRICE_PERCENT)))
    

Table().with_columns(
    COL_PERIOD,  period_groups, 
    COL_PRICE_PERCENT, period_sales
).scatter(0)
# Trim the outliers. 
percents = condo_sales.column(COL_PRICE_CHANGE) / condo_sales.column(COL_PURCHASE_PRICE) * 100

condo_sales = condo_sales.with_column(COL_PRICE_PERCENT, percents)

percent_1 = percentile(1, percents)
percent_99 = percentile(99, percents)

condo_sales = condo_sales.where(COL_PRICE_PERCENT, are.between(percent_1, percent_99))

def plot_price_change_year(year):
    ''' Plot the price change % for a given year'''
    valid_sales = condo_sales.where(COL_PURCHASE_DATE, are.between_or_equal_to( datetime(year, 1, 1), datetime(year+1, 1, 1)))

    min_period = min(periods)
    max_period = max(periods)


    period_groups = []
    period_sales = []

    for i in np.arange(min_period, max_period, 30 ):
        period_groups.append(i)
        period_sales.append(np.mean(valid_sales.where(COL_PERIOD, are.between(i, i+30)).column(COL_PRICE_PERCENT)))

    Table().with_columns(
        COL_PERIOD,  period_groups, 
        COL_PRICE_PERCENT, period_sales
    ).scatter(0)

# Removed interact for published
# _ = interact(plot_price_change_year, year=np.arange(2003,2019) )


print('2007')
plot_price_change_year(2007)

print('2013')
plot_price_change_year(2012)
# Convert to standard units
purchase_dates_timestamps = [ date.timestamp() for date in condo_sales.column(COL_PURCHASE_DATE)]
purchase_dates_su = standard_units(purchase_dates_timestamps)

sold_dates_timestamps = [ date.timestamp() for date in condo_sales.column(COL_SOLD_DATE)]
sold_dates_su = standard_units(sold_dates_timestamps)

purchase_price_su = standard_units(condo_sales.column(COL_PURCHASE_PRICE))

condo_sales_su = condo_sales.with_column(
    COL_PURCHASE_DATE_SU, purchase_dates_su, 
    COL_PURCHASE_PRICE_SU, purchase_price_su, 
    COL_SOLD_DATE_SU, sold_dates_su,
)

# Create the training and testing sets.
training_sales, test_sales = condo_sales_su.split(int(condo_sales.num_rows * 0.6))
training_sales

# Columns used to the calculate the distance between two point i.e. 2 properties that were purchased and sold. 
# We picked the purchase date & price and the sold date converted to standard units as the important columns
# to use for calculating the distance.
distance_columns = [COL_PURCHASE_DATE_SU, COL_PURCHASE_PRICE_SU, COL_SOLD_DATE_SU]

def all_distances(training, new_point):
    """Returns an array of distances
    between each point in the training set
    and the new point (which is a row of attributes)"""
    attributes = training.select(distance_columns)
    return distance.cdist( attributes.to_array().tolist(), [new_point]).flatten()

def table_with_distances(training, new_point):
    """Augments the training table 
    with a column of distances from new_point"""
    return training.with_column('Distance', all_distances(training, new_point))

def closest(training, new_point, k):
    """Returns a table of the k rows of the augmented table
    corresponding to the k smallest distances"""
    with_dists = table_with_distances(training, new_point)
    sorted_by_distance = with_dists.sort('Distance')
    topk = sorted_by_distance.take(np.arange(k))
    return topk

def estimate(training, purchase_point, k):
    """Estimate a price based on nearest neighbours"""
    close_points = closest(training_sales, purchase_point, k)
    avg_price_change = np.mean(close_points.column(COL_PRICE_PERCENT))
    return avg_price_change

def evaluate_accuracy(training, test, k):
    """Evalute the accuracy of the model generating using training data on test data"""
    # select the columns to compare
    test_attributes = test.select(distance_columns)

    # compute the predicted change for each test row
    def price_testrow(row):
        return estimate(training, row, k)

    # Calculate the predicted price and error
    c = test_attributes.apply(price_testrow)
    
    estimated = test[COL_PURCHASE_PRICE] * (100 + c )/100
    error = (test[COL_SOLD_PRICE] - estimated  ) / test[COL_SOLD_PRICE] * 100
    
    return test.with_column("Estimated", estimated, 'Error %', error)
estimates = evaluate_accuracy(training_sales, test_sales, 10)
estimates.scatter(COL_SOLD_PRICE, 'Error %')

col_stats(estimates, "Error %")
