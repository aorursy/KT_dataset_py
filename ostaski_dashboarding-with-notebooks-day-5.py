# to use the csvvalidator package, you'll need to 
# install it. Turn on the internet (in the right-hand
# panel; you'll need to have phone validated your account)

import sys
!{sys.executable} -m pip install csvvalidator
# import everything from the csvvalidator package
from csvvalidator import *

# Specify which fields (columns) your .csv needs to have
# You should include all fields you use in your dashboard
field_names = ('key',
               'date',
               'unit_name',
               'units_sold',
               'store'
               )

# create a validator object using these fields
validator = CSVValidator(field_names)

# write some checks to make sure specific fields 
# are the way we expect them to be
validator.add_value_check('key', # the name of the field
                          int, # a function that accepts a single argument and 
                               # raises a `ValueError` if the value is not valid.
                               # Here checks that "key" is an integer.
                          'EX1', # code for exception
                          'key must be an integer'# message to report if error thrown
                         )
validator.add_value_check('date', 
                          # check for a date with the sepcified format
                          datetime_string('%Y-%m-%d'), 
                          'EX2',
                          'invalid date'
                         )
validator.add_value_check('units_sold',
                          # makes sure the number of units sold is an integer
                          int,
                          'EX3',
                          'number of units sold not an integer'
                         )
validator.add_value_check('store', 
                          # check that the "store" field only has
                          # "store1" and "store2" in it
                          enumeration('store1', 'store2'),
                          'EX4', 
                          'store name not recognized')
import csv
from io import StringIO

# this is example of validating a good dataset that's set
# up the way we want it to be

# sample csv
good_data = StringIO("""key,date,unit_name,units_sold,store,notes
1,2018-12-01,product1,5,store1,""
2,2018-12-01,product2,200,store2,"Big day!"
3,2018-12-04,product1,2,store1,""
""")

# read text in as a csv
test_csv = csv.reader(good_data)

# validate our good csv
validator.validate(test_csv)
# and this one is an example of a dataset with a lot
# of problems, which throws a lot of helpful exceptions

# sample csv
bad_data = StringIO("""key,date,unit_name,units,store
1,2018-12-01,product1,5,store1
2,2018-12-01,product2,two hundred,store2
2.5,12-04-2018,product1,2,store3
""")

# read text in as a csv
test_csv = csv.reader(bad_data)

# validate our bad csv (generates a lot of errors)
validator.validate(test_csv)
# we'll be using the testing functions build into pandas
import pandas as pd

# function to do some data cleaning 
# (adopted from Dinara Sultangulova's dashboad:
# https://www.kaggle.com/oftomorrow/dashboarding-with-notebooks-ny-collisions)
def process_data(raw_data):
    processed_data = raw_data[raw_data.BOROUGH.notnull()]\
    .filter(items=['BOROUGH','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'])\
    .groupby(['BOROUGH'], as_index=False)\
    .sum()
    
    return(processed_data)
# testing data to pass into our function (based on 
# slice from data function was originally written
# for)
test_data = StringIO("""BOROUGH,ZIP CODE,LATITUDE,LONGITUDE,LOCATION,NUMBER OF PERSONS KILLED,NUMBER OF PERSONS INJURED
    QUEENS,11362,40.76272,-73.72816999999999,"{\'longitude\': \'-73.72817\', \'latitude\': \'40.76272\', \'needs_recoding\': False}",0,0
    BROOKLYN,11211,40.710196999999994,-73.95843,"{\'longitude\': \'-73.95843\', \'latitude\': \'40.710197\', \'needs_recoding\': False}",1,3
    BRONX,10454,40.803554999999996,-73.91184,"{\'longitude\': \'-73.91184\', \'latitude\': \'40.803555\', \'needs_recoding\': False}",0,1
    BROOKLYN,11221,40.694922999999996,-73.915565,"{\'longitude\': \'-73.915565\', \'latitude\': \'40.694923\', \'needs_recoding\': False}",0,2""")
test_csv = pd.read_csv(test_data)


# hand built example with what we expect
# our function to output given the test data
output_data_correct = StringIO("""BOROUGH,NUMBER OF PERSONS INJURED,NUMBER OF PERSONS KILLED
    BRONX,1,0
    BROOKLYN,5,1
    QUEENS,0,0""")
output_csv_correct = pd.read_csv(output_data_correct)

# check to see if the test dataframe & the
# dataframe our function prouduces are the same
# (this will produce no output)
pd.testing.assert_frame_equal(output_csv_correct, process_data(test_csv))
# now a dataframe that has an error in it
output_data_errors = StringIO("""BOROUGH,NUMBER OF PERSONS INJURED,NUMBER OF PERSONS KILLED
    BRONX,0,0
    BROOKLYN,5,1
    QUEENS,0,b""")
output_csv_errors = pd.read_csv(output_data_errors)

# check to see if the test dataframe & the
# dataframe our function produces are the same
# (this will raise an error!)
pd.testing.assert_frame_equal(output_csv_errors, process_data(test_csv))