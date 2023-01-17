# having touble adding data from the Add Data interface today, so I'll use urllib.request.urlopen

# 'Ooooops, something went wrong. Please try reopening this dialog. If the problem persists, please contact contact support and we'll get you squared away.'

# https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3

# got an URLError: <urlopen error [Errno -3] Temporary failure in name resolution> ... changed kernel Internet setting to Internet connected

import urllib.request

import shutil



url = 'http://www.ostaski.net/Data/winemag-data_first50k.csv'

file_name = 'winemag-data_first50k.csv'



with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:

    shutil.copyfileobj(response, out_file)



# UGH! now I get OSError: [Errno 30] Read-only file system: '../input/winemag-data_first50k.csv'

# guess I'll wait until the Add Data interface is working ... no need, I can download to my home directory :) 



!ls -ail
# from Rachael's kernel: https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5

import sys

!{sys.executable} -m pip install csvvalidator
# changed kernel Internet setting back to Internet blocked, but then I lose my winemag-data_first50k.csv! Best to wait until we're done with that file

# import everything from the csvvalidator package

from csvvalidator import *



# Specify which fields (columns) your .csv needs to have

# You should include all fields you use in your dashboard

field_names = ('id',

               'country',

               'description',

               'designation',

               'points',

               'price',

               'province',

               'region_1',

               'region_2',

               'variety',

               'winery'

               )



# create a validator object using these fields

validator = CSVValidator(field_names)



# write some checks to make sure specific fields are the way we expect them to be

# here we just want to validate points and price, though variety and winery might come in handy down the road

# can't find a way to validate Strings!?!

validator.add_value_check('points', # the name of the field

                          int, # a function that accepts a single argument and 

                               # raises a `ValueError` if the value is not valid.

                               # Here checks that "points" is an integer.

                          'EX1', # code for exception

                          'points must be an integer'# message to report if error thrown

                         )



validator.add_value_check('price',

                          int, 

                          'EX2',

                          'price must be an integer'

                         )



import csv



file_name = 'winemag-data_first50k.csv'

with open(file_name) as csvfile:

    test_csv = csv.reader(csvfile, delimiter=',')

    # validate our csv (doing this inside the block since test_csv needs to be open)

    validator.validate(test_csv)
import pandas as pd



csv_file = 'winemag-data_first50k.csv'



# the last column is all Nans and we don't need it. might be an artifact from the initial subsetting

cols_to_use = ['id', 'country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery']

data = pd.read_csv(csv_file, usecols = cols_to_use)



# taking a look see

data.head()
# how many rows do we have?

len(data)
# OK, now let's remove any rows where columns of potential interest (points, price, variety or winery) are NA

# maybe a bit drastic, but I can't think of a way to impute values for these columns

#data.dropna(subset = [‘points’]) # SyntaxError: invalid character in identifier

#data.dropna(axis=0, subset=[5]) # points # KeyError: [5]

# what a PITA :(

points = 'points'

price = 'price'

variety = 'variety'

winery = 'winery'

data = data.dropna(axis=0, subset=[points])

data = data.dropna(axis=0, subset=[price])

data = data.dropna(axis=0, subset=[variety])

data = data.dropna(axis=0, subset=[winery])



# how many rows now?

len(data)
# OK now let's check our data types

data.dtypes
# Looks like we only need to cast points as an integer

#import numpy as np

#data = data['points'].astype(np.int64) # ValueError: invalid literal for int() with base 10: ' Franscioni Vineyard'

# Oops! We still had a few misaligned rows ... fixed those

#pd.to_numeric(data['points'], errors='coerce')

#data['points'].astype(int)

data = data.astype({"points": int})

data.dtypes
# Save our data df, hopefully to ../input (nope: OSError: [Errno 30] Read-only file system:)

output = 'processed_winemag-data_first50k.csv'

data.to_csv(output, index=False, encoding='utf8')

!ls -ail

# Finally change the kernel's Internet setting to Internet blocked