# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load in the data. Display the basic info per numeric data column.
kiva_loans = pd.read_csv('../input/kiva_loans.csv')
# kiva_loans.describe()
# Print out the data types for each column
# kiva_loans.dtypes
# When you get time you definitely want to look at the distribution of the data and see what sticks out.
"""
It looks as if the vast majority of the loans are given out for Agriculture, Food, and Retail related reasons.

My first inclination after seeing the overlap between Sector and Activity is to try to pare down the categories
in Activity into the appropriate Sector (which might mean just ignoring the Activity for the analysis and relying
on Sector for initial analysis).

"""
# Sector
# Self-explanatory. Few categories. Easy enough to work with.s
# print(kiva_loans.sector.head(100))
ohe_sector = pd.get_dummies(kiva_loans.sector)
kiva_loans.sector.value_counts().plot.bar()

# Activity
# same thing for activity. Categories.
# print(kiva_loans.activity.head(100))
# kiva_loans.activity.value_counts()
# kiva_loans.activity.value_counts().head(40).plot.bar()
ohe_activity = pd.get_dummies(kiva_loans.activity)

print(ohe_sector.columns & ohe_activity.columns)
print(ohe_sector.columns)
activity_top_15_list = kiva_loans.activity.value_counts().head(15).axes
# print(activity_top_15_list & ohe_sector.columns)
"""
Just use the country, ignore country_code for now. Assume accuracy of data (for now) 0_0.

Wow. Looks like a significant portion of the overall aid goes to the Phillipines and Kenya.
"""
# Country_code and Country
# same with country_code. Might be good to ignore this column or the country column (unnecessary repetition).
# print(kiva_loans.country_code.head(100))
kiva_loans.country.value_counts().head(40).plot.bar()
# Tags
# need to separate out the data with tags. Looks like that might help (as long as the tags were reliably)
# applied to the data. If they weren't, this might be useless.
# print(kiva_loans.tags.head(100))

# Use
# Well shit. Looks like use is probably the most difficult of the categories. Might need to actually do lexical
# analysis of some kind. I'd guess that there are similar categories (purchase / buy, purpose, 
# repair, food...etc...). Might behoove you to separate into categories.
# print(kiva_loans.use.head(100))

# Currency
# might be interesting to see if all of the currencies match up to the country of origin for the loan.
# print(kiva_loans.currency.head(100))

# Region
# Turns out that region has more than one type of geographical range (county, district, state, city, street)
# and they aren't filled out the same across the board, or correctly at that! Scratch the use being the hardest
# part. You'll have to figure out how to correlate general geographic location with the regions given.
# print(kiva_loans.region.head(100))

# Repayment_interval
# These are pretty self explanatory. There seems to be only a few different types of categories. Should be easy
# enough to use. I wonder how this correlates to the region it was issued in and the financial capacity / profes-
# sion of the individual(s) that received the loan.
# print(kiva_loans.repayment_interval.head(100))