import pandas as pd

import numpy as np

import calendar as cal

from datetime import *

from dateutil.relativedelta import *

import holidays
date_range_start = '2018-01-01'

date_range_end = '2020-12-31'
# Example if you would like to use the package holiday calendar

for date, name in sorted(holidays.US(years=[2018,2019,2020]).items()):

    print(date, name)
# Defining custom calendar to match company

class CorporateHolidays(holidays.UnitedStates):

    def _populate(self, year):

        # Populate the holiday list with the default US holidays

        holidays.UnitedStates._populate(self, year)

        # Remove Holiday Date(s)

        self.pop(datetime.date(datetime(year, 10, 1) + relativedelta(weekday=MO(+2))), None) # Removing Columbus Day

        # Add Holiday Date(s)

        self.append({datetime.date(datetime(year, 12, 24)) : "Christmas Eve"})

        self.append({datetime.date(datetime(year, 12, 31)) : "New Years Eve"})

        self.append({datetime.date(datetime(year, 11, 1) + relativedelta(weekday=TH(+4)) + timedelta(days=1)) : "Black Friday"}) # Adding Black Friday
# Accessing custom created calendar



#### Un-comment to see list

# for date, name in sorted(CorporateHolidays(years=[2018,2019,2020]).items()):

#     print(date, name) 
# Setup 

holiday_table = pd.DataFrame(columns=['date', 'holiday_name']).set_index('date')



# CorporateHolidays(years=[2018,2019,2020]) Structure

for date, name in sorted(CorporateHolidays(years=[2018,2019,2020]).items()):

    holiday_table = holiday_table.append({'date': pd.to_datetime(date), 'holiday_name':name}, ignore_index=True)
holiday_table = holiday_table.set_index('date')

holiday_table_dates = holiday_table.index.to_list()
data_date_range = pd.date_range(start=date_range_start, end=date_range_end)
business_data_date_range = pd.date_range(start=date_range_start, end=date_range_end, freq='B')

business_dates = business_data_date_range[~business_data_date_range.isin(holiday_table_dates)]
df = pd.DataFrame({'date': data_date_range}).set_index('date')
# Join holiday table to base date table

df = df.join(holiday_table, on='date').fillna("")
##### Add additional date columns



# Holiday boolean

df['holiday'] = df['holiday_name'].apply(lambda x: 0 if x == "" else 1)



# Day index

df['day_index'] = df.index.day



# Weekday Name

df['weekday_name'] = df.index.weekday_name



# Business day

df['business_day'] = df.index.isin(business_dates).astype(int)
df